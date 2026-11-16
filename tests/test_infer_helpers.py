"""Lightweight tests: no model weights, torch imports or inference required."""
from pathlib import Path
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("release_infer", ROOT / "infer.py")
infer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(infer)


class SplitTextTests(unittest.TestCase):
    def test_preserves_unicode_punctuation_and_whitespace(self):
        text = "  第一段。\n“第二段，留个停顿！”\n\n最后一句，没有句号 \t"
        parts = infer.split_text(text, 12)
        self.assertEqual("".join(parts), text)
        self.assertTrue(all(0 < len(part) <= 12 for part in parts))

    def test_closing_quote_stays_with_its_sentence(self):
        self.assertEqual(infer.split_text("他说：“你好！”", 60), ["他说：“你好！”"])

    def test_no_punctuation_drops_no_characters(self):
        text = "连续文字" * 43
        parts = infer.split_text(text, 30)
        self.assertEqual("".join(parts), text)
        self.assertTrue(all(len(part) <= 30 for part in parts))

    def test_each_complete_sentence_can_receive_pause(self):
        self.assertEqual(infer.split_text("先读资料。然后运行！再检查？"),
                         ["先读资料。", "然后运行！", "再检查？"])

    def test_empty_narration_is_rejected(self):
        for text in ("", " \t\n"):
            with self.assertRaisesRegex(ValueError, "empty"):
                infer.split_text(text)

    def test_dynamic_tokens_have_floor_and_ceiling(self):
        self.assertEqual(infer.token_limit("你好", 384), 64)
        self.assertEqual(infer.token_limit("你" * 120, 384), 384)


class GenerationGuardTests(unittest.TestCase):
    def test_actual_generation_call_receives_and_executes_current_guard(self):
        calls = []
        old_criterion = lambda *_args, **_kwargs: False
        first_guard = lambda *_args, **_kwargs: True
        second_guard = lambda *_args, **_kwargs: False
        active = {"guard": first_guard}

        class FakeCriteriaList(list):
            pass

        def fake_talker_generate(*args, **kwargs):
            criteria = kwargs["stopping_criteria"]
            calls.append((args, kwargs))
            self.assertIsInstance(criteria, FakeCriteriaList)
            return any(criterion("token-ids", None) for criterion in criteria)

        generate = infer.guard_talker_generate(fake_talker_generate, active, FakeCriteriaList)
        existing = FakeCriteriaList([old_criterion])
        self.assertTrue(generate("prompt", stopping_criteria=existing, max_new_tokens=64))
        self.assertEqual(calls[0][0], ("prompt",))
        self.assertEqual(calls[0][1]["max_new_tokens"], 64)
        self.assertEqual(calls[0][1]["stopping_criteria"], [old_criterion, first_guard])
        self.assertEqual(existing, [old_criterion])  # The caller's list stays unchanged.
        active["guard"] = second_guard
        self.assertFalse(generate("next prompt"))
        self.assertEqual(calls[1][1]["stopping_criteria"], [second_guard])

    def test_generation_without_a_guard_never_reaches_talker(self):
        def unexpected_generate(*_args, **_kwargs):
            self.fail("unguarded generation must not be called")

        generate = infer.guard_talker_generate(unexpected_generate, {}, list)
        with self.assertRaisesRegex(RuntimeError, "No active segment"):
            generate()


class InputAndPathTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name)
        self.parser = infer.build_parser()

    def tearDown(self):
        self.tmp.cleanup()

    def args(self, *extra):
        return self.parser.parse_args(["--output", str(self.folder / "new.wav"), *extra])

    def test_cpu_default_and_short_test_text(self):
        args = self.args()
        infer.validate_arguments(args)
        self.assertEqual(args.mode, "cpu")
        self.assertEqual(args.speaker, "user_speaker")
        self.assertEqual(infer.read_text(args), infer.DEFAULT_TEXT)

    def test_pause_and_other_nonfinite_numbers_are_rejected(self):
        for flag, value in (("--pause", "nan"), ("--pause", "-0.1"),
                            ("--pause", "inf"), ("--segment-timeout", "nan"),
                            ("--gpu-budget-gib", "nan")):
            with self.subTest(flag=flag, value=value), self.assertRaises(ValueError):
                infer.validate_arguments(self.args(flag, value))

    def test_empty_explicit_text_does_not_silently_use_sample(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            infer.read_text(self.args("--text", ""))

    def test_bom_file_and_missing_input(self):
        file = self.folder / "稿件.txt"
        file.write_text("这是实际文本。", encoding="utf-8-sig")
        self.assertEqual(infer.read_text(self.args("--text-file", str(file))), "这是实际文本。")
        with self.assertRaises(FileNotFoundError):
            infer.read_text(self.args("--text-file", str(self.folder / "missing.txt")))

    def test_existing_output_is_preserved(self):
        output = self.folder / "new.wav"
        output.write_bytes(b"keep this")
        with self.assertRaises(FileExistsError):
            infer.validate_arguments(self.args())
        self.assertEqual(output.read_bytes(), b"keep this")

    def test_model_directory_requires_decoder_and_config(self):
        with self.assertRaises(FileNotFoundError):
            infer.validate_model_directory(self.folder / "missing", "user_speaker")
        with self.assertRaisesRegex(FileNotFoundError, "speech_tokenizer"):
            infer.validate_model_directory(self.folder, "user_speaker")

    def test_lfs_pointer_is_not_weights(self):
        model = self.folder / "model"
        for name in infer.REQUIRED_FILES:
            file = model / name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(b"{}")
        (model / "model.safetensors").write_text(
            "version https://git-lfs.github.com/spec/v1\noid sha256:dummy\n", encoding="ascii")
        with self.assertRaisesRegex(ValueError, "LFS pointer"):
            infer.validate_model_directory(model, "user_speaker")

    def test_commit_does_not_replace_an_existing_file(self):
        temporary = self.folder / "temporary.wav"
        target = self.folder / "target.wav"
        temporary.write_bytes(b"new waveform")
        target.write_bytes(b"existing waveform")
        with self.assertRaises(FileExistsError):
            infer.commit_new_file(temporary, target)
        self.assertEqual(target.read_bytes(), b"existing waveform")

    def test_help_and_bad_text_work_without_site_packages(self):
        for options, expected in ((["--help"], 0), (["--text", ""], 2)):
            run = subprocess.run(
                [sys.executable, "-S", str(ROOT / "infer.py"), *options],
                capture_output=True, text=True, timeout=10,
            )
            self.assertEqual(run.returncode, expected, run.stderr)
            self.assertNotIn("No module named", run.stderr)


if __name__ == "__main__":
    unittest.main()
