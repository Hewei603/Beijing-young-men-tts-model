#!/usr/bin/env python3
# Copyright 2026 Hewei603
# SPDX-License-Identifier: Apache-2.0
"""Local inference for the released user_speaker checkpoint.

Only standard-library modules are imported until arguments, text and model
files have been checked. --help therefore works without torch installed.
The split-device path follows the author's existing local inference layout:
CPU BF16 staging, CPU FP32 speech decoder, optional CUDA BF16 talker.
This wrapper does not reproduce the private Windows Job/process supervisor.
Its admission checks and generation stops are best-effort resource guards.
"""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time

GIB = 1024 ** 3
DEFAULT_TEXT = "你好，我来简单介绍一下这套工具的使用方法。"
START_RAM_GIB = {"cpu": 12.0, "balanced": 10.5}
RESERVE_RAM_GIB = 5.0
REQUIRED_FILES = (
    "config.json", "generation_config.json", "model.safetensors",
    "preprocessor_config.json", "tokenizer_config.json", "vocab.json",
    "merges.txt", "speech_tokenizer/config.json",
    "speech_tokenizer/model.safetensors",
)


def split_text(text: str, limit: int = 60) -> list[str]:
    """Split at sentence/clause boundaries without discarding any characters.

    Each sentence is generated separately. Long sentences prefer a comma,
    colon or whitespace near the limit, then use a hard cut as a last resort.
    Whitespace-only pieces remain represented, but inference does not speak them.
    """
    if limit < 8:
        raise ValueError("The chunk limit must be at least 8 characters.")
    if not text.strip():
        raise ValueError("The narration is empty.")
    units = re.findall(r'[^。！？!?；;\n]*[。！？!?；;\n]+[”’"」』）》】]*|[^。！？!?；;\n]+$', text)
    result: list[str] = []
    for unit in units:
        while len(unit) > limit:
            cut = max(unit.rfind(mark, 0, limit) for mark in "，、,：: \t")
            cut = cut + 1 if cut >= limit // 2 else limit
            result.append(unit[:cut])
            unit = unit[cut:]
        if unit:
            result.append(unit)
    if "".join(result) != text:
        raise ValueError("Text splitting failed to preserve the complete input.")
    return result


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate local Mandarin speech with the released user_speaker model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model-dir", type=Path, default=Path("model"),
                   help="Local model directory created by the model downloader")
    source = p.add_mutually_exclusive_group()
    source.add_argument("--text", help="Text to synthesize; omit both text options for a short test")
    source.add_argument("--text-file", type=Path, help="UTF-8 narration file")
    p.add_argument("--output", type=Path, default=Path("output.wav"),
                   help="New WAV output path; existing files are never replaced")
    p.add_argument("--mode", choices=("cpu", "balanced"), default="cpu",
                   help="cpu: all CPU; balanced: CUDA talker and CPU decoder")
    p.add_argument("--speaker", default="user_speaker")
    p.add_argument("--language", default="Chinese")
    p.add_argument("--pause", type=float, default=0.32,
                   help="Seconds of added silence between generated text segments")
    p.add_argument("--chunk-chars", type=int, default=60)
    p.add_argument("--max-new-tokens", type=int, default=384,
                   help="Upper token limit per segment; reaching it rejects that segment")
    p.add_argument("--segment-timeout", type=float, default=180.0,
                   help="Best-effort generation timeout checked between model token steps")
    p.add_argument("--gpu-index", type=int, default=0, help="Visible CUDA device index for balanced mode")
    p.add_argument("--gpu-budget-gib", type=float, default=4.5,
                   help="PyTorch allocator budget in balanced mode; not a whole-system GPU cap")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--check", action="store_true",
                   help="Check text, files, dependency versions and RAM without loading the model")
    return p


def validate_arguments(args: argparse.Namespace) -> None:
    if not math.isfinite(args.pause) or not 0 <= args.pause <= 3:
        raise ValueError("--pause must be a finite number from 0 to 3 seconds.")
    if not 8 <= args.chunk_chars <= 120:
        raise ValueError("--chunk-chars must be from 8 to 120.")
    if not 64 <= args.max_new_tokens <= 1024:
        raise ValueError("--max-new-tokens must be from 64 to 1024.")
    if not math.isfinite(args.segment_timeout) or not 1 <= args.segment_timeout <= 900:
        raise ValueError("--segment-timeout must be from 1 to 900 seconds.")
    if args.gpu_index < 0:
        raise ValueError("--gpu-index must be nonnegative.")
    if not math.isfinite(args.gpu_budget_gib) or not 3.5 <= args.gpu_budget_gib <= 16:
        raise ValueError("--gpu-budget-gib must be from 3.5 to 16.")
    if not 0 <= args.seed < 2 ** 32:
        raise ValueError("--seed must be from 0 to 4294967295.")
    if not args.speaker.strip() or not args.language.strip():
        raise ValueError("Speaker and language must not be empty.")
    if args.output.suffix.lower() != ".wav":
        raise ValueError("--output must end in .wav.")
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"Output already exists: {args.output}. Choose a new filename.")


def read_text(args: argparse.Namespace) -> str:
    if args.text_file is not None:
        try:
            value = args.text_file.read_text(encoding="utf-8-sig")
        except UnicodeError as exc:
            raise ValueError("The narration file must be saved as UTF-8.") from exc
    else:
        value = DEFAULT_TEXT if args.text is None else args.text
    if not value.strip():
        raise ValueError("The narration is empty.")
    if len(value) > 20_000:
        raise ValueError("This command accepts at most 20,000 characters. Run long material in separate jobs.")
    return value


def validate_model_directory(path: Path, speaker: str) -> dict:
    if not path.is_dir():
        raise FileNotFoundError(f"Model directory not found: {path}. Run the model downloader first.")
    missing = [name for name in REQUIRED_FILES if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError("Incomplete model directory; missing: " + ", ".join(missing))
    for name in REQUIRED_FILES:
        file = path / name
        if not file.stat().st_size:
            raise ValueError(f"Empty model file: {name}")
        if name.endswith(".safetensors"):
            with file.open("rb") as stream:
                prefix = stream.read(160)
            if prefix.startswith(b"version https://git-lfs.github.com/spec/"):
                raise ValueError(f"{name} is only a Git LFS pointer, not the model weights.")
    config = json.loads((path / "config.json").read_text("utf-8"))
    if config.get("tts_model_type") != "custom_voice":
        raise ValueError("This entry point requires the released custom_voice checkpoint.")
    names = config.get("talker_config", {}).get("spk_id", {})
    if speaker.lower() not in {name.lower() for name in names}:
        raise ValueError(f"Speaker {speaker!r} is unavailable. Available: {', '.join(sorted(names))}")
    return config


def dependency_versions() -> dict[str, str]:
    versions = {}
    for package in ("torch", "torchaudio", "qwen-tts", "transformers", "accelerate",
                    "numpy", "soundfile", "psutil", "safetensors"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                f"Missing dependency: {package}. Install the selected PyTorch build first, "
                "then run python -m pip install -r requirements.txt in an isolated environment."
            ) from exc
    required = {"qwen-tts": "0.1.1", "torch": "2.8.0", "torchaudio": "2.8.0",
                "transformers": "4.57.3", "accelerate": "1.12.0"}
    for package, expected in required.items():
        if versions[package].split("+", 1)[0] != expected:
            raise RuntimeError(f"This release uses {package}=={expected}; found {versions[package]}. "
                               "Use a separate environment with the published requirements.")
    return versions


def ram_check(psutil_module, minimum: float, phase: str) -> float:
    available = psutil_module.virtual_memory().available / GIB
    if available < minimum:
        raise RuntimeError(f"{phase}: {available:.2f} GiB available RAM; this script's conservative "
                           f"admission/reserve threshold is {minimum:.1f} GiB.")
    return available


def token_limit(text: str, ceiling: int) -> int:
    return min(ceiling, max(64, len(text.strip()) * 4 + 40))


def checked_waveform(wave, rate, numpy_module):
    values = numpy_module.asarray(wave, dtype=numpy_module.float32)
    if values.ndim > 2 or (values.ndim == 2 and 1 not in values.shape):
        raise RuntimeError("Expected a mono generated waveform.")
    values = values.reshape(-1)
    if not values.size or not numpy_module.isfinite(values).all():
        raise RuntimeError("Generated waveform is empty or contains NaN/Inf.")
    if not isinstance(rate, int) or isinstance(rate, bool) or not 8_000 <= rate <= 192_000:
        raise RuntimeError("Generated audio has an invalid sample rate.")
    peak = float(numpy_module.abs(values).max())
    if peak <= 1e-6:
        raise RuntimeError("Generated waveform is silent.")
    if peak > 1.000001:
        raise RuntimeError("Generated waveform would clip in PCM audio; no output was committed.")
    return values


def commit_new_file(temporary: Path, target: Path) -> None:
    """Publish the completed WAV without replacing an existing output."""
    try:
        os.link(temporary, target)
    except FileExistsError:
        raise
    except OSError:
        # Filesystems without hardlinks still get exclusive destination creation.
        created = False
        try:
            with target.open("xb") as output:
                created = True
                with temporary.open("rb") as source:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
        except BaseException:
            if created:
                target.unlink(missing_ok=True)
            raise


def guard_talker_generate(original_generate, active: dict, criteria_list_type):
    """Inject the current segment guard at the actual generation call.

    qwen-tts 0.1.1 drops extra generation kwargs in its outer model entry.
    Keep the list type injectable so this forwarding contract is testable
    without importing Transformers or loading weights.
    """
    def guarded_generate(*positional, **kwargs):
        guard = active.get("guard")
        if guard is None:
            raise RuntimeError("No active segment resource guard was installed.")
        existing = list(kwargs.get("stopping_criteria") or [])
        kwargs["stopping_criteria"] = criteria_list_type(existing + [guard])
        return original_generate(*positional, **kwargs)

    return guarded_generate


def synthesize(args, parts: list[str], versions: dict, psutil_module) -> dict:
    # Heavy imports are intentionally inside the actual inference entry.
    import numpy as np
    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel
    from transformers import StoppingCriteria, StoppingCriteriaList

    gpu_device = torch.device(f"cuda:{args.gpu_index}")
    if args.mode == "balanced":
        if not torch.cuda.is_available() or args.gpu_index >= torch.cuda.device_count():
            raise RuntimeError("Balanced mode requires a visible NVIDIA CUDA GPU. Use --mode cpu otherwise.")
        with torch.cuda.device(gpu_device):
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("Balanced mode requires BF16 support on the selected GPU.")
            free, total = torch.cuda.mem_get_info(gpu_device)
            if free < (args.gpu_budget_gib + 1.0) * GIB:
                raise RuntimeError(f"Balanced mode needs at least {args.gpu_budget_gib + 1.0:.1f} GiB "
                                   f"free VRAM; found {free / GIB:.2f} GiB.")
            torch.cuda.set_per_process_memory_fraction(args.gpu_budget_gib * GIB / total, gpu_device)
            # Match the existing split-device entry's conservative SDPA backend.
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)

    ram_check(psutil_module, START_RAM_GIB[args.mode], "Before model loading")
    print("Loading local checkpoint on CPU in BF16...", file=sys.stderr, flush=True)
    model = Qwen3TTSModel.from_pretrained(
        str(args.model_dir.resolve()), device_map="cpu", dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True,
    )
    tokenizer = model.model.speech_tokenizer
    tokenizer.model.to(device="cpu", dtype=torch.float32)
    tokenizer.device = torch.device("cpu")
    if args.mode == "balanced":
        model.model.talker.to(device=gpu_device, dtype=torch.bfloat16)
        model.device = gpu_device  # The wrapper puts text input tensors on this device.
    else:
        model.device = torch.device("cpu")
    model.model.eval()
    tokenizer.model.eval()
    gc.collect()
    ram_check(psutil_module, RESERVE_RAM_GIB, "After model loading")

    class ResourceStop(StoppingCriteria):
        def __init__(self):
            self.started = time.monotonic()
            self.reason = None

        def __call__(self, input_ids, scores, **kwargs):
            if time.monotonic() - self.started > args.segment_timeout:
                self.reason = "generation timeout"
            elif psutil_module.virtual_memory().available < RESERVE_RAM_GIB * GIB:
                self.reason = "available RAM reserve"
            elif args.mode == "balanced":
                free, _ = torch.cuda.mem_get_info(gpu_device)
                if free < GIB:
                    self.reason = "free VRAM reserve"
            return self.reason is not None

    # Reject a token-capped/truncated segment before waveform decoding.
    original_decode = tokenizer.decode
    original_generate = model.model.talker.generate
    active = {"guard": None, "limit": None}

    def checked_decode(encoded, **kwargs):
        if active["guard"] and active["guard"].reason:
            raise RuntimeError("Generation stopped at " + active["guard"].reason + ". No final WAV was committed.")
        if isinstance(encoded, list):
            for item in encoded:
                if len(item["audio_codes"]) >= active["limit"] - 1:
                    raise RuntimeError("Generation reached the token limit. Reduce --chunk-chars or raise "
                                       "--max-new-tokens; refusing a potentially truncated sentence.")
        return original_decode(encoded, **kwargs)

    tokenizer.decode = checked_decode
    model.model.talker.generate = guard_talker_generate(original_generate, active, StoppingCriteriaList)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix="." + output.stem + "-", suffix=".wav", dir=output.parent)
    os.close(handle)
    temporary = Path(name)
    rate = None
    total_frames = 0
    pause_frames = 0
    records = []
    started = time.monotonic()
    writer = None
    try:
        for number, part in enumerate(parts, 1):
            utterance = part.strip()
            if not utterance:
                continue
            ram_check(psutil_module, RESERVE_RAM_GIB, f"Before segment {number}")
            limit = token_limit(utterance, args.max_new_tokens)
            guard = ResourceStop()
            active.update(guard=guard, limit=limit)
            torch.manual_seed(args.seed + number - 1)
            print(f"Segment {number}/{len(parts)}: {len(utterance)} characters, token cap {limit}",
                  file=sys.stderr, flush=True)
            with torch.inference_mode():
                wavs, sample_rate = model.generate_custom_voice(
                    text=utterance, language=args.language, speaker=args.speaker,
                    max_new_tokens=limit,
                )
            if guard.reason:
                raise RuntimeError("Generation stopped at " + guard.reason + ".")
            if len(wavs) != 1:
                raise RuntimeError("Expected one generated waveform per text segment.")
            wave = checked_waveform(wavs[0], sample_rate, np)
            if rate is not None and rate != sample_rate:
                raise RuntimeError("Sample rate changed between text segments.")
            if writer is None:
                rate = sample_rate
                writer = sf.SoundFile(str(temporary), "w", samplerate=rate,
                                      channels=1, subtype="PCM_16", format="WAV")
            elif args.pause:
                silence = np.zeros(round(rate * args.pause), dtype=np.float32)
                writer.write(silence)
                total_frames += len(silence)
                pause_frames += len(silence)
            writer.write(wave)
            total_frames += len(wave)
            records.append({"characters": len(utterance), "seconds": len(wave) / rate,
                            "max_new_tokens": limit})
            del wave, wavs
            if args.mode == "balanced":
                torch.cuda.synchronize(gpu_device)
                with torch.cuda.device(gpu_device):
                    torch.cuda.empty_cache()
            gc.collect()
        if writer is None or not total_frames:
            raise RuntimeError("No audio was generated.")
        writer.close()
        writer = None
        info = sf.info(str(temporary))
        if info.frames != total_frames or info.samplerate != rate:
            raise RuntimeError("The written WAV does not match the generated audio.")
        commit_new_file(temporary, output)
    finally:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)
        tokenizer.decode = original_decode
        model.model.talker.generate = original_generate
    return {
        "output": str(output), "sample_rate": rate, "duration_seconds": total_frames / rate,
        "segments": records, "added_pause_seconds": pause_frames / rate, "mode": args.mode,
        "model_dtype": "bfloat16", "decoder_device": "cpu", "decoder_dtype": "float32",
        "speaker": args.speaker, "elapsed_generation_seconds": round(time.monotonic() - started, 2),
        "dependencies": versions,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        validate_arguments(args)
        text = read_text(args)
        parts = split_text(text, args.chunk_chars)
        validate_model_directory(args.model_dir, args.speaker)
        versions = dependency_versions()
        import psutil
        available = ram_check(psutil, START_RAM_GIB[args.mode], "Startup")
        if args.check:
            print(json.dumps({
                "model_directory": str(args.model_dir.resolve()), "text_characters": len(text),
                "segments": len(parts), "mode": args.mode, "available_ram_gib": round(available, 3),
                "model_loaded": False, "cuda_runtime_checked": False, "dependencies": versions,
            }, ensure_ascii=False, indent=2))
            return 0
        report = synthesize(args, parts, versions, psutil)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except KeyboardInterrupt:
        print("Interrupted. Check whether the output was committed before retrying.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
