"""Small local-HTTP tests. No model packages, public downloads, or GPU needed."""
from contextlib import contextmanager, redirect_stderr
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock
from urllib.parse import unquote, urlsplit
from urllib.request import ProxyHandler, build_opener
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import download_model as downloader


def digest(data):
    return sha256(data).hexdigest()


def fixture_manifest(payload, chunks=None, path="model.safetensors"):
    chunks = chunks or [payload]
    parts, assets = [], {}
    for index, chunk in enumerate(chunks):
        asset = f"weights.part-{index + 1:03d}"
        assets[asset] = chunk
        parts.append({"asset": asset, "size": len(chunk), "sha256": digest(chunk)})
    manifest = {
        "schema_version": 1,
        "repository": "Hewei603/Beijing-young-men-tts-model",
        "release": "v1.0.0",
        "speaker": "user_speaker",
        "files": [{"path": path, "size": len(payload), "sha256": digest(payload),
                   "parts": parts}],
    }
    return manifest, assets


@contextmanager
def local_assets(assets):
    state = {"assets": assets, "requests": [], "truncate_once": set(),
             "ignore_range": False, "wrong_range": False}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_GET(self):
            asset = unquote(urlsplit(self.path).path.rsplit("/", 1)[-1])
            requested_range = self.headers.get("Range")
            state["requests"].append((asset, requested_range))
            if asset not in state["assets"]:
                self.send_error(404)
                return
            data = state["assets"][asset]
            offset = 0
            status = 200
            if requested_range and not state["ignore_range"]:
                offset = int(requested_range.removeprefix("bytes=").removesuffix("-"))
                status = 206
            body = data[offset:]
            self.send_response(status)
            if status == 206:
                start = offset + 1 if state["wrong_range"] else offset
                self.send_header("Content-Range", f"bytes {start}-{len(data)-1}/{len(data)}")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            if asset in state["truncate_once"]:
                state["truncate_once"].remove(asset)
                body = body[:max(1, len(body) // 2)]
            try:
                self.wfile.write(body)
                self.wfile.flush()
            except (ConnectionResetError, BrokenPipeError):
                pass
            self.close_connection = True

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield state, f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=3)


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.output = self.root / "model"
        self.direct_http = mock.patch.object(
            downloader, "urlopen", build_opener(ProxyHandler({})).open
        )
        self.direct_http.start()

    def tearDown(self):
        self.direct_http.stop()
        self.temp.cleanup()

    def download(self, manifest, base_url, **kwargs):
        return downloader.download_model(
            manifest, self.output, base_url=base_url, report=None, retries=1, **kwargs
        )

    def test_multipart_model_is_assembled_verified_and_cache_removed(self):
        payload = b"model-prefix" * 1700 + b"model-suffix" * 1900
        cut = 20000
        manifest, assets = fixture_manifest(payload, [payload[:cut], payload[cut:]],
                                            path="speech_tokenizer/model.safetensors")
        with local_assets(assets) as (state, url):
            self.download(manifest, url)
        target = self.output / "speech_tokenizer/model.safetensors"
        self.assertEqual(target.read_bytes(), payload)
        self.assertEqual(downloader.verify_model(manifest, self.output), [])
        self.assertEqual(len(state["requests"]), 2)
        self.assertFalse((self.output / downloader.CACHE_NAME).exists())

    def test_interrupted_download_resumes_with_http_range(self):
        payload = b"0123456789abcdef" * 4096
        manifest, assets = fixture_manifest(payload)
        with local_assets(assets) as (state, url):
            state["truncate_once"].add("weights.part-001")
            with self.assertRaises(downloader.DownloadError):
                self.download(manifest, url)
            self.assertFalse((self.output / "model.safetensors").exists())
            partial = list((self.output / downloader.CACHE_NAME).rglob("*.part"))
            self.assertEqual(len(partial), 1)
            partial_size = partial[0].stat().st_size
            self.assertGreater(partial_size, 0)
            self.assertLess(partial_size, len(payload))
            self.download(manifest, url)
        self.assertEqual(state["requests"][-1][1], f"bytes={partial_size}-")
        self.assertEqual((self.output / "model.safetensors").read_bytes(), payload)

    def test_server_ignoring_range_restarts_without_duplicate_prefix(self):
        payload = b"abcdefghijklmnopqrstuvwxyz" * 2500
        manifest, assets = fixture_manifest(payload)
        with local_assets(assets) as (state, url):
            state["truncate_once"].add("weights.part-001")
            with self.assertRaises(downloader.DownloadError):
                self.download(manifest, url)
            state["ignore_range"] = True
            self.download(manifest, url)
        self.assertIsNotNone(state["requests"][-1][1])
        self.assertEqual((self.output / "model.safetensors").read_bytes(), payload)

    def test_wrong_content_range_never_becomes_a_model_file(self):
        payload = b"0123456789" * 3000
        manifest, assets = fixture_manifest(payload)
        with local_assets(assets) as (state, url):
            state["truncate_once"].add("weights.part-001")
            with self.assertRaises(downloader.DownloadError):
                self.download(manifest, url)
            state["wrong_range"] = True
            with self.assertRaisesRegex(downloader.DownloadError, "Range response"):
                self.download(manifest, url)
        self.assertFalse((self.output / "model.safetensors").exists())

    def test_bad_asset_checksum_is_rejected_and_final_path_stays_absent(self):
        payload = b"the expected checkpoint"
        manifest, assets = fixture_manifest(payload)
        assets["weights.part-001"] = b"x" * len(payload)
        with local_assets(assets) as (_, url):
            with self.assertRaisesRegex(downloader.DownloadError, "SHA-256"):
                self.download(manifest, url)
        self.assertFalse((self.output / "model.safetensors").exists())
        self.assertEqual(list((self.output / downloader.CACHE_NAME).rglob("*.part")), [])

    def test_final_multipart_checksum_is_checked_independently(self):
        payload = b"first-half" + b"second-half"
        manifest, assets = fixture_manifest(payload, [b"first-half", b"second-half"])
        manifest["files"][0]["sha256"] = "0" * 64
        with local_assets(assets) as (_, url):
            with self.assertRaisesRegex(downloader.DownloadError, "Assembled file"):
                self.download(manifest, url)
        self.assertFalse((self.output / "model.safetensors").exists())
        self.assertEqual(list((self.output / downloader.CACHE_NAME).rglob("*.assembling")), [])

    def test_existing_invalid_file_is_never_overwritten(self):
        payload = b"expected"
        manifest, assets = fixture_manifest(payload)
        self.output.mkdir()
        target = self.output / "model.safetensors"
        target.write_bytes(b"the user's existing file")
        with local_assets(assets) as (state, url):
            with self.assertRaisesRegex(downloader.DownloadError, "not overwritten"):
                self.download(manifest, url)
        self.assertEqual(target.read_bytes(), b"the user's existing file")
        self.assertEqual(state["requests"], [])

    def test_verified_existing_file_needs_no_network(self):
        payload = b"expected"
        manifest, assets = fixture_manifest(payload)
        self.output.mkdir()
        target = self.output / "model.safetensors"
        target.write_bytes(payload)
        with local_assets(assets) as (state, url):
            self.download(manifest, url)
        self.assertEqual(state["requests"], [])
        self.assertEqual(target.read_bytes(), payload)

    def test_keep_cache_reuses_verified_parts_after_failed_or_new_output(self):
        payload = b"part-one" + b"part-two"
        manifest, assets = fixture_manifest(payload, [b"part-one", b"part-two"])
        with local_assets(assets) as (state, url):
            self.download(manifest, url, keep_cache=True)
            self.assertEqual(len(state["requests"]), 2)
            self.assertEqual(len(list((self.output / downloader.CACHE_NAME).rglob("*.part"))), 2)
            (self.output / "model.safetensors").unlink()
            self.download(manifest, url, keep_cache=True)
            self.assertEqual(len(state["requests"]), 2)
        self.assertEqual((self.output / "model.safetensors").read_bytes(), payload)

    def test_manifest_paths_are_validated_before_output_or_network(self):
        manifest, _ = fixture_manifest(b"sample")
        invalid = ["../escape", "/absolute", "a/../escape", "a\\escape",
                   "C:/absolute", "a//b", "a/./b", "NUL", "CON.txt",
                   ".download-cache/payload"]
        for path in invalid:
            with self.subTest(path=path):
                manifest["files"][0]["path"] = path
                with self.assertRaises(downloader.ManifestError):
                    self.download(manifest, "http://127.0.0.1:1")
                self.assertFalse(self.output.exists())

    def test_duplicate_or_conflicting_paths_are_rejected(self):
        manifest, _ = fixture_manifest(b"sample")
        first = manifest["files"][0]
        for second_path in ["MODEL.SAFETENSORS", "model.safetensors/child"]:
            with self.subTest(path=second_path):
                manifest["files"] = [first, {**first, "path": second_path}]
                with self.assertRaises(downloader.ManifestError):
                    downloader.validate_manifest(manifest)

    def test_verify_only_is_read_only_and_reports_missing_files(self):
        manifest, _ = fixture_manifest(b"sample")
        manifest_path = self.root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with redirect_stderr(io.StringIO()) as err:
            code = downloader.main(["--manifest", str(manifest_path),
                                    "--output", str(self.output), "--verify-only"])
        self.assertEqual(code, 1)
        self.assertIn("Missing: model.safetensors", err.getvalue())
        self.assertFalse(self.output.exists())

    def test_unknown_extra_metadata_is_preserved(self):
        manifest, _ = fixture_manifest(b"sample")
        manifest["base_model"] = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        normalized = downloader.validate_manifest(manifest)
        self.assertEqual(normalized["speaker"], "user_speaker")
        self.assertEqual(normalized["base_model"], manifest["base_model"])


if __name__ == "__main__":
    unittest.main()
