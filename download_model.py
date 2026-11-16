#!/usr/bin/env python3
"""Download the released voice checkpoint with streaming SHA-256 verification.

Only Python's standard library is required. Model inference dependencies are
separate. Interrupted asset downloads remain in a manifest-scoped cache.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import http.client
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import time
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen

BUFFER_SIZE = 8 * 1024 * 1024
CACHE_NAME = ".download-cache"
LOCK_NAME = ".model-download.lock"
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
REPOSITORY_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
RANGE_RE = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$")
WINDOWS_RESERVED = {"CON", "PRN", "AUX", "NUL"} | {
    f"{prefix}{number}" for prefix in ("COM", "LPT") for number in range(1, 10)
}


class DownloadError(RuntimeError):
    """A recoverable download, integrity, or destination error."""


class ManifestError(ValueError):
    """The release manifest is invalid or unsafe."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(BUFFER_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def _size(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ManifestError(f"{context}: size must be a non-negative integer")
    return value


def _digest(value: object, context: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise ManifestError(f"{context}: expected a SHA-256 digest")
    return value.lower()


def _relative_path(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ManifestError("Every model file needs a nonempty relative path")
    if "\\" in value or ":" in value or any(ord(c) < 32 for c in value):
        raise ManifestError(f"Unsafe model path: {value!r}")
    parts = value.split("/")
    if any(not p or p in {".", ".."} or p.endswith((" ", ".")) for p in parts):
        raise ManifestError(f"Unsafe model path: {value!r}")
    if PurePosixPath(value).is_absolute():
        raise ManifestError(f"Absolute model path is not allowed: {value!r}")
    if parts[0].casefold() in {CACHE_NAME.casefold(), LOCK_NAME.casefold()}:
        raise ManifestError("A model path conflicts with downloader state")
    if any(p.split(".")[0].upper() in WINDOWS_RESERVED for p in parts):
        raise ManifestError(f"Reserved Windows device path: {value!r}")
    return value


def validate_manifest(data: object) -> dict:
    """Validate all paths and asset relationships before any network/write."""
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ManifestError("Expected a manifest with schema_version=1")
    repository = data.get("repository")
    if not isinstance(repository, str) or not REPOSITORY_RE.fullmatch(repository):
        raise ManifestError("Expected repository in owner/name form")
    release = data.get("release")
    if not isinstance(release, str) or release in {"", ".", ".."}:
        raise ManifestError("Expected a nonempty release tag")
    if any(ord(c) < 32 for c in release):
        raise ManifestError("Invalid control character in release tag")
    files = data.get("files")
    if not isinstance(files, list) or not files:
        raise ManifestError("Manifest files must be a nonempty list")
    normalized, paths, assets = [], set(), {}
    for record in files:
        if not isinstance(record, dict):
            raise ManifestError("Each file entry must be an object")
        path = _relative_path(record.get("path"))
        key = path.casefold()
        if key in paths:
            raise ManifestError(f"Duplicate model path: {path}")
        paths.add(key)
        file_size = _size(record.get("size"), path)
        file_sha = _digest(record.get("sha256"), path)
        parts = record.get("parts")
        if not isinstance(parts, list) or not parts:
            raise ManifestError(f"{path}: parts must be a nonempty list")
        clean_parts = []
        for part in parts:
            if not isinstance(part, dict):
                raise ManifestError(f"{path}: every part must be an object")
            asset = part.get("asset")
            if (not isinstance(asset, str) or not asset or asset in {".", ".."}
                    or any(c in asset for c in "/\\:")
                    or any(ord(c) < 32 for c in asset)):
                raise ManifestError(f"{path}: unsafe release asset name")
            part_size = _size(part.get("size"), asset)
            part_sha = _digest(part.get("sha256"), asset)
            identity = (part_size, part_sha)
            if asset in assets and assets[asset] != identity:
                raise ManifestError(f"Inconsistent metadata for asset: {asset}")
            assets[asset] = identity
            clean_parts.append({"asset": asset, "size": part_size, "sha256": part_sha})
        if sum(p["size"] for p in clean_parts) != file_size:
            raise ManifestError(f"{path}: part sizes do not sum to file size")
        if len(clean_parts) == 1 and clean_parts[0]["sha256"] != file_sha:
            raise ManifestError(f"{path}: single-part and file digests disagree")
        normalized.append({"path": path, "size": file_size, "sha256": file_sha,
                           "parts": clean_parts})
    for path in paths:
        parent = PurePosixPath(path).parent
        while str(parent) != ".":
            if str(parent) in paths:
                raise ManifestError("A model path is also used as a directory")
            parent = parent.parent
    return {**data, "files": normalized}


def load_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as handle:
        return validate_manifest(json.load(handle))


def _safe_path(root: Path, relative: str) -> Path:
    candidate = root.joinpath(*PurePosixPath(relative).parts)
    current = root
    for part in PurePosixPath(relative).parts:
        current = current / part
        if current.is_symlink():
            raise DownloadError(f"Refusing symlink inside model directory: {relative}")
    if not candidate.resolve().is_relative_to(root.resolve()):
        raise DownloadError(f"Destination escapes model directory: {relative}")
    return candidate


def _matches(path: Path, size: int, digest: str) -> bool:
    return path.is_file() and path.stat().st_size == size and sha256_file(path) == digest


def verify_model(manifest: dict, output: Path) -> list[str]:
    """Read-only full-file verification. Return one message per failed file."""
    manifest = validate_manifest(manifest)
    output = Path(output).resolve()
    problems = []
    for item in manifest["files"]:
        target = _safe_path(output, item["path"])
        if not target.is_file():
            problems.append(f"Missing: {item['path']}")
        elif target.stat().st_size != item["size"]:
            problems.append(f"Size mismatch: {item['path']}")
        elif sha256_file(target) != item["sha256"]:
            problems.append(f"SHA-256 mismatch: {item['path']}")
    return problems


@contextmanager
def _output_lock(output: Path):
    """An OS-released advisory lock, including after an interrupted process."""
    lock_path = _safe_path(output, LOCK_NAME)
    handle = lock_path.open("a+b")
    locked = False
    try:
        if handle.seek(0, os.SEEK_END) == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except OSError as error:
            raise DownloadError("Another downloader is using this model directory") from error
        yield
    finally:
        if locked:
            if os.name == "nt":
                import msvcrt
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _download_attempt(url: str, part: dict, cache_file: Path, timeout: float) -> None:
    offset = cache_file.stat().st_size if cache_file.exists() else 0
    headers = {"User-Agent": "Beijing-young-men-tts-model/1.0",
               "Accept-Encoding": "identity"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = Request(url, headers=headers)
    try:
        response = urlopen(request, timeout=timeout)
    except HTTPError as error:
        # Do not print redirect URLs or headers, which may contain signed tokens.
        code = error.code
        error.close()
        raise DownloadError(f"HTTP {code} for asset {part['asset']}") from None
    with response:
        status = response.getcode()
        if status == 206:
            match = RANGE_RE.fullmatch(response.headers.get("Content-Range", ""))
            if not match:
                cache_file.unlink(missing_ok=True)
                raise DownloadError(f"Invalid Content-Range for {part['asset']}")
            start, end, total = map(int, match.groups())
            if start != offset or total != part["size"] or not (start <= end < total):
                cache_file.unlink(missing_ok=True)
                raise DownloadError(f"Range response does not match {part['asset']}")
            mode = "ab" if offset else "wb"
            expected_response = end - start + 1
        elif status == 200:
            # A server may ignore Range. Start from zero instead of appending.
            offset, mode, expected_response = 0, "wb", part["size"]
        else:
            raise DownloadError(f"Unexpected HTTP status {status} for {part['asset']}")
        length = response.headers.get("Content-Length")
        if length is not None:
            try:
                actual_length = int(length)
            except ValueError:
                raise DownloadError(f"Invalid Content-Length for {part['asset']}") from None
            if actual_length != expected_response:
                raise DownloadError(f"Response size does not match {part['asset']}")
        written = offset
        with cache_file.open(mode) as destination:
            while True:
                block = response.read(BUFFER_SIZE)
                if not block:
                    break
                if written + len(block) > part["size"]:
                    raise DownloadError(f"Response exceeds expected asset size: {part['asset']}")
                destination.write(block)
                written += len(block)
            destination.flush()
            os.fsync(destination.fileno())
        if written != part["size"]:
            raise DownloadError(f"Incomplete asset {part['asset']}; partial download retained")


def _fetch_part(part: dict, cache_dir: Path, base_url: str, timeout: float,
                retries: int, report: Callable[[str], None] | None) -> Path:
    cache_file = cache_dir / (part["sha256"] + ".part")
    if cache_file.is_symlink():
        raise DownloadError("Refusing symlink in download cache")
    if _matches(cache_file, part["size"], part["sha256"]):
        if report:
            report(f"Using verified cache: {part['asset']}")
        return cache_file
    if cache_file.exists() and cache_file.stat().st_size >= part["size"]:
        cache_file.unlink()
    url = base_url.rstrip("/") + "/" + quote(part["asset"], safe="")
    for attempt in range(1, retries + 1):
        offset = cache_file.stat().st_size if cache_file.exists() else 0
        if report:
            report(f"Downloading {part['asset']} ({offset:,}/{part['size']:,} bytes cached)")
        try:
            _download_attempt(url, part, cache_file, timeout)
            if not _matches(cache_file, part["size"], part["sha256"]):
                cache_file.unlink(missing_ok=True)
                raise DownloadError(f"SHA-256 mismatch for asset {part['asset']}")
            return cache_file
        except (DownloadError, URLError, OSError, http.client.HTTPException) as error:
            # Retain a truncated prefix for Range resume, discard complete bad data.
            if cache_file.exists() and cache_file.stat().st_size >= part["size"]:
                cache_file.unlink()
            if attempt == retries:
                if isinstance(error, DownloadError):
                    raise
                raise DownloadError(f"Download interrupted for asset {part['asset']}; "
                                    "run the same command again to resume") from None
            if report:
                report(f"Retrying {part['asset']} (attempt {attempt + 1}/{retries})")
            time.sleep(min(attempt, 3))
    raise AssertionError("unreachable")


def _clean_parts(item: dict, cache_dir: Path) -> None:
    for part in item["parts"]:
        path = cache_dir / (part["sha256"] + ".part")
        if path.is_file() and not path.is_symlink():
            path.unlink()


def download_model(manifest: dict, output: Path, *, keep_cache: bool = False,
                   timeout: float = 30, retries: int = 3, base_url: str | None = None,
                   report: Callable[[str], None] | None = print) -> None:
    """Download and verify each file, then atomically place it in output."""
    manifest = validate_manifest(manifest)
    if timeout <= 0 or retries < 1:
        raise ValueError("timeout must be positive and retries must be at least 1")
    if base_url is None:
        base_url = ("https://github.com/" + manifest["repository"]
                    + "/releases/download/" + quote(manifest["release"], safe=""))
    if urlsplit(base_url).scheme not in {"http", "https"}:
        raise ValueError("Download base URL must use HTTP or HTTPS")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256(json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")).hexdigest()[:24]
    cache_dir = _safe_path(output, CACHE_NAME + "/" + fingerprint)
    with _output_lock(output):
        cache_dir.mkdir(parents=True, exist_ok=True)
        for item in manifest["files"]:
            target = _safe_path(output, item["path"])
            if target.exists():
                if not _matches(target, item["size"], item["sha256"]):
                    raise DownloadError(f"Existing file does not match manifest: {item['path']}. "
                                        "Move it aside before downloading; it was not overwritten.")
                if report:
                    report(f"Verified existing file: {item['path']}")
                if not keep_cache:
                    _clean_parts(item, cache_dir)
                continue
            pieces = [_fetch_part(part, cache_dir, base_url, timeout, retries, report)
                      for part in item["parts"]]
            target.parent.mkdir(parents=True, exist_ok=True)
            staging = cache_dir / (item["sha256"] + ".assembling")
            if staging.is_symlink():
                raise DownloadError("Refusing symlink in download staging")
            # A single-part file can be moved directly after full validation.
            if len(pieces) == 1 and not keep_cache:
                staging = pieces[0]
            else:
                with staging.open("wb") as destination:
                    digest = hashlib.sha256()
                    total = 0
                    for piece in pieces:
                        with piece.open("rb") as source:
                            for block in iter(lambda: source.read(BUFFER_SIZE), b""):
                                destination.write(block)
                                digest.update(block)
                                total += len(block)
                    destination.flush()
                    os.fsync(destination.fileno())
                if total != item["size"] or digest.hexdigest() != item["sha256"]:
                    staging.unlink(missing_ok=True)
                    raise DownloadError(f"Assembled file checksum mismatch: {item['path']}")
            # Recheck the path and existence after network/assembly, before publish.
            target = _safe_path(output, item["path"])
            if target.exists():
                raise DownloadError(f"Destination appeared during download: {item['path']}")
            os.replace(staging, target)
            if not keep_cache:
                _clean_parts(item, cache_dir)
            if report:
                report(f"Ready: {item['path']}")
        if not keep_cache:
            try:
                cache_dir.rmdir()
                cache_dir.parent.rmdir()
            except OSError:
                pass  # Preserve any other manifest's pending downloads.


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=Path(__file__).resolve().with_name("model-manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("model"),
                        help="Model directory (default: ./model)")
    parser.add_argument("--verify-only", action="store_true",
                        help="Check every final file without downloading or writing")
    parser.add_argument("--keep-cache", action="store_true",
                        help="Keep verified downloaded parts after successful assembly")
    parser.add_argument("--timeout", type=float, default=30,
                        help="Network read timeout in seconds (default: 30)")
    parser.add_argument("--retries", type=int, default=3,
                        help="Download attempts per asset (default: 3)")
    args = parser.parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        if args.verify_only:
            problems = verify_model(manifest, args.output)
            if problems:
                for problem in problems:
                    print(problem, file=sys.stderr)
                return 1
            print(f"Verified all {len(manifest['files'])} model files in {args.output}")
        else:
            download_model(manifest, args.output, keep_cache=args.keep_cache,
                           timeout=args.timeout, retries=args.retries)
            print(f"Model ready: {args.output.resolve()}")
        return 0
    except KeyboardInterrupt:
        print("Interrupted. Download cache retained; rerun the same command to resume.",
              file=sys.stderr)
        return 130
    except (ManifestError, DownloadError, ValueError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
