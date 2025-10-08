#!/usr/bin/env python3
"""Integration tests for LittleFS extraction helpers."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.proto_example import (  # noqa: E402  - added to sys.path dynamically
    analyze_memory_for_littlefs,
    process_memory_dump_with_littlefs,
    read_binary_file,
)


def test_analyze_memory_for_littlefs_detects_filesystem() -> None:
    dump_bytes = read_binary_file("src/bin/memory_dump.bin")
    analysis = analyze_memory_for_littlefs(dump_bytes, block_sizes=(4096,))
    assert analysis["successful_mounts"], "Expected at least one successful mount"

    first_mount = analysis["successful_mounts"][0]
    assert first_mount["block_size"] == 4096
    assert "home" in first_mount["root_entries"]


def test_process_memory_dump_with_littlefs_extracts_files(tmp_path: Path) -> None:
    output_dir = tmp_path / "extracted"
    result = process_memory_dump_with_littlefs("src/bin/memory_dump.bin", output_dir=output_dir)

    assert result["extracted_files"], "No files extracted from LittleFS image"

    for entry in result["extracted_files"][:3]:
        assert Path(entry["output_path"]).exists()
