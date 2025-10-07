#!/usr/bin/env python3
"""
Example of how to use LittleFS extraction functionality.
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.proto_example import (
    test_littlefs_extraction,
    process_memory_dump_with_littlefs,
    analyze_memory_for_littlefs,
    read_binary_file
)


def main():
    """
    Example usage of LittleFS functionality.
    """
    print("=== LittleFS Integration Example ===\n")
    
    print("Available functions:")
    print("1. test_littlefs_extraction(filename) - Simple test function")
    print("2. process_memory_dump_with_littlefs(filename) - Full processing")
    print("3. analyze_memory_for_littlefs(data) - Analyze memory for LittleFS signatures")
    print("4. read_binary_file(filename) - Read and analyze binary files")
    
    print("\nUsage examples:")
    print("# Test LittleFS extraction from a memory dump")
    print("test_littlefs_extraction('memory_dump.bin')")
    
    print("\n# Full processing with analysis")
    print("results = process_memory_dump_with_littlefs('flash_dump.bin')")
    print("print(results['analysis'])")
    print("print(results['extracted_files'])")
    
    print("\n# Analyze memory data")
    print("data = read_binary_file('raw_memory.bin')")
    print("analysis = analyze_memory_for_littlefs(data)")
    
    print("\nSupported file formats:")
    print("- Raw memory dumps from embedded devices")
    print("- Flash memory dumps containing LittleFS")
    print("- Variable memory regions with filesystem data")
    
    print("\nFile placement:")
    print("Place your memory dump files in src/bin/ directory")
    print("Supported names: memory_dump.bin, flash_dump.bin, littlefs_dump.bin")
    
    # Check if any example files exist
    example_files = [
        "src/bin/memory_dump.bin",
        "src/bin/flash_dump.bin", 
        "src/bin/littlefs_dump.bin"
    ]
    
    found_files = [f for f in example_files if os.path.exists(f)]
    
    if found_files:
        print(f"\nFound {len(found_files)} memory dump file(s):")
        for file in found_files:
            print(f"  - {file}")
            
        print("\nRunning test extraction on first file...")
        test_littlefs_extraction(found_files[0])
    else:
        print("\nNo memory dump files found.")
        print("To test the functionality, place a LittleFS memory dump in src/bin/")


if __name__ == "__main__":
    main()