#!/usr/bin/env python3
"""
Test cases for the main module.
"""

import unittest
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.main import hello_world


class TestMain(unittest.TestCase):
    """Test cases for main module functions."""

    def test_hello_world(self):
        """Test the hello_world function."""
        result = hello_world()
        self.assertEqual(result, "Hello, World! Welcome to PROTO!")
        self.assertIsInstance(result, str)

    def test_hello_world_not_empty(self):
        """Test that hello_world returns a non-empty string."""
        result = hello_world()
        self.assertTrue(len(result) > 0)


if __name__ == "__main__":
    unittest.main()