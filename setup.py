#!/usr/bin/env python3
"""
Setup script for PROTO project.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="proto",
    version="0.1.0",
    author="Mario Gómez-Barea",
    author_email="m.gomez@onalabs.com",
    description="A Python project for the extraction of binary protobuf waveform files.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/proto",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "proto=src.main:main",
        ],
    },
)