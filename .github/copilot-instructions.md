# PROTO Project Instructions

## Project Setup Checklist

- [x] ✅ Verify that the copilot-instructions.md file in the .github directory is created.
- [x] ✅ Clarify Project Requirements: Python project for prototyping and development.
- [x] ✅ Scaffold the Project: Created complete Python project structure with src/, tests/, docs/ directories.
- [x] ✅ Customize the Project: Set up Hello World project as starting point.
- [x] ✅ Install Required Extensions: No extensions needed (skipped).
- [x] ✅ Compile the Project: Installed dependencies and verified functionality.
- [x] ✅ Create and Run Task: Created "Run PROTO" task in VS Code.
- [x] ✅ Launch the Project: Project ready to run with task or directly.
- [x] ✅ Ensure Documentation is Complete: README.md and instructions updated.

## Project Overview

**PROTO** is a Python project set up for prototyping and development with the following structure:

```
PROTO/
├── src/                    # Source code
│   ├── __init__.py
│   └── main.py            # Main application entry point
├── tests/                  # Test files
│   ├── __init__.py
│   └── test_main.py       # Unit tests
├── docs/                   # Documentation
├── .github/                # GitHub configuration
│   └── copilot-instructions.md
├── .gitignore             # Git ignore rules
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
└── setup.py              # Package setup
```

## Quick Start

1. **Run the project**: Use the "Run PROTO" task in VS Code or execute `python src/main.py`
2. **Run tests**: Execute `python -m pytest tests/ -v`
3. **Virtual environment**: Already created in `venv/` directory

## Development Guidelines

- Keep communication concise and focused
- Follow Python best practices
- Use the virtual environment for all development
- Add new dependencies to `requirements.txt`
- Write tests for new functionality in `tests/` directory
