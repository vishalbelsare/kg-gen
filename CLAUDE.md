# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build/Run/Test Commands
- Install: `pip install -e .`
- Run tests with pytest: `pytest tests/`
- Run single pytest test: `pytest tests/test_file.py::test_function_name`
- Run specific test file directly: `python tests/test_file.py`
- Import and use: `from kg_gen import KGGen`

## Code Style Guidelines
- Type hints: Use Python type annotations (Union, List, Dict, Optional)
- Imports: Standard library first, then third-party, then local imports
- Formatting: 2-space indentation, no trailing whitespace
- Naming: snake_case for functions/variables, PascalCase for classes
- Models: Use Pydantic models for structured data validation
- DSPy integration: Follow DSPy module patterns for LLM interaction
- Error handling: Validate inputs with appropriate error messages
- Documentation: Document classes with docstrings, include param descriptions
- Testing: Write pytest tests for new functionality