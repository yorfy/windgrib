# Contributing to WindGrib

🎉 First off, thanks for taking the time to contribute! 🎉

The following is a set of guidelines for contributing to WindGrib. These are mostly guidelines, not rules. Use your best judgment, and feel free to propose changes to this document in a pull request.

## Table of Contents

- [How Can I Contribute?](#how-can-i-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Enhancements](#suggesting-enhancements)
  - [Pull Requests](#pull-requests)
- [Development Setup](#development-setup)
- [Style Guidelines](#style-guidelines)
  - [Python Style Guide](#python-style-guide)
  - [Git Commit Messages](#git-commit-messages)
- [Testing](#testing)
- [Documentation](#documentation)

## How Can I Contribute?

### Reporting Bugs

This section guides you through submitting a bug report. Following these guidelines helps maintainers and the community understand your report, reproduce the behavior, and find related reports.

**Before Submitting A Bug Report:**
- Check if the issue has already been reported
- Try to reproduce the issue with the latest version

**How Do I Submit A Good Bug Report?**

Bugs are tracked as GitHub issues. Create an issue and provide the following information:

- **Use a clear and descriptive title**
- **Describe the exact steps to reproduce the problem**
- **Provide specific examples** (code snippets, error messages)
- **Describe the behavior you observed** vs **what you expected**
- **Include screenshots** if applicable

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. Provide the following information:

- **Use a clear and descriptive title**
- **Provide a step-by-step description of the suggested enhancement**
- **Describe the current behavior** and **the desired behavior**
- **Explain why this enhancement would be useful**
- **Include examples** if possible

### Pull Requests

The process described here has several goals:
- Maintain code quality
- Fix problems that are important to users
- Engage the community in working toward the best possible solution
- Enable a sustainable system for maintainers to review contributions

**Before Submitting A Pull Request:**
1. Fork the repository and create your branch from `main`
2. If you've added code, add tests
3. Ensure the test suite passes
4. Make sure your code follows the style guidelines

**How Do I Submit A Good Pull Request?**

1. Use a clear and descriptive title
2. Include a summary of the changes
3. Reference any related issues
4. Include screenshots or animated GIFs if applicable
5. Follow the style guidelines

## Development Setup

### Prerequisites
- Python 3.7+
- Git
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/windgrib.git
cd windgrib

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=windgrib

# Run specific tests
pytest tests/test_grib.py
```

## Style Guidelines

### Python Style Guide

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide
- Use [Black](https://github.com/psf/black) for code formatting
- Use [isort](https://github.com/PyCQA/isort) for import sorting
- Use [flake8](https://github.com/PyCQA/flake8) for linting

**Formatting:**
```bash
# Format code with Black
black .

# Sort imports with isort
isort .

# Check code quality with flake8
flake8 windgrib/
```

### Git Commit Messages

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters or less
- Reference issues and pull requests liberally

**Good commit message:**
```
Add GFS model support

- Implement GFS data download functionality
- Add GFS configuration to MODELS dictionary
- Include GFS-specific data processing

Fixes #123
```

## Testing

All contributions should include tests. We use `pytest` for testing.

**Test Structure:**
- Place tests in the `tests/` directory
- Use descriptive test function names
- Test both happy paths and edge cases
- Include docstrings for test functions

**Example Test:**
```python
def test_grib_download():
    """Test that GRIB files can be downloaded successfully."""
    grib = Grib(model='gfswave')
    result = grib.download()
    assert result > 0  # Should download at least one file
```

## Documentation

Good documentation is essential for any open source project.

### Documentation Guidelines

- Use clear, concise language
- Include code examples
- Document all public APIs
- Keep documentation up-to-date
- Use Markdown format

### Updating Documentation

1. Edit the relevant `.md` files
2. Update docstrings in the code
3. Add examples if applicable
4. Ensure all links work

## Recognition

We appreciate all contributions, big and small! Contributors will be recognized in:
- The project's `CONTRIBUTORS.md` file
- Release notes
- GitHub contributors list

## Questions?

If you have any questions about contributing, please:
- Open an issue with your question
- Ask in our discussions
- Contact the maintainers directly

We're happy to help you get started!

---

*Thank you for contributing to WindGrib! 🌬️*