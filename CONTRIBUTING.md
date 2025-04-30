# Contributing to the Intraday Statistical Arbitrage System

Thank you for your interest in contributing to this project! We welcome contributions from everyone, whether it's fixing bugs, improving documentation, adding new features, or providing feedback.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork to your local machine
3. Create a new branch for your changes
4. Make your changes
5. Push your changes to your fork
6. Submit a pull request

## Development Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # For development dependencies
```

2. Set up the pre-commit hooks:
```bash
pre-commit install
```

## Code Style

We follow a strict code style to maintain consistency across the codebase:

- Follow PEP 8 guidelines for Python code
- Use descriptive variable, function, and class names
- Write docstrings for all functions, classes, and methods
- Add type hints where possible
- Keep functions focused and short

## Testing

Before submitting a pull request, make sure your changes pass all tests:

```bash
pytest
```

If you're adding a new feature, please include tests that cover your code.

## Pull Request Process

1. Ensure your code adheres to the project's code style
2. Update the documentation if necessary
3. Update the README.md if necessary
4. Include a clear description of your changes in the pull request
5. Link any related issues

## Feature Requests

We welcome feature requests! Please submit them as issues, and use the "feature request" template.

## Bug Reports

Found a bug? Please submit an issue using the "bug report" template. Include:

- A clear description of the bug
- Steps to reproduce
- Expected behavior
- Actual behavior
- Screenshots if applicable
- System information

## Areas for Contribution

Here are some areas where contributions would be especially welcome:

- Expanding the test suite
- Implementing additional cointegration test methods
- Improving visualization tools
- Adding machine learning components for spread prediction
- Optimizing performance for large datasets
- Adding support for additional asset classes
- Creating a web-based user interface

## Code of Conduct

Please be respectful and constructive in all interactions. We aim to foster an inclusive and welcoming community.

## License

By contributing to this project, you agree that your contributions will be licensed under the project's MIT License.

Thank you for contributing to the Intraday Statistical Arbitrage System! 