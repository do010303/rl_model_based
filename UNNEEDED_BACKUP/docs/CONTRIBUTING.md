# Contributing to Robotarm-RL-4DoF

We welcome contributions to this project! Here's how you can help:

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/Robotarm-RL-4DoF.git`
3. Install dependencies: `pip install -r requirements.txt`
4. Create a branch: `git checkout -b feature/your-feature-name`

## Development Setup

```bash
# Create virtual environment
python -m venv robotarm_env
source robotarm_env/bin/activate  # Linux/Mac
# robotarm_env\Scripts\activate  # Windows

# Install in development mode
pip install -e .

# Run tests
python -m pytest tests/
```

## Types of Contributions

### 🐛 Bug Reports
- Use the issue tracker to report bugs
- Include system information, error messages, and steps to reproduce

### ✨ Feature Requests
- Describe the feature and its use case
- Consider providing a simple implementation proposal

### 🔧 Code Contributions
- Follow the existing code style
- Add tests for new functionality
- Update documentation as needed

## Code Style Guidelines

### Python Code Style
- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add docstrings to functions and classes
- Keep functions focused and small

### Commit Messages
- Use conventional commit format: `type: description`
- Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`
- Example: `feat: add curriculum learning scheduler`

## Testing

- Write tests for new functionality
- Ensure all tests pass before submitting PR
- Include integration tests for major features

```bash
# Run all tests
python -m pytest

# Run specific test
python -m pytest tests/test_environment.py

# Run with coverage
python -m pytest --cov=robotarm_rl
```

## Documentation

- Update README.md for significant changes
- Add docstrings to new functions/classes
- Include usage examples for new features

## Pull Request Process

1. Ensure your code follows the style guidelines
2. Add or update tests as appropriate
3. Update documentation if needed
4. Submit PR with clear description of changes
5. Respond to review feedback

## Code Review Checklist

- [ ] Code follows style guidelines
- [ ] Tests are included and passing
- [ ] Documentation is updated
- [ ] No breaking changes (or clearly documented)
- [ ] Performance impact considered

## Questions?

Feel free to open an issue for questions or reach out via email: vnquan.hust.200603@gmail.com

Thank you for contributing! 🚀
