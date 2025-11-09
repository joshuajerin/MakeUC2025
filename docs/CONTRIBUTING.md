# Contributing to EmpathLens

Thank you for your interest in contributing to EmpathLens! This document provides guidelines and information for contributors.

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Remember this tool helps people in distress - quality and safety are paramount

## Ways to Contribute

### 1. Bug Reports
- Check existing issues first
- Provide detailed reproduction steps
- Include system info (OS, Python version, Ollama version)
- Share relevant logs (sanitize any personal data)

### 2. Feature Requests
- Explain the use case clearly
- Consider safety and privacy implications
- Propose how it fits with existing design

### 3. Code Contributions

#### Setup Development Environment
```bash
# Fork and clone the repo
git clone https://github.com/yourusername/empathlens.git
cd empathlens

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run tests
python test_service.py
```

#### Code Style
- Follow PEP 8 for Python code
- Use type hints where possible
- Keep functions focused and documented
- Add docstrings for public methods

#### Testing
Before submitting a PR:
1. Run the test suite: `python test_service.py`
2. Test all critical paths (panic, overwhelm, crisis, stop)
3. Verify safety filters work correctly
4. Check that fallbacks engage when services are unavailable

#### Pull Request Process
1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make your changes with clear commit messages
3. Ensure all tests pass
4. Update documentation if needed
5. Submit PR with description of changes

### 4. Documentation
- Fix typos or unclear instructions
- Add examples or use cases
- Improve setup guides
- Translate documentation (future)

### 5. Safety & Ethics
Help us maintain safety:
- Review crisis keyword lists
- Suggest improvements to safety filters
- Report any responses that could be harmful
- Propose better intervention phrasing

## Development Guidelines

### Adding New Crisis Keywords
Edit `detector.py`:
```python
CRISIS_KEYWORDS = [
    r"\bhurt\s+myself\b",
    # Add new pattern here
]
```

### Adding New Interventions
1. Add type to `models.py`: `InterventionType`
2. Map state in `interventions.py`: `INTERVENTIONS`
3. Update LLM prompts in `ollama_client.py`
4. Test thoroughly with real scenarios

### Modifying State Thresholds
Edit `state_machine.py`:
```python
THRESHOLDS = {
    DistressState.CALM: (0.0, 0.2),
    # Adjust ranges here
}
```

### Safety Considerations
When contributing:
- **Never** reduce safety filters
- **Always** test crisis scenarios
- **Consider** edge cases that could harm users
- **Document** why changes improve safety

## Testing Checklist

Before submitting:
- [ ] All tests pass (`python test_service.py`)
- [ ] Crisis keywords trigger escalation
- [ ] Stop command works immediately
- [ ] Responses are ≤18 words
- [ ] No medical advice in outputs
- [ ] Fallbacks work when services unavailable
- [ ] Session management handles timeouts
- [ ] Privacy: no data stored inappropriately

## Questions?

Open a discussion or issue - we're happy to help!

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

