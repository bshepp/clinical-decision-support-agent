# Contributing to CDS Agent

Thank you for your interest in contributing to the Clinical Decision Support Agent!

## Getting Started

1. Fork the repository
2. Clone your fork locally
3. Follow the setup instructions in [README.md](README.md)
4. Create a feature branch from `master`

## Development Setup

```bash
# Backend
cd src/backend
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
copy .env.template .env      # Add your Google AI Studio key

# Frontend
cd src/frontend
npm install
```

## Running Tests

```bash
cd src/backend

# RAG quality (no server needed)
python test_rag_quality.py --rebuild --verbose

# E2E pipeline (requires running backend on port 8002)
python test_e2e.py

# Clinical test suite
python test_clinical_cases.py

# External validation
python -m validation.run_validation --medqa --max-cases 5
```

## How to Contribute

### Reporting Issues

- Use GitHub Issues for bug reports and feature requests
- Include reproduction steps, expected behavior, and actual behavior
- For clinical accuracy concerns, note the patient scenario and expected medical reasoning

### Submitting Changes

1. Create a branch: `git checkout -b feature/your-feature`
2. Make changes with clear, focused commits
3. Ensure existing tests still pass
4. Submit a pull request with a description of what changed and why

### Areas Where Contributions Are Welcome

- **Clinical guidelines** — Adding new guidelines to `app/data/clinical_guidelines.json` (must cite authoritative sources: ACC/AHA, ADA, IDSA, etc.)
- **Test cases** — Additional clinical scenarios in `test_clinical_cases.py`
- **Validation harnesses** — Improving or adding dataset harnesses in `validation/`
- **Frontend polish** — UI/UX improvements, accessibility, responsive design
- **Documentation** — Corrections, clarifications, translations

### Code Style

- **Python:** Standard library conventions, type hints where practical, Pydantic models for data structures
- **TypeScript/React:** Functional components, hooks, Tailwind CSS utility classes
- **Commits:** Descriptive messages with a prefix (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`)

## Important Notes

- **This is a medical AI project.** Changes to clinical reasoning, guidelines, or conflict detection require extra scrutiny. If you're not a domain expert, flag your PR for clinical review.
- **No patient data.** Never commit real patient information. All test cases must be synthetic.
- **API keys.** Never commit API keys or secrets. Use `.env` (gitignored).

## License

By contributing, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).
