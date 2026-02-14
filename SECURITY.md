# Security Policy

## Important Disclaimer

**This is a research / demonstration system. It is NOT approved for clinical use and must NOT be used to make real medical decisions.** All clinical decisions must be made by qualified healthcare professionals.

## Reporting a Vulnerability

If you discover a security vulnerability in this project, please report it responsibly:

1. **Do NOT open a public GitHub issue** for security vulnerabilities
2. Email the maintainer directly (see GitHub profile for contact info)
3. Include a description of the vulnerability, steps to reproduce, and potential impact

We will acknowledge receipt within 48 hours and aim to provide a fix or mitigation within 7 days for critical issues.

## Scope

### In scope

- Authentication/authorization bypasses (if auth is added in the future)
- Injection vulnerabilities (prompt injection, SQL injection, command injection)
- Sensitive data exposure (API keys, patient data leakage)
- Dependency vulnerabilities in `requirements.txt` or `package.json`
- CORS misconfigurations that could enable data exfiltration

### Out of scope

- Clinical accuracy of AI-generated recommendations (this is a known limitation, not a vulnerability)
- Denial of service via expensive LLM calls (known limitation of the architecture)
- Issues in third-party services (Google AI Studio, OpenFDA, RxNorm)

## Security Considerations for This Project

### Patient Data

- This system processes clinical text that could contain protected health information (PHI)
- **No real patient data should ever be used** with this demonstration system
- In a production deployment, HIPAA compliance would require: encrypted storage, audit logging, access controls, and BAAs with all third-party services
- The Gemma model can be self-hosted on-premises to avoid sending data to external APIs

### API Keys

- The Google AI Studio API key is stored in `.env` (gitignored)
- Never commit `.env` or any file containing API keys
- The `.env.template` file shows required variables without actual values

### LLM-Specific Risks

- **Prompt injection:** The system processes untrusted user input (patient text) that is sent to the LLM. Adversarial inputs could potentially manipulate LLM behavior.
- **Hallucination:** The LLM may generate plausible but incorrect medical information. The conflict detection step and RAG grounding mitigate but do not eliminate this risk.
- **Over-reliance:** The system is designed as decision *support*, not decision *making*. UI disclaimers and caveats are included to reinforce this.

## Supported Versions

| Version | Supported |
|---------|-----------|
| Current `master` branch | Yes |
| Older commits | No |
