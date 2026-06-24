# Security Policy

## Supported versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | ✅ Yes    |

## Reporting a vulnerability

**Please do not report security vulnerabilities via public GitHub Issues.**

Use [GitHub's private vulnerability reporting](https://github.com/aurora-creatives/taproot/security/advisories/new) to submit a confidential report. Include:

- A description of the vulnerability
- Steps to reproduce it
- Potential impact
- Any suggested mitigations

You will receive a response within **48 hours** acknowledging receipt. We aim to release a fix within **14 days** of a confirmed vulnerability.

## Scope

taproot is a local CLI tool. The primary security concerns are:

- **Secret handling** — API keys must only be read from environment variables (`.env`). Never hardcoded, never logged.
- **Input validation** — ticket content and LLM responses are treated as untrusted input. Pydantic validation is applied at all model boundaries.
- **No remote execution** — taproot does not expose any network ports or accept inbound connections.
- **ITSM credentials** — when connecting to real ITSM systems, credentials are read from environment variables only and are never written to disk or included in output files.

## Out of scope

- Vulnerabilities in third-party dependencies (report those upstream)
- Issues requiring physical access to the machine
