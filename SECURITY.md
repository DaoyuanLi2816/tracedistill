# Security policy

## Supported versions

Security fixes are applied to the latest released minor version.

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting feature on the repository's
**Security** tab. Include the affected version, a minimal reproduction, impact, and any
suggested mitigation.

Do not open a public issue for vulnerabilities involving arbitrary code execution,
unsafe model loading, dependency compromise, credential exposure, or malicious model
artifacts.

The project loads third-party models only when a user explicitly starts training.
`trust_remote_code` is disabled by default; enable it only for model repositories you
trust and have reviewed.
