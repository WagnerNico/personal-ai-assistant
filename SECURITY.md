# Security and publication notes

Airis is an experimental local-first macOS assistant. It performs limited desktop actions and can access optional third-party services. Review its permission boundaries before use.

## Permissions and data boundaries

- Qwen language-model inference runs locally on Apple Silicon. Optional web retrieval and Gmail operations require external network access.
- Gmail connections require explicit OAuth authorization with a read-only scope. Never publish OAuth client secrets, access/refresh tokens, or real message contents.
- Gmail content and external web pages are untrusted data, not authorization to invoke tools or desktop actions.
- Supported macOS tool actions are deliberately constrained; the assistant is not intended to run arbitrary LLM-generated shell commands.
- Voice references and generated samples should only be shared if you hold the relevant rights and have consent when necessary.

## Public-release checklist

This `personal-ai-assistant` repository was exported into a new Git history from a sanitized development snapshot. The original private project and its earlier commits are **not** part of this repository.

Before changing visibility to Public:

1. Run an independent repository-wide secret and personal-information scan; the initial review was selective, not exhaustive.
2. Inspect documentation, tests, examples, and embedded text for private identifiers and confidential material.
3. Review dependencies and third-party licenses, including pretrained models and any audio or images added as demos.
4. Verify setup and representative actions on a clean macOS installation.
5. Confirm no reference audio, account tokens, or local environment artifacts have entered new commits.
6. Decide whether you want to offer an explicit software license. None is currently granted.

Use test accounts and synthetic data in any publicly shared demo. If you find a suspected privacy or security issue, contact the repository owner privately rather than posting secrets in an issue.

## Technical limitations

This is a portfolio/development prototype, not a security-audited product. Permission checks and unit tests do not eliminate all risks of external integrations.
