# Security and publication notes

Airis is an experimental macOS AI assistant. Tool integrations have meaningful operating-system and account permissions; review the code and grant only permissions you intend.

## Scope

- Local model inference is different from end-to-end offline operation. Optional Gmail and web integrations use external networks.
- Gmail authorization is **read-only**, and messages are treated as untrusted content. Do not use live personal inboxes for public demos.
- macOS application tools are limited to supported, validated actions. No arbitrary LLM-generated shell execution is intended.
- Voice cloning needs recordings for which the operator has explicit rights/consent.

## Public release — not yet authorized

**Do not make the existing development repository public as-is.** The showcase branch shares its underlying Git history with the private `main` branch. Removing files from the branch does not remove old blobs, recordings, machine paths, or possible credentials from history.

For a truly public showcase:

1. Export **only the reviewed, sanitized branch working tree**, not its history.
2. Create a **new, empty repository** with a fresh Git history.
3. Inspect all exported files, including nested assets, tests, configs, and example data for secret strings, personal identifiers, licensing, account URLs, and private documents.
4. Run an independent secret scan on the files **and** check that the fresh repository contains only the intended new commit history.
5. Supply only original or properly licensed demo screenshots and recordings.
6. Verify a clean install and test representative desktop actions on a local Mac.
7. Choose a software license intentionally; there is currently no open-source license.
8. Publish the new repository only after the human owner reviews the export.

Example *local* export procedure from a working clone (replace the destination with a new, separately created local directory; commands must not be run from a directory with uncommitted work):

```bash
mkdir -p ../airis-public-export
git archive showcase/hackathon-2026 | tar -x -C ../airis-public-export
cd ../airis-public-export
git init
git add .
git commit -m "Initial Airis showcase release"
```

Before pushing this fresh directory, inspect `git status`, the full file list, and the new Git log. Never push the original `main` or showcase branch history to the public remote.

## Reporting problems

Please contact the repository owner privately if you discover a security or privacy issue. Do not post secrets or real user data in a public issue.
