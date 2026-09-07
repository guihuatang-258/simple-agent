# Global Git policy

For tasks that change source code, tests, scripts, or project configuration:

- Before editing, find the Git root and inspect `git status`.
- Preserve existing user changes. Never discard or commit unrelated work.
- Run appropriate tests or checks and report failures.
- Commit after completing a coherent unit of work. For incomplete tasks or small intermediate changes, keep changes uncommitted and combine them into a later focused commit.
- Stage only task-related changes and review the staged diff before committing.
- If task changes cannot be separated safely from existing changes, do not commit; explain why.
- Never commit secrets, `.env` files, caches, or unexpected generated files.
- Do not bypass Git hooks, push, rebase, amend, rewrite history, or create a PR unless explicitly requested.
- Follow an explicit user request to commit or leave changes uncommitted.
- Do not run `git init` without permission.
- Report commits, verification results, and remaining uncommitted changes.
