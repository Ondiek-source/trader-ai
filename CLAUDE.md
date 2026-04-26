# TraderAI — Claude Code Instructions

## After Every Major Change

After completing any significant code change (bug fix, refactor, feature, performance improvement), always:

1. Generate a complete, descriptive commit message summarising what changed and why
2. Commit the changes (`git add` the relevant files, then `git commit`)
3. Push to origin (`git push`)

Do this automatically without being asked. A "major change" includes anything that modifies runtime behaviour, fixes a bug, changes an API, or touches more than one file in a meaningful way. Trivial formatting-only edits do not require a push.

## Commit Message Format

Every commit message must follow this structure:

```text
<short imperative summary line>

<body: bullet points explaining what changed and why>

Author: Martin Ondiek
Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
```

The `Author: Martin Ondiek` line is mandatory and must always appear before the Co-Authored-By trailer. Never omit it.

## Working Style & Expectations

### Answering Questions vs Making Changes

When a message asks "is X happening?", "does Y work?", or "should we do Z?" —
**answer first, do not write code unless explicitly asked**.

- Diagnostic questions ("is activity log sent on kill switch?") → answer with
  file:line references pointing at the exact code that handles it. One or two
  sentences is enough.
- Trade-off questions ("should we use a JSON delete queue?") → give a direct
  recommendation and the main tradeoff in 2–3 sentences. Present it as a
  suggestion the user can redirect, not a decided plan. Do not implement until
  the user agrees.
- Only proceed to implementation when the user confirms, or when the request
  is unambiguously an instruction ("fix X", "add Y").

### Scope of Changes

Only touch what the task requires.

- Do not clean up surrounding code, rename variables, or add error handling for
  scenarios that are not part of the ask.
- Do not add features, abstractions, or refactors beyond what was requested.
- When fixing a bug in one method, leave adjacent methods untouched even if
  they look improvable.

### Commit Hygiene

Only stage files that were modified by the current task.

- Always check `git diff --stat` before staging to identify pre-existing
  unstaged changes in other files. Do not sweep them into the commit.
- Stage by explicit file path (`git add src/engine/live.py`), never
  `git add .` or `git add -A`.

### Code References

When pointing at code in answers, always use `file_path:line_number` or
markdown links so the location is immediately navigable. Never describe code
in prose when a line reference is available.

### Logs & Dashboard Precision

- All threshold values in log entries must be `round(..., 2)`.
- All timestamps in logs and dashboard entries use `local_now()` from
  `core.config`, not `datetime.now(timezone.utc)`.
- Dashboard labels and activity log entries must be human-readable and
  accurate — never silently swallow events that the user would want to see.

### Notification Channel Separation

- **Discord**: per-signal trade placed alerts + per-result alerts +
  session reports on demand.
- **Telegram**: daily summary only, sent once when a daily target
  (win-count or profit) is reached. Never per-trade.
- Do not add Telegram to any new per-event notification path without
  explicit instruction.

### Config-Driven Behaviour

Prefer driving behaviour from `.env` / `Config` over hardcoded values.
When adding a new threshold, flag, or limit, check whether it belongs in
`Config` before hardcoding it.

### Verification Before Acting on Memory

If recalling a file path, function name, or module from a previous session,
verify it still exists in the current codebase before referencing or modifying
it. Memory can be stale after refactors.
