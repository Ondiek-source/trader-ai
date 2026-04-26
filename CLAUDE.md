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
