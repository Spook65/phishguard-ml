# First Codex task: dataset inspection ONLY

Recommended model: newest GPT-6 coding model available to you in Codex.
Reasoning effort: Medium for this first task; use High later for diagnosing leakage or model evaluation.
Thread: NEW, in this repository, separate from SecureAgent Lab.

## Prompt

You are working on PhishGuard ML, a two-week COMP 151 AI final project. First inspect README.md, AGENTS.md, and the proposal in docs/. This is a focused offline ML benchmark, not a URL scanner.

**Checkpoint 1 - dataset intake only:**
1. Inspect the existing scaffold. Keep its simplicity.
2. Implement a small script that downloads/loads official UCI Phishing Websites dataset ID 327 using `ucimlrepo`. No raw page or URL fetching.
3. Verify and print: sample count, feature count, column names, exact label counts and mapping, missing values, duplicate feature vectors, and whether identical features have contradictory labels.
4. Add tests for stable schema/label assertions without making tests depend on network access.
5. Run relevant tests and report commands, results, and any concerns.

Do NOT train classifiers, generate a UI, create a GitHub repository, deploy anything, or change the scope. If a source is unavailable, report it and stop rather than inventing data.

Explain to me what changed, why it matters, and what I should verify before we proceed to Checkpoint 2.
