# Agent instructions for PhishGuard ML

This is a two-week university AI course experiment. Optimize for correctness, clarity, testability, and a narrow scope.

- Work in small checkpoints; explain WHAT, WHY, HOW, files changed, and tests run.
- Do not introduce frontend, APIs, SaaS integrations, LLM calls, or live website scanning.
- Data source: UCI Phishing Websites dataset ID 327, with 30 precomputed categorical/integer features. Verify the target mapping after downloading.
- Avoid data leakage: separate held-out test from parameter tuning; inspect duplicate feature patterns.
- Primary methods: sklearn entropy-based DecisionTreeClassifier and KNeighborsClassifier; explicitly distinguish sklearn's tree from exact ID3.
- Report phishing-class precision/recall/F1, confusion matrix, accuracy, and prediction time; no fabricated metrics.
- Stay within local repository; ask before any destructive operations or external repo changes.
- No execution or retrieval of live suspicious URLs or unsafe attachments.
- Do not silently write files outside the project or remove existing user files.
- Do not make security promises about model predictions; it is research, not an enforcement boundary.
