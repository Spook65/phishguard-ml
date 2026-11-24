# Agent instructions for PhishGuard ML

PhishGuard ML is a completed COMP 151 Artificial Intelligence research project. Preserve its recorded methods and results; optimize any future changes for correctness, clarity, testability, and a narrow scope.

- Work in small, reviewable changes; explain WHAT, WHY, HOW, files changed, and tests run.
- Do not introduce frontend, APIs, SaaS integrations, LLM calls, or live website scanning.
- Data source: UCI Phishing Websites dataset ID 327, with 30 precomputed categorical/integer features. Preserve the raw labels and the documented target-mapping uncertainty.
- Preserve the group-safe split and frozen held-out evaluation. Do not use test results for parameter tuning or silently discard duplicate feature patterns.
- Primary methods: sklearn entropy-based DecisionTreeClassifier and KNeighborsClassifier; explicitly distinguish sklearn's tree from exact ID3.
- Report phishing-class precision/recall/F1, confusion matrix, accuracy, and prediction time; no fabricated metrics.
- Stay within local repository; ask before any destructive operations or external repo changes.
- No execution or retrieval of live suspicious URLs or unsafe attachments.
- Do not silently write files outside the project or remove existing user files.
- Do not make security promises about model predictions; it is research, not an enforcement boundary.
