# Notebooks Directory

This folder contains exploratory and reporting notebooks used throughout the project. Use it for data exploration, prototyping models, and presenting experiment narratives.

## Usage Guidelines

- Prefix notebooks with an ordered index (for example, `01_`, `02_`) to keep execution flows organized.
- Keep heavy preprocessing logic inside modules under `../src/` and import it here; notebooks should stay lightweight.
- Clear or summarize large cell outputs before committing. Consider using tools such as `nbstripout`.
- Document the intent of each notebook in the opening markdown cell, including data subsets and key questions.

## Recommended Subfolders

- `drafts/` — scratch work not ready to share.
- `reports/` — polished notebooks suitable for presentation or export.

Feel free to add more subdirectories as the project grows, but ensure their purpose is documented in this README.
