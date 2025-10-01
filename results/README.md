# Results Directory

The `results/` folder aggregates artifacts that summarize completed experiments—plots, tables, and reports that are ready for sharing.

## Organization Tips

- Group related outputs into subfolders named after the experiment ID or topic (for example, `exp001_lr_sweep/`).
- Include a short `readme.txt` or `notes.md` alongside visualizations to explain key findings and how to reproduce them.
- Store vector-friendly formats (`.png`, `.pdf`, `.csv`) for figures and tables so they can be reused in reports.
- Move only curated artifacts here; keep raw or intermediate outputs under `../experiments/`.

## Version Control Guidance

- Large binary assets should be tracked with Git LFS or referenced externally if they exceed repository limits.
- When regenerating results, replace outdated files to avoid clutter. Archive legacy outputs elsewhere if they must be retained.
