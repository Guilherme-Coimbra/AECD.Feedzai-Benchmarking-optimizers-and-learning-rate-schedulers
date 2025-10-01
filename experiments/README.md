# Experiments Directory

Use this folder to track controlled experiments, configuration files, and run artifacts. Each experiment should live in its own subdirectory to keep inputs and outputs reproducible.

## Suggested Layout Per Experiment

```
experiments/<exp-id>/
├── config.yaml        # Training hyperparameters, seeds, data splits
├── logs/              # Console logs, callbacks, or framework-specific logs
├── checkpoints/       # Saved model weights (ignored by Git if large)
├── metrics.json       # Key evaluation metrics for quick comparison
└── notes.md           # Decisions, observations, and follow-up ideas
```

## Best Practices

- Name experiment folders with incremental IDs (e.g., `exp001`) plus short tags describing intent.
- Record commit hashes or notebook references used to launch the run.
- Store generated artifacts (plots, tables) that belong to the experiment here before promoting them to `../results/`.
- Avoid committing large binaries; prefer symlinks, cloud storage references, or artifact registries if needed.

Update this README when you introduce new conventions or automation scripts for experiment management.
