# Data Directory Guide

## Overview

The `data/` folder stores all datasets, intermediate artifacts, and metadata used throughout this benchmarking project. Keep large or sensitive files out of version control and document any additions here so collaborators know how to reproduce the required inputs.

## Organization Recommendations

- Place raw downloads (ZIP/TAR/CSV) directly under `data/` and extract them into subdirectories named after the archive.
- Stage derived datasets or preprocessing outputs inside clearly labeled subfolders (for example, `processed/` or `features/`).
- Maintain small helper files—such as sampling scripts or schema notes—in this directory when they relate to the data pipeline.
- Ignore raw and intermediate artifacts via `.gitignore`; the core datasets referenced below already have their ZIP archives and extracted folders on that list.

## Bank Account Fraud Dataset (NeurIPS 2022)

### Source

- Dataset: Bank Account Fraud Dataset (NeurIPS 2022)
- Maintainer: Sergio Pérez (sgpjesus)
- Direct download: https://www.kaggle.com/api/v1/datasets/download/sgpjesus/bank-account-fraud-dataset-neurips-2022

### Download

Fetch the archive from the dataset endpoint while located in `data/`:

```bash
cd data/
curl -L -o bank-account-fraud-dataset-neurips-2022.zip \
  https://www.kaggle.com/api/v1/datasets/download/sgpjesus/bank-account-fraud-dataset-neurips-2022
```

The command saves `bank-account-fraud-dataset-neurips-2022.zip` alongside this README.

### Extract

Unzip the archive into a folder that matches the ZIP filename:

```bash
unzip bank-account-fraud-dataset-neurips-2022.zip \
  -d bank-account-fraud-dataset-neurips-2022
```

After extraction, the `data/` directory should resemble:

```
data/
├── README.md
├── bank-account-fraud-dataset-neurips-2022.zip
└── bank-account-fraud-dataset-neurips-2022/
    ├── train.csv
    ├── test.csv
    └── ...
```

Both the ZIP archive and the extracted directory are ignored by Git so you can safely remove or regenerate them locally without touching version history.

Remove the ZIP once verified if you need to reclaim space.
