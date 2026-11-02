# Python Projects

A collection of small Python exercises and data-analysis experiments.

## Current project: Wine Quality Exploratory Analysis

The repository currently includes `KNN.py`. Despite the filename, the script does **not** implement or train a K-Nearest Neighbors model. It downloads the Wine Quality dataset through `kagglehub`, loads the CSV with pandas, and performs exploratory data analysis and visualization.

### What the script does

- Downloads the Wine Quality dataset using `kagglehub`.
- Loads `WineQT.csv` into a pandas DataFrame.
- Computes and visualizes a correlation matrix.
- Generates pair plots and jitter/strip plots to explore relationships between physicochemical measurements and wine quality.
- Displays dataset information and plots with Matplotlib and Seaborn.

### Requirements

- Python 3
- kagglehub
- pandas
- numpy
- matplotlib
- seaborn

Install the dependencies:

```bash
python -m pip install kagglehub pandas numpy matplotlib seaborn
```

### Run

```bash
python KNN.py
```

The first run may require Kaggle dataset access/configuration depending on your environment. The script currently assumes the downloaded dataset contains `WineQT.csv` and uses a Windows-style path separator when loading it; portability improvements are still needed.

## Scope and limitations

This is an exploratory analysis exercise, not a completed machine-learning experiment. It currently has no KNN training, train/test split, model evaluation, cross-validation, or reported predictive metrics. No conclusions about predictive performance should be inferred from the visualizations alone.

## Future improvements

- Rename the script to reflect its current exploratory-analysis purpose, or implement an actual KNN workflow.
- Use portable path handling with `pathlib`.
- Add a reproducible environment file and dataset/version information.
- If implementing KNN, define the prediction target, preprocessing, evaluation protocol, and baseline models.

## Author

Ruan Molinari — [GitHub](https://github.com/RuanMolinari)
