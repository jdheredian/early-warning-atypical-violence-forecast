# Early Warning Atypical Violence Forecast

## Project Overview

This project contains econometric, machine learning, and deep learning analysis for early warning atypical violence forecasting.

## Project Structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/                  # Original, immutable data
│   ├── temp/                 # Temporary intermediate files
│   └── processed/            # Cleaned and processed data
│
├── notebooks/
│   ├── 0_exploration/        # Initial data exploration
│   │   └── paths.yml
│   ├── 1_cleaning/           # Data cleaning notebooks
│   │   └── paths.yml
│   ├── 2_stats/              # Statistical analysis
│   │   └── paths.yml
│   ├── 3_models/             # Model development
│   │   └── paths.yml
│   └── 9_report/             # Final reports and presentations
│       └── paths.yml
│
├── src/
│   ├── python/               # Python source code
│   │   ├── cleaning/         # Data cleaning scripts
│   │   │   └── paths.yml
│   │   ├── models/           # Model implementations
│   │   │   └── paths.yml
│   │   └── evaluation/       # Model evaluation scripts
│   │       └── paths.yml
│   └── r/                    # R source code
│       ├── cleaning/
│       │   └── paths.yml
│       ├── models/
│       │   └── paths.yml
│       └── evaluation/
│           └── paths.yml
│
├── outputs/
│   ├── other/                # Other outputs
│   ├── figures/              # Generated graphics and visualizations
│   ├── tables/               # Generated tables
│   ├── bib/                  # BibTeX references
│   ├── lit/                  # Literature and documentation
│   ├── slides/               # Presentation slides
│   └── docs/                 # Generated documentation
│
└── docs/
    ├── methodology.md        # Methodology documentation
    └── pseudocode.md         # Pseudocode and algorithms
```

## Setup and Installation

### Prerequisites

- Python 3.8 or higher
- R 4.0 or higher
- pip (Python package manager)

### Installation Steps

1. Clone the repository:
```bash
git clone <repository-url>
cd early-warning-atypical-violence-forecast
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Install R packages (if using R):
```R
# Run in R console
install.packages(c("tidyverse", "caret", "glmnet", "randomForest"))
```

## Configuration

### Path Configuration Files

Each working directory contains a `paths.yml` file with absolute paths to project resources. These files are **ignored by git** to allow different users to work on different machines.

#### Setting Up paths.yml for Your Machine

When replicating this project on a new computer, you need to update all `paths.yml` files with your local absolute paths. The files are located in:

- `notebooks/0_exploration/paths.yml`
- `notebooks/1_cleaning/paths.yml`
- `notebooks/2_stats/paths.yml`
- `notebooks/3_models/paths.yml`
- `notebooks/9_report/paths.yml`
- `src/python/cleaning/paths.yml`
- `src/python/models/paths.yml`
- `src/python/evaluation/paths.yml`
- `src/r/cleaning/paths.yml`
- `src/r/models/paths.yml`
- `src/r/evaluation/paths.yml`

#### paths.yml Schema

Replace `<YOUR_PROJECT_ROOT>` with the absolute path to your project directory:

```yaml
project:
  root: "<YOUR_PROJECT_ROOT>"

data:
  raw: "<YOUR_PROJECT_ROOT>/data/raw"
  temp: "<YOUR_PROJECT_ROOT>/data/temp"
  processed: "<YOUR_PROJECT_ROOT>/data/processed"

outputs:
  figures: "<YOUR_PROJECT_ROOT>/outputs/figures"
  tables: "<YOUR_PROJECT_ROOT>/outputs/tables"
  slides: "<YOUR_PROJECT_ROOT>/outputs/slides"
  docs: "<YOUR_PROJECT_ROOT>/outputs/docs"
  model: "<YOUR_PROJECT_ROOT>/outputs/model"

logs:
  root: "<YOUR_PROJECT_ROOT>/logs"
```

## Usage

### Running Notebooks

Navigate to the appropriate notebook directory and launch Jupyter:

```bash
jupyter notebook
```

### Running Scripts

Execute Python scripts from the project root:

```bash
python src/python/cleaning/clean_data.py
```

Execute R scripts:

```bash
Rscript src/r/models/model_script.R
```

## Data Management

- Place raw data files in `data/raw/`
- Never modify raw data directly
- Store temporary files in `data/temp/`
- Save processed data in `data/processed/`

## Output Management

- Figures: Save in `outputs/figures/`
- Tables: Save in `outputs/tables/`
- Reports: Save in `outputs/docs/`
- Presentations: Save in `outputs/slides/`

## Contributing

When contributing to this project:

1. Create a new branch for your feature
2. Update relevant documentation
3. Ensure all paths use the `paths.yml` configuration
4. Do not commit data files or `paths.yml` files

## License

See LICENSE file for details.

## Contact

For questions or issues, please contact me.
