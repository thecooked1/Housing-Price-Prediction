
# California Housing Price Prediction

## Problem Statement and Objectives

The real estate market is influenced by a complex interplay of geographic, economic, and demographic factors. Simple intuition is often insufficient to predict property values accurately.

The goal of this project is to predict house prices in California districts based on various features (such as median income, housing age, and location) and analyze the primary drivers of the housing market.  
We compare two machine learning approaches: Multiple Linear Regression and Decision Tree Regressor.

* **Data Quality:** Implement a reproducible preprocessing pipeline to handle missing values and extreme outliers using statistical methods (IQR).

* **Market Analysis:** Conduct Exploratory Data Analysis (EDA) to identify key drivers of house prices through statistical visualization.

* **Predictive Modeling:** Build and compare Multiple Linear Regression and Decision Tree Regressor models to determine which algorithm best captures the patterns in California housing data.

* **Evaluation:** Measure model performance using R-squared (R2R2) and Mean Squared Error (MSE).

## Dataset Description and Source

The dataset used in this project is the **_California Housing Prices dataset_**, sourced from _Kaggle_.

**Source:** Kaggle - California Housing Prices

**Scale:** 20,640 records across 10 initial features.

**Key Features:**

* `median_income:` Median income for households within a block of houses.
* `housing_median_age:` Median age of a house within a block.
* `total_rooms` & `total_bedrooms:` Aggregate counts for the district.
* `ocean_proximity:` Categorical location relative to the coast.
* `median_house_value` (Target): The median house price for households within a block.

## Project Structure

The repository is organized following professional data science standards:

* `data/`
* `raw/`: Original, immutable Kaggle dataset.
    * `processed/`: Cleaned and transformed data ready for modeling.

* `notebooks/`:
    * `01_data_exploration.ipynb:` Initial data inspection and quality report.
    * `02_data_preprocessing.ipynb:` Data cleaning, outlier handling, and feature engineering.
    * `03_eda_visualization.ipynb:` Statistical analysis and visualizations.
    * `04_machine_learning.ipynb:` Model training, evaluation, and comparison.

* `src/:` Reusable Python modules.
    * `init.py`
* `data_processing.py:` Modular cleaning and transformation functions.
    * `visualization.py:` Functions for generating consistent plots.
    * `models.py:` Model training and evaluation logic.

* `reports/:`
* `figures/:` Saved visualizations (PNG format).
    * `results/:` Numerical metrics and model performance comparisons (JSON/CSV).

* `requirements.txt:` List of Python dependencies and versions.
* `.gitignore:` Files excluded from version control.
* `README.md:` Comprehensive project description.
* `CONTRIBUTIONS.md:` Team member contributions.

## Installation and Setup

To set up the project environment on your local machine:

1. **Clone the repository:**

       git clone [your-repository-link]
       cd Housing-Price-Prediction
2. **Create and activate a virtual environment (Recommended):**

       python -m venv .venv
       # Windows:
       .venv\Scripts\activate
       # macOS/Linux:
       source .venv/bin/activate

3. **Install required dependencies:**

       pip install -r requirements.txt

## Usage Examples
The project is designed to be executed sequentially through the provided Jupyter notebooks.

### Execution Order:
1.  `notebooks/01_data_exploration.ipynb`: Run this first to generate the initial Data Quality Report and statistical summaries.

2.  `notebooks/02_data_preprocessing.ipynb:` Executes the cleaning pipeline. It imports logic from src/data_processing.py and saves the cleaned data to data/processed/.

3.  `notebooks/03_eda_visualization.ipynb:` Generates 5+ visual types (Heatmaps, Scatter, Boxplots, etc.) and saves them to reports/figures/.

4.  `notebooks/04_machine_learning.ipynb:` Trains the models, evaluates performance, and identifies the most important features.

##  Results Summary
### Data Insights:
-   **Median Income** was identified as the most significant predictor of house value ( `r≈0.64`).

-   **Proximity to the Ocean** significantly impacts both the price and the density of housing units.

-   Engineered features such as **Rooms per Household** provided more statistical value than raw room counts.

### Model Performance:
| Metric | Linear Regression |Decision Tree Regressor
|--|--|--|
| R-Squared | 0.6116 |0.5669 |
|MSE|3359271807.5834|3746393949.6182 |

**Final Conclusion:** The **Decision Tree Regressor** outperformed Linear Regression. This is attributed to the Decision Tree's ability to capture non-linear relationships and geographic thresholds (latitude/longitude interactions) that a linear model cannot inherently process.

## Technical Standards & Code Quality

-   **Modularity:** Logic is strictly separated into the `src/` directory for data processing, visualization, and modeling.

-   **Documentation:** All Python functions follow the standard docstring format (Parameters, Returns, Raises).

-   **Style:** All code is compliant with **PEP 8** standards.

-   **Safety:** Implemented comprehensive error handling with try-except blocks for all data IO and visualization tasks.



## Honor Code

I certify that this work is my own.

