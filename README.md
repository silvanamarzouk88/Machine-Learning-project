# Car Price Prediction — Regression & Classification

ML pipeline on a UK used car dataset covering price regression and price category classification using Linear Regression and KNN.

## Dataset

`car_price.csv` — 72,435 used car listings with 10 columns. Rows with missing `price` values were dropped, reducing the dataset to 68,814 rows.

**Numerical features:** year, price, mileage, tax, mpg, engineSize  
**Categorical features:** model, transmission, fuelType, Make

## Pipeline

### 1. Exploratory Data Analysis
- Price distribution is right-skewed (Mean: £16,578 | Median: £14,495), with values reaching up to £145,000
- Correlation heatmap shows engineSize has the strongest positive correlation with price; mileage has a negative correlation; year has a moderate positive relationship
- Bar charts for Make brand counts and average price per brand

### 2. Train / Test Split
80/20 split using `train_test_split` (random_state=42) performed before any preprocessing to prevent data leakage.

### 3. Missing Value Imputation
Fit on train only, applied to test.

- Numerical columns: median imputation via `SimpleImputer`
- Categorical columns: mode imputation via `SimpleImputer`

### 4. Outlier Treatment
Applied per-column using 1st–99th percentile capping, fit on train and applied to test. Remaining nulls after capping were filled using per-Make median to account for brand-specific distributions.

- `mileage`: clipped to [1st percentile, 99th percentile]
- `tax`: same approach
- `mpg`: same approach
- `engineSize`: zero values replaced with null for non-electric cars before capping, then filled with per-Make mode/median

### 5. Encoding & Scaling
- **Categorical:** One-Hot Encoding for model, transmission, fuelType, Make — chosen because these features have no natural ordering, and label encoding would imply a false numerical relationship
- **Numerical:** StandardScaler applied to year, mileage, tax, mpg, engineSize — essential for KNN since it relies on distance calculations where unscaled features with large ranges would dominate

### 6. Feature Importance Experiment
The most correlated feature with price (engineSize) was removed to measure its impact. R² decreased when it was excluded, confirming it contributes significantly to the model's predictive power.

### 7. Regression — Linear Regression

| Metric | With engineSize | Without engineSize |
|---|---|---|
| R² | Higher | Lower |
| MAE | Lower | Higher |
| RMSE | Lower | Higher |

A scatter plot of actual vs predicted values with a 45° reference line was used to visualize prediction quality.

### 8. Classification Target Engineering
Car prices were binned into three categories using the 33rd and 67th percentiles as thresholds, producing a balanced class distribution:

| Category  | Price Range        | Train Count |
|-----------|--------------------|-------------|
| Cheap     | below £11,490      | ~18,289     |
| Moderate  | £11,490 – £18,000  | ~18,410     |
| Expensive | above £18,000      | ~18,352     |

Quantile-based thresholds were preferred over equal-width binning to ensure balanced classes and avoid bias toward a dominant group.

### 9. Classification — KNN Classifier

K values tested manually: 1, 3, 5, 7, 9, 11, 15, 21. Best manual K selected as k=9 with `weights='distance'`.

GridSearchCV then ran over:
- `n_neighbors`: 1–30
- `metric`: euclidean, manhattan
- `weights`: uniform, distance
- `cv`: 5-Fold KFold (shuffle=True, random_state=42)

Final model uses the best GridSearchCV parameters. Evaluation includes accuracy, classification report (precision, recall, F1 per class), and confusion matrix heatmap.

### 10. Experiments & Observations

**Effect of binning strategy:** Equal-width binning produced slightly higher accuracy than quantile binning but introduced class imbalance and unrealistic price groupings, making quantile binning the more robust choice.

**Effect of removing scaling from KNN:** Removing StandardScaler produced marginally higher accuracy, but scaling remains necessary to prevent large-range features (mileage) from dominating the distance metric and distorting class boundaries.

### 11. Regression vs Classification Comparison

Linear Regression is better suited for predicting exact market values (evaluated by R², MAE, RMSE), while KNN works well for predicting price categories (evaluated by accuracy, F1). Classification is easier on this dataset because grouping prices into ranges reduces the precision required from the model, but it loses exact price information — two cars priced at £15,000 and £19,000 can fall into the same category despite a £4,000 difference.

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy
```

Install with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

## Usage

Run the notebook cells in order. The train/test split is performed early in the pipeline; all imputation, capping, encoding, and scaling steps must be run before the models.
