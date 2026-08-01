# California Housing Price Prediction Using Clustering and Regression Models

A machine learning project that predicts median house values in California by combining unsupervised clustering with supervised regression. The workflow includes data preprocessing, feature engineering, K-means clustering, KNN-based cluster-label transfer, PCA visualisation, hyperparameter tuning, model comparison, and neural-network regression.

## Project Overview

The project investigates whether cluster information can support house-price prediction. K-means is fitted only on the training data, and a K-Nearest Neighbours classifier transfers the learned cluster labels to the test data. These cluster labels are then added as categorical input features for the regression models.

Four regression models are compared:

- Linear Regression
- Support Vector Regression with an RBF kernel
- Random Forest Regression
- Multi-Layer Perceptron

Model performance is evaluated using Root Mean Squared Error, Mean Absolute Error, and the coefficient of determination.

## Objectives

- Explore the distribution and relationships of California housing variables.
- Handle missing values and encode categorical data.
- Create ratio-based features that better describe housing characteristics.
- Group similar housing records using K-means clustering.
- Transfer training-set cluster labels to the test set using KNN.
- Visualise cluster structure using Principal Component Analysis.
- Train and tune several regression models.
- Compare training and test performance to identify the most accurate model and assess overfitting.

## Dataset

The dataset contains California housing, demographic, income, and location-based information. The prediction target is:

```text
median_house_value
```

Expected input columns include:

```text
longitude
latitude
housing_median_age
total_rooms
total_bedrooms
population
households
median_income
median_house_value
ocean_proximity
```

The original dataset contains 20,640 records. Rows with missing values in `total_bedrooms` are removed before modelling.

The dataset file is not included in this repository. Place it at:

```text
data/housing.csv
```

## Project Workflow

### 1. Data Preprocessing

- Load `housing.csv`.
- Inspect data types, summary statistics, and missing values.
- Remove rows containing missing values.
- One-hot encode `ocean_proximity`.
- Split the data into 80% training and 20% testing sets using `random_state=42`.
- Fit scalers only on the training data and apply the fitted transformations to the test data.

### 2. Feature Engineering

Three ratio features are created:

```text
rooms_per_household
total_rooms / households

bedrooms_per_room
total_bedrooms / total_rooms

population_per_household
population / households
```

These features provide more meaningful household-level information than the original aggregate counts alone.

### 3. K-Means Clustering

K-means clustering is performed using:

```text
median_income
rooms_per_household
bedrooms_per_room
latitude
```

The features are standardised before clustering. Candidate values of `k` from 2 to 10 are assessed using:

- Elbow method
- Silhouette score

The final model uses:

```text
k = 4
```

### 4. KNN Cluster-Label Transfer

Because K-means is fitted only on the training set, a KNN classifier is trained using the K-means training labels. It then assigns each test record to one of the existing clusters.

The cluster labels are one-hot encoded and added to the regression feature set.

### 5. PCA Visualisation

Principal Component Analysis reduces the selected clustering features to two components for visualisation. PCA is used only to inspect the cluster structure and compare training and test distributions; it is not used as the regression input.

### 6. Regression Models

#### Linear Regression

Hyperparameters tuned with `GridSearchCV`:

```text
fit_intercept
positive
```

#### Random Forest Regression

Hyperparameters tuned with `RandomizedSearchCV`:

```text
n_estimators
max_depth
min_samples_split
min_samples_leaf
max_features
bootstrap
```

#### Support Vector Regression

An RBF-kernel SVR is tuned with `RandomizedSearchCV` using:

```text
C
gamma
epsilon
```

Both the input features and target values are standardised for SVR.

#### Multi-Layer Perceptron

The neural network contains three hidden layers:

```text
128 neurons -> 64 neurons -> 32 neurons -> 1 output neuron
```

Main settings:

```text
Activation: ReLU
Dropout rate: 0.2
L2 regularisation: 0.00001
Learning rate: 0.001
Optimizer: Adam
Loss function: Mean Squared Error
Output activation: Linear
```

Five-fold cross-validation and early stopping are used during hyperparameter selection.

## Evaluation Metrics

The models are evaluated with:

- **RMSE:** penalises larger prediction errors more heavily.
- **MAE:** measures the average absolute prediction error in the original target units.
- **R²:** measures the proportion of variance in house values explained by the model.

Lower RMSE and MAE values are better, while a higher R² value is better.

## Results

| Model | Train RMSE | Train MAE | Train R² | Test RMSE | Test MAE | Test R² |
|---|---:|---:|---:|---:|---:|---:|
| Linear Regression | 66,579.71 | 48,198.88 | 0.662 | 68,103.34 | 49,517.42 | 0.657 |
| Support Vector Regression | 49,685.75 | 34,007.26 | 0.812 | 55,182.32 | 38,046.13 | 0.775 |
| Random Forest Regression | 27,782.18 | 17,359.80 | 0.941 | 50,398.49 | 32,835.02 | 0.812 |
| Multi-Layer Perceptron | 60,451.25 | 42,029.32 | 0.725 | 61,786.20 | 43,196.39 | 0.718 |

## Key Findings

- Random Forest achieved the strongest test performance, with the lowest test RMSE and MAE and the highest test R².
- The gap between Random Forest training and test performance indicates some overfitting.
- SVR produced the second-best test results and showed more balanced generalisation.
- The MLP performed better than Linear Regression but did not outperform Random Forest or SVR.
- Linear Regression produced the weakest results, suggesting that a simple linear relationship is insufficient for this dataset.
- The target variable contains many observations capped at 500,000, making the most expensive properties difficult for all models to predict accurately.
- PCA and silhouette analysis indicate that the four clusters have moderate, rather than strong, separation.

## Repository Structure

```text
california-housing-price-prediction/
├── README.md
├── california_housing_models.ipynb
├── requirements.txt
├── .gitignore
├── data/
│   ├── README.md
│   └── housing.csv
└── saved_models/
    ├── KMeans.pkl
    ├── KNN.pkl
    ├── LR.pkl
    ├── RF.pkl
    ├── SVR.pkl
    └── MLP.keras
```

The dataset and generated model files do not need to be uploaded to GitHub.

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/california-housing-price-prediction.git
cd california-housing-price-prediction
```

Create and activate a virtual environment.

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install the required packages:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Running the Project

1. Place the dataset at:

```text
data/housing.csv
```

2. Start Jupyter Notebook:

```bash
jupyter notebook
```

3. Open:

```text
california_housing_models.ipynb
```

4. Run the notebook cells in order.

## Main Dependencies

```text
ipython
joblib
jupyter
matplotlib
numpy
pandas
scikit-learn
scipy
seaborn
tensorflow
```

## Generated Outputs

Running the notebook produces outputs such as:

- Feature-distribution plots
- Correlation matrix
- Target-correlation chart
- Elbow and silhouette plots
- PCA cluster visualisations
- Learning curves
- Actual-versus-predicted plots
- Residual plots
- Model-performance comparison charts
- Random Forest feature importance
- Saved scikit-learn and TensorFlow models

## Limitations

- The median-house-value target is capped at 500,000, which limits the modelling of very expensive properties.
- K-means assumes compact clusters and may not fully represent complex housing-market structures.
- The selected clusters show only moderate separation.
- Random Forest has a noticeable train-test performance gap.
- The results are based on one train-test split, although cross-validation is used during model tuning.

## Future Improvements

- Compare the current models with gradient-boosting methods such as XGBoost or LightGBM.
- Apply logarithmic transformations to highly skewed variables.
- Investigate spatial feature engineering using longitude and latitude.
- Evaluate whether cluster labels genuinely improve regression performance through an ablation study.
- Use repeated cross-validation for a more robust estimate of model performance.
- Explore alternative clustering methods for non-spherical or overlapping groups.

## Author

**Zhaojun Yao**  
MSc Data Science, University of Exeter

## Academic and Privacy Note

This repository was prepared as an academic machine learning project. Personal identifiers and student information have been excluded from the public repository.
