# Machine Learning

A structured collection of **Machine Learning concepts, practical implementations, and Jupyter notebooks**, covering the complete workflow from data gathering and preprocessing to supervised learning, unsupervised learning, ensemble methods, and handling imbalanced datasets.

This repository contains my learning journey through Machine Learning, with hands-on implementations, experiments, datasets, and practical examples using Python and popular data science libraries.

## Topics Covered

| No. | Topic | Concepts |
|---|---|---|
| 01 | Data Gathering | Web Scraping, APIs, CSV, JSON, SQL |
| 02 | Understanding Data | Exploratory Data Analysis (EDA), Univariate and Multivariate Analysis, Pandas Profiling |
| 03 | Feature Engineering | Feature Scaling, Encoding, Column Transformer, Pipelines, Mathematical Transformations, Binning, Binarization, Date-Time Features |
| 04 | Handling Missing Values | Complete Case Analysis, Univariate Imputation, Random Imputation, KNN Imputer, MICE |
| 05 | Handling Outliers | Z-Score Method, IQR Method, Percentile Method |
| 06 | Feature Construction | Feature Construction, Feature Splitting, Principal Component Analysis (PCA), Digit Recognition using PCA |
| 07 | Linear Regression and Regression Metrics | Linear Regression, Multiple Linear Regression, Polynomial Regression, Regression Assumptions, Evaluation Metrics |
| 08 | Gradient Descent | Gradient Descent from Scratch, Batch Gradient Descent, Stochastic Gradient Descent, Mini-Batch Gradient Descent |
| 09 | Regularization and Regression | Ridge (L2), Lasso (L1), ElasticNet, Regularization from Scratch |
| 10 | Logistic Regression and Classification Metrics | Perceptron, Sigmoid Function, Loss Function, Gradient Descent, Classification Metrics, ROC Curve, AUC |
| 11 | Softmax / Multinomial Logistic Regression | Softmax Regression, Polynomial Logistic Regression |
| 12 | Decision Tree / Regression Tree | Decision Tree Classifier and Regression Tree Concepts |
| 13 | Ensemble Learning | Voting, Bagging, Random Forest, AdaBoost, Gradient Boosting, Stacking, Hyperparameter Tuning |
| 14 | Clustering | K-Means, Agglomerative Hierarchical Clustering, K-Nearest Neighbors (KNN), DBSCAN |
| 16 | Naive Bayes | Naive Bayes Classification |
| 17 | Handling Imbalanced Data | Random Undersampling, Random Oversampling, SMOTE, Balanced Random Forest, Class Weights |

## Repository Structure

The repository is organized topic-wise, following the structure of the local folders and notebooks.

```text
Machine-Learning/
│
├── .gitignore
├── README.md
│
├── 01_Data Gathering/
│   ├── Web_Scraping.ipynb
│   ├── Working_with_API.ipynb
│   ├── Working_with_csv.ipynb
│   ├── Working_with_json_SQL.ipynb
│   ├── aug_train.csv
│   ├── train.json
│   └── world.sql
│
├── 02_Understanding_Data/
│   ├── Day_1.ipynb
│   ├── EDA_Multivariate.ipynb
│   ├── EDA_Univariate.ipynb
│   ├── Pandas_Profiling.ipynb
│   └── train.csv
│
├── 03_Feature_Engineeringh/
│   ├── 1_Feature_Scaling.ipynb
│   ├── 2_Encoding.ipynb
│   ├── 3_Cloumn_Transformer.ipynb
│   ├── 4_Pipeline.ipynb
│   ├── 5_pipeline_predict_from_pickle.ipynb
│   ├── 6_Mathematical_Transformations.ipynb
│   ├── 7_Binning_Binarization.ipynb
│   ├── Date_Time.ipynb
│   ├── models/
│   │   ├── clf.pkl
│   │   ├── ohe_embarked.pkl
│   │   └── ohe_sex.pkl
│   ├── pipe.pkl
│   └── [CSV datasets]
│
├── 04_Handling_Missing_Values/
│   ├── 1_Complete_case_Analysis_(removing_missing_rows).ipynb
│   ├── 2_Univariate_Imputation_Numericals.ipynb
│   ├── 3_Univariate_Imputation_Categorical.ipynb
│   ├── 4_Univariate_Imputation_Random.ipynb
│   ├── 5_Multivariate_KNN_Imputer.ipynb
│   ├── 6_Multivariate_MICE.ipynb
│   └── [CSV datasets]
│
├── 05_Handling_Outliers/
│   ├── 1_Outlier_Zscore_Method.ipynb
│   ├── 2_Outlier_IQR_Method.ipynb
│   ├── 3_Outlier_Percentile_Method.ipynb
│   └── [CSV datasets]
│
├── 06_Feature_Construction/
│   ├── 1_Feature_Construction_and_Splitting.ipynb
│   ├── 2_PCA.ipynb
│   ├── 3_PCA_Digit_Recognition.ipynb
│   └── [CSV datasets]
│
├── 07_Linear_Regression_and_Regression_Matrices/
│   ├── 1_Assumptions_of_Linear_Regression.ipynb
│   ├── 2_Linear_Regression_and_Evaluation_Matrices.ipynb
│   ├── 3_MeraMLR.ipynb
│   ├── 4_MeraLR.ipynb
│   ├── 5_MultiLinear_and_stochastic.ipynb
│   ├── 6_Polynomial_Linear_Regression.ipynb
│   └── [CSV datasets]
│
├── 08_Gradient_Descent/
│   ├── 1_m_b_handling.ipynb
│   ├── 2_GD_Class.ipynb
│   ├── 3_Batch_GD.ipynb
│   ├── 4_Stochastic_GD.ipynb
│   └── 5_Mini-Batch_GD.ipynb
│
├── 09_Regularization_and_Regression/
│   ├── 1_L2_Ridge_Regularization_and_Regression/
│   │   ├── 1_Ridge.ipynb
│   │   ├── 2_Ridge_from_Scratch.ipynb
│   │   ├── 3_Ridge_Regression_from_Scratch_for_2D.ipynb
│   │   ├── 4_Ridge_Regression_from_Scratch_for_nD.ipynb
│   │   ├── 5_Ridge_Regression_from_Scratch_GD.ipynb
│   │   └── Ridge_Lasso_Regression.ipynb
│   │
│   ├── 2_L1_Lasso_Regularization_and_Regression/
│   │   └── 1_Lasso_Regression_and_Keypoints.ipynb
│   │
│   └── 3_ElasticNet_Regression/
│       └── 1_Combinaton_Of_Ridge_And_Lasso.ipynb
│
├── 10_Logistic_Regression_and_Classification_Matrices/
│   ├── 1_Perceptron_and_Sigmoid.ipynb
│   ├── 2_Loss_Function_GD.ipynb
│   ├── 3_Matrices.ipynb
│   ├── 4_ROC_AUC_Matrice_curve.ipynb
│   └── train.csv
│
├── 11_SoftMax_aka_MultiNomial_Logistic_Regression/
│   ├── 1_SoftMax.ipynb
│   ├── 2_Polynomial_Logistic_Regression.ipynb
│   └── ushape.csv
│
├── 12_Decision_Tree_aka_Regression_Tree/
│   └── 1_Decision_Tree.ipynb
│
├── 13_Ensemble_Learning/
│   ├── 1_Voting/
│   │   └── 1_Voting_Classifier_and_Regressor.ipynb
│   │
│   ├── 2_Bagging/
│   │   ├── 1_Bagging_Ensembles.ipynb
│   │   ├── 2_Bagging_Classifier.ipynb
│   │   └── 3_Bagging_Regression.ipynb
│   │
│   ├── 3_Random_Forest_A_Bagging_Technique/
│   │   ├── 1_RF_Learning.ipynb
│   │   ├── 2_RF_Demo.ipynb
│   │   ├── 3_Bagging_vs_RF.ipynb
│   │   ├── 4_Hyperparameters_Tuning.ipynb
│   │   ├── 5_OOB_Score.ipynb
│   │   ├── 6_Feature_Importance_Calculations.ipynb
│   │   └── heart.csv
│   │
│   ├── 4_Adaboost/
│   │   ├── 1_Adaboost_Demo.ipynb
│   │   └── 2_Hyperparameter_Tuning.ipynb
│   │
│   ├── 5_Gradient_Boosting/
│   │   ├── 1_Gradient_Boosting_Regression.ipynb
│   │   └── 2_GB_Classification.ipynb
│   │
│   └── 6_Stacking/
│       ├── 1_Stacking.ipynb
│       └── heart.csv
│
├── 14_Clustering/
│   ├── 1_K-Mean_Clustering/
│   │   ├── 1_K-Mean_Clustering.ipynb
│   │   ├── 2_K-Mean_Clustering_Class_Building.ipynb
│   │   └── student_clustering.csv
│   │
│   ├── 2_Agglomerative_Hierarchical_Clustering/
│   │   ├── 1_Hierarchical_Clustering.ipynb
│   │   └── hierarchical-clustering-with-python-and-scikit-learn-shopping-data.csv
│   │
│   ├── 3_KNN/
│   │   ├── 1_KNN.ipynb
│   │   └── data.csv
│   │
│   └── 4_DBScan/
│       └── 1_DBScan.ipynb
│
├── 16_Naive_Bayes/
│   ├── 1_NB.ipynb
│   └── play_tennis.csv
│
└── 17_Handling_Imbalance_Data/
    ├── 1_Random_Undersampling.ipynb
    ├── 2_Random_Oversampling.ipynb
    ├── 3_SMOTE.ipynb
    ├── 4_Ensemble_Balanced_Random_Forest.ipynb
    └── 5_Class_Weights.ipynb
```

*Note: The `[CSV datasets]` entries group the supporting dataset files to keep the tree readable. Their actual filenames remain in their respective folders. The numbering and directory names above follow the current local repository.*

## Tech Stack

- **Programming Language:** Python
- **Development Environment:** Jupyter Notebook
- **Data Manipulation:** NumPy, Pandas
- **Data Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Data Collection:** Requests, Web Scraping, APIs
- **Model Persistence:** Pickle

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Udaykaran-Singh/Machine-Learning.git
```

### 2. Navigate to the Project

```bash
cd Machine-Learning
```

### 3. Create a Virtual Environment (Optional)

```bash
python -m venv .venv
```

Activate it:

**macOS / Linux**
```bash
source .venv/bin/activate
```

**Windows**
```bash
.venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter requests
```

### 5. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open the relevant topic folder and run the notebooks.

## Learning Resource

The concepts and practical implementations in this repository are based on my self-learning and practice, including the **CampusX Machine Learning course**.

- [CampusX – YouTube Channel](https://www.youtube.com/@campusx-official)

## Purpose

The goal of this repository is to maintain an organized collection of my Machine Learning notes, practical implementations, experiments, and coding exercises while strengthening my understanding of core Machine Learning algorithms and techniques.

## Author

**Udaykaran Singh**

B.Tech – Data Science  
National Institute of Technology, Jalandhar

- GitHub: [Udaykaran-Singh](https://github.com/Udaykaran-Singh)

---

⭐ If you find this repository useful, feel free to explore the notebooks and learn along!
