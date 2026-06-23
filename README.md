# 📚 Scikit-Learn Cookbook

> **Tugas 2 - Enrichment for Machine Learning Classes (Individual Task)**  
> "Code Reproduction + Theoretical Deep-Dive from scikit-learn Cookbook (O'Reilly)"

A comprehensive, hands-on collection of Jupyter Notebooks covering the full machine learning pipeline — from data preprocessing to model deployment — using [scikit-learn](https://scikit-learn.org/).

---

## 📖 Table of Contents

| #  | Chapter | Topics Covered |
|----|---------|----------------|
| 01 | **API Elements of scikit-learn** | Estimators, Transformers, Pipelines, Datasets API, fit/predict paradigm |
| 02 | **Data Preprocessing** | Scaling, Normalization, Encoding, Imputation, ColumnTransformer |
| 03 | **Dimensionality Reduction** | PCA, LDA, t-SNE, Feature Selection, VarianceThreshold, RFE |
| 04 | **Models with Distance Metrics and Nearest Neighbors** | KNN Classification & Regression, Distance Metrics, Ball Tree, KD-Tree |
| 05 | **Linear Models and Regularization** | Linear Regression, Ridge, Lasso, ElasticNet, SGD |
| 06 | **Advanced Logistic Regression** | Binary & Multiclass Classification, ROC Curves, Class Imbalance |
| 07 | **Support Vector Machines and Kernel Methods** | SVC, SVR, Kernel Trick (RBF/Poly/Linear), GridSearchCV |
| 08 | **Tree-Based Algorithms and Ensemble Methods** | Decision Trees, Random Forest, Gradient Boosting, AdaBoost, Voting |
| 09 | **Text Processing and Multiclass Classification** | CountVectorizer, TF-IDF, N-grams, Naive Bayes, LinearSVC |
| 10 | **Clustering Techniques** | K-Means, DBSCAN, Hierarchical, Silhouette Score, Elbow Method |
| 11 | **Novelty and Outlier Detection** | Isolation Forest, One-Class SVM, LOF, Elliptic Envelope |
| 12 | **Cross-Validation and Model Evaluation** | K-Fold, GridSearchCV, RandomizedSearchCV, Metrics, Learning Curves |
| 13 | **Deploying Models in Production** | joblib/pickle, Production Pipelines, Model Versioning, Batch Inference |

---

## 📝 Chapter Summaries

### Chapter 1: API Elements of scikit-learn
Introduces the core scikit-learn API design: **Estimator** (learns via `fit()`), **Transformer** (transforms via `transform()`), and **Predictor** (predicts via `predict()`). All objects follow a consistent interface enabling composability through **Pipelines**, which prevent data leakage by fitting transformers only on training data. The Datasets API provides toy datasets and synthetic generators for quick experimentation.

### Chapter 2: Data Preprocessing
Covers essential data transformation techniques before model training. **StandardScaler** normalizes features to zero mean/unit variance; **RobustScaler** handles outliers using median/IQR; **OneHotEncoder** handles categorical variables without ordinal assumptions. **SimpleImputer** and **KNNImputer** handle missing values. **ColumnTransformer** applies different preprocessing to different feature types in a single pipeline.

### Chapter 3: Dimensionality Reduction
**PCA** (unsupervised) finds directions of maximum variance via eigendecomposition — always standardize first. **LDA** (supervised) maximizes between-class separation (max n_classes−1 components). **t-SNE** is a non-linear technique for **visualization only** — cannot transform new data. Feature selection methods (**VarianceThreshold**, **SelectKBest**, **RFE**) reduce dimensionality while preserving interpretability.

### Chapter 4: Models with Distance Metrics and Nearest Neighbors
KNN is a **lazy learner**: no explicit training phase, all computation at prediction time. Distance metrics (Euclidean, Manhattan, Minkowski) directly impact performance — **always scale features** before KNN. The optimal k is found via cross-validation. `weights='distance'` often outperforms uniform weighting. **Ball Tree** is more efficient for high-dimensional data.

### Chapter 5: Linear Models and Regularization
Linear models form the foundation of ML. **Ridge (L2)** shrinks coefficients uniformly, preventing multicollinearity. **Lasso (L1)** produces sparse solutions (automatic feature selection). **ElasticNet** combines both — often the best choice. Regularization strength is controlled by alpha (or C=1/alpha in SVM convention). **SGD** enables training on large-scale datasets via stochastic updates.

### Chapter 6: Advanced Logistic Regression
Despite its name, Logistic Regression is a **classifier** using the sigmoid function to output probabilities. **C** (inverse regularization) is key: small C → strong regularization. **ROC-AUC** is preferred over accuracy for imbalanced data. Multiclass is handled via OvR or Multinomial (softmax). `class_weight='balanced'` is crucial for imbalanced datasets.

### Chapter 7: Support Vector Machines and Kernel Methods
SVM finds the **maximum margin hyperplane** separating classes. **C** controls the margin-misclassification tradeoff. The **kernel trick** implicitly maps data to higher dimensions: **RBF** handles most non-linear problems, **polynomial** captures feature interactions. **gamma** controls the influence radius of each training point. Always standardize before SVM. **SVR** extends SVM to regression with epsilon-insensitive loss.

### Chapter 8: Tree-Based Algorithms and Ensemble Methods
Decision Trees split data using impurity measures (Gini/Entropy). Deep trees overfit; control via `max_depth`. **Random Forest** uses bagging + random feature subsets → reduced variance, robust performance. **Gradient Boosting** sequentially corrects residuals via weak learners → strong bias reduction but requires careful tuning. **AdaBoost** reweights misclassified samples. Feature importance from tree methods enables effective feature selection.

### Chapter 9: Text Processing and Multiclass Classification
Text must be converted to numerical representations. **CountVectorizer** creates raw term frequency matrices. **TF-IDF** downweights common words (the, is, a) and highlights distinctive terms. **N-grams** capture local context. **Multinomial Naive Bayes** is a strong baseline for text. **Linear SVC** + TF-IDF is often the best combination for text classification.

### Chapter 10: Clustering Techniques
**K-Means** minimizes inertia (within-cluster sum of squares) — fast and scalable but requires specifying k and assumes spherical clusters. The **Elbow method** and **Silhouette score** help select optimal k. **DBSCAN** automatically discovers clusters of arbitrary shape and identifies noise points — no need to specify k. **Hierarchical clustering** builds a tree of clusters visualized via dendrogram.

### Chapter 11: Novelty and Outlier Detection
Anomaly detection identifies data points that deviate significantly from normal distribution. **Isolation Forest** isolates anomalies using random trees — points requiring fewer splits are outliers. **One-Class SVM** learns a boundary around normal data in feature space. **LOF** compares local densities — points with much lower density than neighbors are flagged. **EllipticEnvelope** assumes Gaussian distribution and uses Mahalanobis distance.

### Chapter 12: Cross-Validation and Model Evaluation
Reliable model evaluation requires CV. **StratifiedKFold** preserves class distribution across folds. **GridSearchCV** exhaustively searches all hyperparameter combinations; **RandomizedSearchCV** samples n random combinations — more efficient for large search spaces. Classification metrics: Accuracy (misleading for imbalanced), F1-score (balance precision-recall), ROC-AUC (threshold-independent). Learning curves diagnose underfitting vs overfitting.

### Chapter 13: Deploying Models in Production
**joblib** is preferred over pickle for scikit-learn models (efficient numpy compression). Always save the **complete pipeline** (preprocessing + model) to ensure consistency between training and inference. Store **model metadata** (version, hyperparameters, metrics, timestamp) in JSON for reproducibility. Batch inference is far more efficient than single-sample prediction. Implement model versioning with clear naming schemes for A/B testing.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Jupyter Notebook or JupyterLab

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/scikit-learn-cookbook.git
cd scikit-learn-cookbook

# Install dependencies
pip install scikit-learn numpy pandas matplotlib seaborn scipy jupyter joblib

# Launch Jupyter
jupyter notebook
```

### Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `scikit-learn` | ≥1.0 | Machine learning algorithms |
| `numpy` | ≥1.21 | Numerical computing |
| `pandas` | ≥1.3 | Data manipulation |
| `matplotlib` | ≥3.4 | Visualization |
| `seaborn` | ≥0.11 | Statistical visualization |
| `scipy` | ≥1.7 | Scientific computing |
| `joblib` | ≥1.0 | Model serialization |

---

## 🗂️ Repository Structure

```
scikit-learn-cookbook/
├── 01. API Elements of scikit-learn/
│   └── chapter_01_api_elements.ipynb
├── 02. Data Preprocessing/
│   └── chapter_02_preprocessing.ipynb
├── 03. Dimensionality Reduction/
│   └── chapter_03_dimensionality_reduction.ipynb
├── 04. Models with Distance Metrics and Nearest Neighbors/
│   └── chapter_04_knn.ipynb
├── 05. Linear Models and Regularization/
│   └── chapter_05_linear_models.ipynb
├── 06. Advanced Logistic Regression/
│   └── chapter_06_logistic_regression.ipynb
├── 07. Support Vector Machines and Kernel Methods/
│   └── chapter_07_svm.ipynb
├── 08. Tree-Based Algorithms and Ensemble Methods/
│   └── chapter_08_trees_ensemble.ipynb
├── 09. Text Processing and Multiclass Classification/
│   └── chapter_09_text_processing.ipynb
├── 10. Clustering Techniques/
│   └── chapter_10_clustering.ipynb
├── 11. Novelty and Outlier Detection/
│   └── chapter_11_outlier_detection.ipynb
├── 12. Cross-Validation and Model Evaluation/
│   └── chapter_12_crossval_evaluation.ipynb
├── 13. Deploying Models in Production/
│   └── chapter_13_deploying_models.ipynb
└── README.md
```

Each notebook contains:
- **Summary**: Overview of what the chapter covers
- **Theoretical Deep-Dive**: Mathematical foundations and conceptual explanations
- **Code Reproduction**: Step-by-step implementations from the book with annotations

---

## 💡 How to Use This Cookbook

1. **Sequential learning** — Work through chapters 01–13 for a complete ML curriculum
2. **Recipe-based** — Jump to any chapter for targeted techniques
3. **Reference** — Use notebooks as quick references for specific scikit-learn APIs

---

## 📄 Reference

- [scikit-learn Cookbook (O'Reilly)](https://www.oreilly.com/library/view/scikit-learn-cookbook/9781789955750/)
- [Official scikit-learn Documentation](https://scikit-learn.org/stable/)
