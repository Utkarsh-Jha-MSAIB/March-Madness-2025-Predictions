# 🏀 March Data Crunch Madness 2025

This repository contains the full pipeline and analysis for predicting NCAA March Madness outcomes using historical statistics, PCA, macroeconomic indicators, and Reddit sentiment. It includes both data processing and machine learning workflows, complemented by a deep dive presentation.

## 🏆 Prediction Results

✅ This project successfully predicted the **Final Four**, **Final Two**, and the **National Champion** of March Madness 2025!

## 🎯 Key Highlights (from [Presentation.pptx](./output/Presentation.pptx))

- **Multi-source Modeling**: Combines performance stats, economic signals, and fan sentiment
- **Principal Component Analysis**: Reduces dimensionality and noise from correlated features
- **ChatGPT-Generated Variables**: Uses generative AI to create 500+ optimized feature combinations
- **Sentiment Modeling**: Reddit comments (positive only) enhance predictions with real-world buzz
- **Model Validation**: Train-test split using **year-based sampling** to simulate future tournaments

### 📊 Notebooks

| Notebook | Description |
|----------|-------------|
| [Data Processing.ipynb](./Data%20Processing.ipynb) | Cleans raw input datasets, performs feature engineering, and prepares macro-level variables like inflation and trend proxies |
| [Modeling Data Preparation.ipynb](./Modeling%20Data%20Preparation.ipynb) | Combines engineered features, generates PCA components (PC1–PC4), and builds modeling-ready training and test datasets |
| [Modeling Pipeline.ipynb](./Modeling%20Pipeline.ipynb) | Trains and evaluates multiple models (XGBoost, RF, NN, Logistic Regression) across 500+ variable combinations |
| [Redddit_Scraping.ipynb](./Redddit_Scraping.ipynb) | Extracts Reddit sentiment data from fan discussions to quantify buzz and public support for teams |

## 🧠 Models Used

- `XGBoostClassifier`
- `RandomForestClassifier`
- `MLPClassifier` (Neural Net)
- `LogisticRegression`

Each model was evaluated on:
- Accuracy, Precision, Recall, F1 Score, AUC, Log Loss
- Visualized with ROC Curves and Feature Importances

## 📎 Example Output

Outputs include:
- Excel summaries with top 5 models highlighted
- PCA-transformed datasets
- Final predictions for 2025

## 📌 Getting Started

1. Clone this repo
2. Install dependencies via `pip install -r requirements.txt`
3. Run notebooks in order:
   - Data Processing → Modeling Data Preparation → Modeling Pipeline → Reddit Scraping

## 📬 Contact

For collaboration or feedback, please open an issue or reach out via GitHub Discussions.


