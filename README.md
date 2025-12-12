# Twitter Rumor Detection using Machine Learning

A comprehensive machine learning project for detecting and classifying rumors in Twitter data using multiple classification algorithms. This project implements 11 different machine learning models to classify tweets into four categories: False, True, Unverified, and Non-rumor.

## 🎯 Project Overview

This project focuses on rumor detection in social media, specifically Twitter, using natural language processing and machine learning techniques. The system analyzes tweet content to automatically classify rumors and assess their veracity.

### Key Features
- **Multi-class Classification**: Classifies tweets into 4 categories (False, True, Unverified, Non-rumor)
- **11 Machine Learning Models**: Comprehensive comparison of different algorithms
- **Automated Model Selection**: Identifies top 3 performing models based on test accuracy
- **Comprehensive Evaluation**: Detailed metrics, confusion matrices, and classification reports
- **Data Integration**: Combines Twitter15 and Twitter16 datasets

## 📊 Dataset

The project uses combined Twitter datasets from two sources:
- **Twitter15**: Rumor detection dataset from 2015
- **Twitter16**: Rumor detection dataset from 2016

### Dataset Structure
- **Total Tweets**: Combined from both datasets
- **Features**: Tweet ID, Tweet Text, Label, Dataset Source
- **Labels**: 
  - `False`: False rumors
  - `True`: True information
  - `Unverified`: Unverified claims
  - `Non-rumor`: Non-rumor content

## 🏗️ Project Structure

```
├── README.md                           # Project documentation
├── combined_twitter_dataset.xlsx       # Combined dataset
├── combine_twitter_data.py            # Data combination script
├── select_best_three_models.py        # Model comparison and selection
├── add_numeric_labels.py              # Label preprocessing
├── analyze_existing_results.py        # Results analysis
│
├── twitter15/                         # Twitter 2015 dataset
│   ├── source_tweets.txt
│   └── label.txt
│
├── twitter16/                         # Twitter 2016 dataset
│   ├── source_tweets.txt
│   └── label.txt
│
├── Models/                            # Machine learning models
│   ├── utils.py                       # Common utilities
│   ├── multinomial_naive_bayes.py
│   ├── bernoulli_naive_bayes.py
│   ├── gaussian_naive_bayes.py
│   ├── multinomial_logistic_regression.py
│   ├── ovr_logistic_regression.py
│   ├── ovo_logistic_regression.py
│   ├── linear_discriminant_analysis.py
│   ├── quadratic_discriminant_analysis.py
│   ├── softmax_regression.py
│   ├── voting_classifier.py
│   └── stacking_classifier.py
│
├── Results/                           # Individual model results
│   ├── MultinomialNB/
│   ├── BernoulliNB/
│   ├── GaussianNB/
│   ├── MultinomialLogisticRegression/
│   ├── OvRLogisticRegression/
│   ├── OvOLogisticRegression/
│   ├── LinearDiscriminantAnalysis/
│   ├── QuadraticDiscriminantAnalysis/
│   ├── SoftmaxRegression/
│   ├── VotingClassifier/
│   └── StackingClassifier/
│
└── Best_of_three/                     # Top 3 models results
    ├── best_three_summary.txt
    ├── best_three_detailed.csv
    ├── best_three_comparison.csv
    ├── Rank_1_SoftmaxRegression/
    ├── Rank_2_StackingClassifier/
    └── Rank_3_VotingClassifier/
```

## 🤖 Machine Learning Models

The project implements and compares 11 different machine learning algorithms:

### 1. **Naive Bayes Family**
- **Multinomial Naive Bayes**: Best for text classification with word counts
- **Bernoulli Naive Bayes**: Binary feature representation
- **Gaussian Naive Bayes**: Assumes normal distribution of features

### 2. **Logistic Regression Variants**
- **Multinomial Logistic Regression**: Multi-class logistic regression
- **One-vs-Rest (OvR) Logistic Regression**: Binary classifiers for each class
- **One-vs-One (OvO) Logistic Regression**: Pairwise binary classifiers

### 3. **Discriminant Analysis**
- **Linear Discriminant Analysis (LDA)**: Linear decision boundaries
- **Quadratic Discriminant Analysis (QDA)**: Quadratic decision boundaries

### 4. **Neural Networks**
- **Softmax Regression**: Single-layer neural network (no hidden layers)

### 5. **Ensemble Methods**
- **Voting Classifier**: Combines multiple algorithms via voting
- **Stacking Classifier**: Meta-learning approach with multiple base models

## 🏆 Model Performance

Based on comprehensive evaluation, the top 3 performing models are:

| Rank | Model | Test Accuracy | Test F1-Macro | Test F1-Weighted |
|------|-------|---------------|---------------|------------------|
| 🥇 1 | **Softmax Regression** | **82.24%** | **81.69%** | **82.14%** |
| 🥈 2 | **Stacking Classifier** | **81.07%** | **80.48%** | **80.95%** |
| 🥉 3 | **Voting Classifier** | **80.61%** | **79.02%** | **80.06%** |

### Winner: Softmax Regression
- **Test Accuracy**: 82.24%
- **Precision (Macro)**: 82.61%
- **Recall (Macro)**: 81.26%
- **F1-Score (Macro)**: 81.69%

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn openpyxl
```

### Installation
1. **Clone the repository**
```bash
git clone https://github.com/leekhithnunna/Twitter_15_16_Rumour_Verification.git
cd Twitter_15_16_Rumour_Verification
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Usage

#### 1. Prepare the Dataset
```bash
python combine_twitter_data.py
```
This combines Twitter15 and Twitter16 datasets into a single Excel file.

#### 2. Add Numeric Labels
```bash
python add_numeric_labels.py
```
Converts text labels to numeric format for machine learning.

#### 3. Run Individual Models
```bash
# Run a specific model
python Models/softmax_regression.py

# Or run any other model
python Models/multinomial_naive_bayes.py
```

#### 4. Compare All Models and Select Best 3
```bash
python select_best_three_models.py
```
This script:
- Runs all 11 models
- Compares their performance
- Selects top 3 based on test accuracy
- Saves results to `Best_of_three/` folder

#### 5. Analyze Results
```bash
python analyze_existing_results.py
```

## 📈 Feature Engineering

### Text Preprocessing
- **Lowercasing**: Convert all text to lowercase
- **URL Removal**: Remove HTTP links
- **Mention Removal**: Remove @username mentions
- **Hashtag Removal**: Remove #hashtag symbols
- **Punctuation Removal**: Clean special characters

### Feature Extraction
- **TF-IDF Vectorization**: Term Frequency-Inverse Document Frequency
- **N-grams**: Unigrams and bigrams (1,2)
- **Max Features**: 5000 most important features
- **Stop Words**: English stop words removal

## 📊 Evaluation Metrics

Each model is evaluated using comprehensive metrics:

### Primary Metrics
- **Accuracy**: Overall classification accuracy
- **Precision**: Macro and weighted averages
- **Recall**: Macro and weighted averages
- **F1-Score**: Macro and weighted averages

### Visualizations
- **Confusion Matrix**: Classification performance per class
- **Metrics Comparison**: Training vs Test performance
- **Classification Report**: Detailed per-class metrics

## 🔍 Results Analysis

### Model Outputs
Each model generates:
- **Metrics File**: `{model}_metrics.txt`
- **Classification Reports**: Training and test set reports
- **Confusion Matrices**: Visual performance analysis
- **Comparison Charts**: Training vs test metrics

### Best Models Analysis
The `Best_of_three/` folder contains:
- **Summary Report**: `best_three_summary.txt`
- **Detailed Data**: `best_three_detailed.csv`
- **Comparison Data**: `best_three_comparison.csv`
- **Individual Results**: Complete results for top 3 models

## 🛠️ Technical Implementation

### Data Pipeline
1. **Data Loading**: Load Twitter15 and Twitter16 datasets
2. **Data Combination**: Merge datasets with source tracking
3. **Text Preprocessing**: Clean and normalize tweet text
4. **Feature Extraction**: Convert text to numerical features
5. **Train-Test Split**: 80-20 split with stratification
6. **Model Training**: Train multiple algorithms
7. **Evaluation**: Comprehensive performance assessment

### Model Architecture
- **Input**: TF-IDF vectorized tweet text (5000 features)
- **Output**: 4-class classification (False, True, Unverified, Non-rumor)
- **Validation**: Stratified train-test split
- **Optimization**: Grid search and cross-validation ready

## 📋 Requirements

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
openpyxl>=3.0.0
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Twitter15 and Twitter16 datasets for rumor detection research
- Scikit-learn community for machine learning tools
- Open source contributors and researchers in NLP and social media analysis

## 📞 Contact

- **Author**: Leekhith Nunna
- **GitHub**: [@leekhithnunna](https://github.com/leekhithnunna)
- **Project Link**: [https://github.com/leekhithnunna/Twitter_15_16_Rumour_Verification](https://github.com/leekhithnunna/Twitter_15_16_Rumour_Verification)

---

## 🔬 Research Applications

This project can be extended for:
- **Real-time Rumor Detection**: Deploy models for live Twitter monitoring
- **Cross-platform Analysis**: Extend to other social media platforms
- **Temporal Analysis**: Study rumor propagation over time
- **Network Analysis**: Incorporate user interaction patterns
- **Deep Learning**: Implement transformer-based models (BERT, RoBERTa)

## 📚 Future Enhancements

- [ ] Real-time Twitter API integration
- [ ] Deep learning models (LSTM, BERT)
- [ ] User network features
- [ ] Temporal propagation analysis
- [ ] Multi-language support
- [ ] Web interface for model deployment
- [ ] API endpoint for rumor detection service

---

*Last Updated: December 2025*