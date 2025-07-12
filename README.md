# COVID-19 Fake News Classification

A machine learning project for classifying COVID-19 related news as real or fake using advanced NLP techniques and ensemble methods.

## 📊 Project Overview

This project implements an optimized machine learning pipeline to detect fake news related to COVID-19. The solution combines deep learning (LSTM) with traditional machine learning algorithms to achieve robust classification performance.

## 🎯 Key Features

- **Advanced Text Preprocessing**: Lemmatization, stopword removal, and URL cleaning
- **Deep Learning Model**: Optimized LSTM architecture with dropout regularization
- **Ensemble Method**: Combines LSTM, Logistic Regression, and Naive Bayes
- **Performance Optimization**: Early stopping and learning rate reduction
- **Comprehensive Evaluation**: Multiple metrics and visualization

## 📈 Model Performance

| Model | F1 Score | ROC AUC | Precision | Recall |
|-------|----------|---------|-----------|--------|
| **LSTM (Optimized)** | **0.5652** | **0.9047** | 0.91 | 0.41 |
| Logistic Regression | 0.3761 | - | - | - |
| Naive Bayes | 0.1188 | - | - | - |
| **Ensemble** | 0.3898 | **0.9178** | 1.00 | 0.24 |

## 🛠️ Technologies Used

- **Python 3.11**
- **TensorFlow 2.19**: Deep learning framework
- **scikit-learn**: Traditional ML algorithms
- **NLTK**: Natural language processing
- **pandas & numpy**: Data manipulation
- **matplotlib & seaborn**: Visualization

## 📋 Requirements

```
tensorflow>=2.19.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
nltk>=3.8.0
```

## 🚀 Installation & Usage

1. **Clone the repository**
   ```bash
   git clone https://github.com/RayendraNagata/Covid-19-Fake-News-Detection.git
   cd "Covid 19 Fake News Classification"
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate  # Windows
   # source .venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download dataset**
   - Place your COVID-19 fake news dataset as `COVID Fake News Data.csv`
   - Dataset should have columns: `headlines` (text) and `outcome` (0/1)

5. **Run the classification**
   ```bash
   python fakenews_classification_correct.py
   ```

## 📁 Project Structure

```
Covid 19 Fake News Classification/
├── fakenews_classification_correct.py  # Main script
├── COVID Fake News Data.csv           # Dataset (not included)
├── requirements.txt                   # Dependencies
├── README.md                         # Project documentation
└── .venv/                           # Virtual environment
```

## 🔧 Model Architecture

### LSTM Model
- **Embedding Layer**: 5000 vocab → 128 dimensions
- **Stacked LSTM**: 128 → 64 units with dropout (0.3)
- **Dense Layers**: 64 → 32 → 1 with dropout (0.5, 0.3)
- **Optimization**: Adam optimizer with early stopping

### Ensemble Method
- Combines predictions from LSTM, Logistic Regression, and Naive Bayes
- Uses soft voting (average probabilities)
- Achieves highest ROC AUC score (0.9178)

## 📊 Features

### Text Preprocessing
- URL, mention, and hashtag removal
- Lemmatization with WordNet
- Stopword filtering
- Minimum word length filtering (>2 characters)

### Model Optimization
- Early stopping (patience=3)
- Learning rate reduction (factor=0.2)
- Stratified train-test split
- Comprehensive evaluation metrics

## 🎨 Visualizations

The script generates:
- Training history plots (accuracy & loss)
- Confusion matrices for all models
- Performance comparison charts

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- TensorFlow team for the deep learning framework
- NLTK contributors for NLP tools
- scikit-learn developers for ML algorithms

## 📞 Contact

For questions or suggestions, please open an issue in the repository.

---
⭐ **Star this repository if you found it helpful!**
