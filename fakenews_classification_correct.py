# -*- coding: utf-8 -*-
"""FakeNews_Classification_Correct.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1CFa4fsC12soL5n9RvCjlenKQq3Zo8q3n
"""

# Step 1: Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# Import TensorFlow dengan cara yang lebih kompatibel
try:
    import tensorflow as tf
    # Import dengan format yang lebih eksplisit untuk menghindari warning
    import tensorflow.keras.preprocessing.text as keras_text
    import tensorflow.keras.preprocessing.sequence as keras_sequence
    import tensorflow.keras.models as keras_models
    import tensorflow.keras.layers as keras_layers
    import tensorflow.keras.callbacks as keras_callbacks
    
    # Assign ke nama yang biasa digunakan
    Tokenizer = keras_text.Tokenizer
    pad_sequences = keras_sequence.pad_sequences
    Sequential = keras_models.Sequential
    Embedding = keras_layers.Embedding
    LSTM = keras_layers.LSTM
    Dense = keras_layers.Dense
    Dropout = keras_layers.Dropout
    EarlyStopping = keras_callbacks.EarlyStopping
    ReduceLROnPlateau = keras_callbacks.ReduceLROnPlateau
    
    print(f"TensorFlow version: {tf.__version__}")
except ImportError as e:
    print(f"Error importing TensorFlow: {e}")
    print("Please install TensorFlow: pip install tensorflow")
    
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

# Step 2: Load Dataset
try:
    df = pd.read_csv('COVID Fake News Data.csv')
    print("Dataset berhasil dimuat!")
    print(df.head())
    print(f"Shape dataset: {df.shape}")
except FileNotFoundError:
    print("Error: File 'COVID Fake News Data.csv' tidak ditemukan!")
    print("Pastikan file sudah didownload dan ada di direktori yang sama dengan script ini")
    exit()

# Step 3: Advanced Preprocessing

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def advanced_clean_text(text):
    """Enhanced text cleaning with lemmatization and better filtering"""
    if pd.isna(text):
        return ""
    
    # Hapus URL, mention, hashtag
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    
    # Hanya ambil huruf
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    
    # Tokenisasi dan lemmatisasi
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words and len(w) > 2]
    
    return ' '.join(words)

df['clean_text'] = df['headlines'].apply(advanced_clean_text)
print(f"Preprocessing completed. Sample cleaned text: {df['clean_text'].iloc[0][:100]}...")

# Step 4: Text Tokenization and Sequence Padding

tokenizer = Tokenizer(num_words=5000)  # Limit to top 5000 most frequent words
tokenizer.fit_on_texts(df['clean_text'])
X = tokenizer.texts_to_sequences(df['clean_text'])  # Convert text to sequences
X = pad_sequences(X, maxlen=100)  # Pad sequences to uniform length of 100
y = df['outcome'].values  # Target variable

# Step 5: Split Data with Stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 6: Build Optimized LSTM Model with Callbacks

def create_optimized_model():
    """
    Creates an optimized LSTM model with:
    - Embedding layer for text representation
    - Stacked LSTM layers with dropout
    - Dense layers with regularization
    - Binary classification output
    """
    model = Sequential()
    model.add(Embedding(5000, 128, input_length=100))
    model.add(LSTM(128, dropout=0.3, recurrent_dropout=0.3, return_sequences=True))
    model.add(LSTM(64, dropout=0.3, recurrent_dropout=0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(
        loss='binary_crossentropy', 
        optimizer='adam', 
        metrics=['accuracy', 'precision', 'recall']
    )
    return model

# Callbacks untuk optimasi training
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.0001, verbose=1)

model = create_optimized_model()
print("Model architecture:")
model.summary()

history = model.fit(X_train, y_train, 
                   epochs=15, 
                   batch_size=32,
                   validation_data=(X_test, y_test),
                   callbacks=[early_stopping, reduce_lr],
                   verbose=1)

# Step 7: Comprehensive Evaluation

def plot_training_history(history):
    """Plot training and validation accuracy/loss over epochs"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

plot_training_history(history)

# Detailed predictions and metrics
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype("int32")

print("=== LSTM Model Performance ===")
print(classification_report(y_test, y_pred))
print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_prob):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")

# Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("LSTM Model - Confusion Matrix")
plt.show()

# Step 8: Ensemble Model with Traditional ML

# Build TF-IDF features for traditional ML models
print("\n=== Building Ensemble Model ===")
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_tfidf = tfidf.fit_transform(df['clean_text'])
X_tfidf_train, X_tfidf_test, _, _ = train_test_split(X_tfidf, y, test_size=0.2, random_state=42, stratify=y)

# Traditional ML models
lr_model = LogisticRegression(random_state=42, max_iter=1000)
nb_model = MultinomialNB()

# Train traditional models
print("Training Logistic Regression...")
lr_model.fit(X_tfidf_train, y_train)
print("Training Naive Bayes...")
nb_model.fit(X_tfidf_train, y_train)

# Ensemble predictions (voting)
lstm_pred = y_pred_prob.flatten()
lr_pred = lr_model.predict_proba(X_tfidf_test)[:, 1]
nb_pred = nb_model.predict_proba(X_tfidf_test)[:, 1]

ensemble_pred = (lstm_pred + lr_pred + nb_pred) / 3
ensemble_pred_binary = (ensemble_pred > 0.5).astype(int)

print("\n=== Ensemble Model Performance ===")
print(classification_report(y_test, ensemble_pred_binary))
print(f"ROC AUC Score: {roc_auc_score(y_test, ensemble_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, ensemble_pred_binary):.4f}")

# Confusion Matrix untuk Ensemble
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, ensemble_pred_binary), annot=True, fmt='d', cmap='Greens')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Ensemble Model - Confusion Matrix")
plt.show()

# Performance comparison
print("\n=== Model Comparison ===")
print(f"LSTM F1 Score: {f1_score(y_test, y_pred):.4f}")
print(f"Logistic Regression F1 Score: {f1_score(y_test, (lr_pred > 0.5).astype(int)):.4f}")
print(f"Naive Bayes F1 Score: {f1_score(y_test, (nb_pred > 0.5).astype(int)):.4f}")
print(f"Ensemble F1 Score: {f1_score(y_test, ensemble_pred_binary):.4f}")

print("\n=== Optimization Complete! ===")
print("Model optimizations applied:")
print("1. Advanced text preprocessing with lemmatization")
print("2. Deeper LSTM architecture with dropout")
print("3. Early stopping and learning rate reduction")
print("4. Ensemble method with multiple algorithms")
print("5. Comprehensive evaluation metrics")