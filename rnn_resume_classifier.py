"""End-to-end RNN for Resume Classification.

This module trains an RNN/LSTM model on the resume dataset to classify
resumes into job categories (HR, IT, Finance, etc.).
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def load_resume_data(csv_path: str, sample_size: int = None) -> Tuple[pd.DataFrame, int]:
    """Load resume dataset from CSV.
    
    Args:
        csv_path: Path to Resume.csv
        sample_size: Optional limit on number of samples to load
        
    Returns:
        DataFrame with resume text and category, number of classes
    """
    print(f"Loading resume dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    if sample_size:
        df = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    print(f"Loaded {len(df)} resumes")
    print(f"Categories: {df['Category'].nunique()}")
    print(f"Sample categories: {df['Category'].unique()[:5]}")
    
    return df, df['Category'].nunique()


def preprocess_text(text: str, max_length: int = 500) -> str:
    """Clean and preprocess resume text.
    
    Args:
        text: Raw resume text
        max_length: Maximum words to keep
        
    Returns:
        Preprocessed text
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    import re
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove special characters but keep spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    
    # Limit to max_length words
    words = text.split()[:max_length]
    return ' '.join(words)


def build_rnn_classifier(vocab_size: int, num_classes: int, 
                          embedding_dim: int = 64, 
                          max_length: int = 500) -> models.Model:
    """Build RNN model for text classification.
    
    Args:
        vocab_size: Size of vocabulary
        num_classes: Number of job categories
        embedding_dim: Embedding dimension
        max_length: Maximum sequence length
        
    Returns:
        Compiled Keras model
    """
    model = models.Sequential([
        layers.Input(shape=(max_length,)),
        
        # Embedding layer
        layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        
        # Bidirectional LSTM layers
        layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
        layers.Dropout(0.3),
        
        layers.Bidirectional(layers.LSTM(64, return_sequences=False)),
        layers.Dropout(0.3),
        
        # Dense classification head
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax'),
    ])
    
    return model


def train_rnn_resume_classifier(csv_path: str,
                                epochs: int = 10,
                                batch_size: int = 32,
                                sample_size: int = None,
                                model_path: str = 'rnn_resume_classifier.h5',
                                verbose: int = 1) -> Tuple[models.Model, dict]:
    """Train RNN on resume classification task.
    
    Args:
        csv_path: Path to Resume.csv
        epochs: Number of training epochs
        batch_size: Batch size
        sample_size: Optional subset of data to use
        model_path: Path to save model
        verbose: Verbosity level
        
    Returns:
        Trained model and training metrics
    """
    # Load data
    df, num_classes = load_resume_data(csv_path, sample_size)
    
    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(df['Category'])
    y = tf.keras.utils.to_categorical(y, num_classes)
    
    print(f"Classes: {le.classes_}")
    
    # Preprocess text
    print("\nPreprocessing resume text...")
    X_text = df['Resume_str'].apply(lambda x: preprocess_text(x)).values
    
    # Tokenize
    print("Tokenizing text...")
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=5000, oov_token='<OOV>')
    tokenizer.fit_on_texts(X_text)
    X_sequences = tokenizer.texts_to_sequences(X_text)
    X = tf.keras.preprocessing.sequence.pad_sequences(X_sequences, maxlen=500)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Build model
    print("\nBuilding RNN model...")
    model = build_rnn_classifier(
        vocab_size=5000,
        num_classes=num_classes,
        embedding_dim=64,
        max_length=500
    )
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(model.summary())
    
    # Train
    print("\nTraining model...")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path, 
            save_best_only=True, 
            monitor='val_accuracy'
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=3, 
            restore_best_weights=True
        ),
    ]
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=verbose
    )
    
    # Evaluate
    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=verbose)
    
    # Save final model
    final_path = os.path.splitext(model_path)[0] + '_final.h5'
    model.save(final_path)
    
    metrics = {
        'test_loss': float(test_loss),
        'test_accuracy': float(test_acc),
        'num_classes': num_classes,
        'classes': le.classes_.tolist()
    }
    
    return model, {'history': history.history, **metrics}


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train RNN on Resume Dataset')
    parser.add_argument('--csv-path', type=str, 
                        default=r'c:\Users\student\Downloads\resume_dataset2\Resume\Resume.csv')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--sample-size', type=int, default=None,
                        help='Limit to N samples (useful for testing)')
    parser.add_argument('--model-path', type=str, default='rnn_resume_classifier.h5')
    args = parser.parse_args()
    
    model, results = train_rnn_resume_classifier(
        csv_path=args.csv_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        sample_size=args.sample_size,
        model_path=args.model_path
    )
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"Test Loss: {results['test_loss']:.4f}")
    print(f"Classes trained: {results['num_classes']}")
    print(f"{'='*70}")
