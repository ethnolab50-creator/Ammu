"""End-to-End LSTM for Resume Classification.

Complete pipeline for classifying resumes into job categories using LSTM.
"""

import os
import warnings
warnings.filterwarnings('ignore')

from typing import Tuple, Dict, List
import numpy as np
import re


def load_resume_data(csv_path: str, sample_size: int = None) -> Tuple[List[str], List[str], List[str]]:
    """Load resume dataset from CSV.
    
    Args:
        csv_path: Path to Resume.csv
        sample_size: Optional limit on number of samples
        
    Returns:
        Tuple of (texts, categories, unique_categories)
    """
    try:
        import pandas as pd
        print(f"Loading resume dataset from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        if sample_size:
            df = df.sample(n=min(sample_size, len(df)), random_state=42)
        
        print(f"✓ Loaded {len(df)} resumes")
        print(f"✓ Found {df['Category'].nunique()} job categories")
        
        return df['Resume_str'].tolist(), df['Category'].tolist(), df['Category'].unique().tolist()
    
    except ImportError:
        print("⚠ pandas not available, using mock data")
        return get_mock_resume_data(sample_size)


def get_mock_resume_data(sample_size: int = 100) -> Tuple[List[str], List[str], List[str]]:
    """Generate mock resume data for testing without dependencies."""
    
    sample_resumes = {
        'Information-Technology': [
            'Python developer with 5 years experience in Django, Flask, REST APIs, database design',
            'Full stack engineer proficient in Java, Spring Boot, microservices, cloud platforms',
            'Data scientist specializing in machine learning, deep learning, TensorFlow, PyTorch',
        ],
        'HR': [
            'HR manager with expertise in recruitment, employee relations, compliance, training programs',
            'Talent acquisition specialist with experience in staffing, sourcing, onboarding',
            'Human resources business partner focusing on employee development and organizational culture',
        ],
        'Finance': [
            'Financial analyst with CPA certification, experience in auditing, FP&A, budgeting',
            'Investment banker with background in mergers acquisitions, valuations, financial modeling',
            'Accountant specializing in tax planning, accounting standards, financial reporting',
        ],
        'Engineering': [
            'Mechanical engineer with CAD proficiency, manufacturing design, project management',
            'Civil engineer experienced in infrastructure, construction, structural analysis',
            'Electrical engineer with expertise in power systems, automation, electronics',
        ],
        'Healthcare': [
            'Registered nurse with 8 years in ICU, emergency department, critical care',
            'Physician with specialization in cardiology, patient care, medical research',
            'Pharmacist experienced in clinical practice, medication therapy, pharmacy operations',
        ],
    }
    
    texts = []
    categories = []
    
    category_list = list(sample_resumes.keys())
    samples_per_category = (sample_size or 100) // len(category_list)
    
    for category, resumes in sample_resumes.items():
        for i in range(samples_per_category):
            texts.append(resumes[i % len(resumes)])
            categories.append(category)
    
    print(f"✓ Generated {len(texts)} mock resumes")
    print(f"✓ {len(category_list)} job categories")
    
    return texts, categories, category_list


def preprocess_text(text: str, max_length: int = 500) -> str:
    """Clean and preprocess resume text."""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    
    # Limit to max_length words
    words = text.split()[:max_length]
    return ' '.join(words)


def build_vocabulary(texts: List[str], vocab_size: int = 5000) -> Dict[str, int]:
    """Build vocabulary from texts."""
    word_freq = {}
    
    for text in texts:
        words = text.split()
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort by frequency
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    
    # Create vocabulary (word -> index mapping)
    vocab = {word: idx + 1 for idx, (word, _) in enumerate(sorted_words[:vocab_size])}
    
    print(f"✓ Built vocabulary with {len(vocab)} words")
    return vocab


def tokenize_texts(texts: List[str], vocab: Dict[str, int], max_length: int = 500) -> np.ndarray:
    """Convert texts to sequences of integers."""
    sequences = []
    
    for text in texts:
        sequence = [vocab.get(word, 0) for word in text.split()]
        # Pad or truncate
        if len(sequence) < max_length:
            sequence = sequence + [0] * (max_length - len(sequence))
        else:
            sequence = sequence[:max_length]
        sequences.append(sequence)
    
    return np.array(sequences, dtype=np.int32)


def encode_labels(categories: List[str], category_map: List[str]) -> np.ndarray:
    """Convert categories to one-hot encoded labels."""
    num_classes = len(category_map)
    category_to_idx = {cat: idx for idx, cat in enumerate(category_map)}
    
    labels = np.zeros((len(categories), num_classes), dtype=np.int32)
    for i, cat in enumerate(categories):
        labels[i, category_to_idx[cat]] = 1
    
    return labels


def build_lstm_model(vocab_size: int = 5000, 
                     num_classes: int = 24,
                     embedding_dim: int = 64,
                     max_length: int = 500) -> 'Model':
    """Build LSTM model for resume classification.
    
    Returns a compiled Keras model.
    """
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models
        
        model = models.Sequential([
            layers.Input(shape=(max_length,)),
            
            # Embedding layer
            layers.Embedding(vocab_size + 1, embedding_dim, input_length=max_length),
            
            # LSTM layers
            layers.LSTM(256, return_sequences=True, dropout=0.2),
            layers.LSTM(128, return_sequences=True, dropout=0.2),
            layers.LSTM(64, return_sequences=False, dropout=0.2),
            
            # Dense classification head
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax'),
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    except ImportError:
        print("⚠ TensorFlow not available, returning mock model info")
        return None


def train_lstm_resume_classifier(csv_path: str = None,
                                 epochs: int = 10,
                                 batch_size: int = 32,
                                 sample_size: int = None,
                                 model_path: str = 'lstm_resume_classifier.h5',
                                 verbose: int = 1) -> Dict:
    """End-to-end LSTM training pipeline for resume classification.
    
    Args:
        csv_path: Path to Resume.csv (optional, will use mock data if not provided)
        epochs: Number of training epochs
        batch_size: Batch size for training
        sample_size: Number of samples to use
        model_path: Path to save the model
        verbose: Verbosity level
        
    Returns:
        Dictionary with results
    """
    
    print("\n" + "="*70)
    print("END-TO-END LSTM FOR RESUME CLASSIFICATION")
    print("="*70 + "\n")
    
    # Step 1: Load data
    print("STEP 1: Loading Data")
    print("-" * 70)
    
    if csv_path:
        texts, categories, category_list = load_resume_data(csv_path, sample_size)
    else:
        texts, categories, category_list = get_mock_resume_data(sample_size)
    
    num_classes = len(category_list)
    
    # Step 2: Preprocess text
    print("\nSTEP 2: Preprocessing Text")
    print("-" * 70)
    print(f"Preprocessing {len(texts)} resumes...")
    
    processed_texts = [preprocess_text(text) for text in texts]
    print(f"✓ Preprocessing complete")
    
    # Step 3: Build vocabulary
    print("\nSTEP 3: Building Vocabulary")
    print("-" * 70)
    
    vocab = build_vocabulary(processed_texts, vocab_size=5000)
    
    # Step 4: Tokenize
    print("\nSTEP 4: Tokenizing Texts")
    print("-" * 70)
    print(f"Tokenizing {len(processed_texts)} texts...")
    
    X = tokenize_texts(processed_texts, vocab, max_length=500)
    print(f"✓ Tokenization complete - Shape: {X.shape}")
    
    # Step 5: Encode labels
    print("\nSTEP 5: Encoding Labels")
    print("-" * 70)
    print(f"Encoding {num_classes} categories...")
    
    y = encode_labels(categories, category_list)
    print(f"✓ Label encoding complete - Shape: {y.shape}")
    
    # Step 6: Split data
    print("\nSTEP 6: Splitting Data")
    print("-" * 70)
    
    split_idx = int(0.8 * len(X))
    val_idx = int(0.9 * len(X))
    
    X_train, X_val, X_test = X[:split_idx], X[split_idx:val_idx], X[val_idx:]
    y_train, y_val, y_test = y[:split_idx], y[split_idx:val_idx], y[val_idx:]
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Step 7: Build model
    print("\nSTEP 7: Building LSTM Model")
    print("-" * 70)
    print(f"Vocabulary size: 5000")
    print(f"Embedding dimension: 64")
    print(f"LSTM units: 256 → 128 → 64")
    print(f"Dense units: 256 → 128 → {num_classes}")
    print(f"Max sequence length: 500")
    
    model = build_lstm_model(vocab_size=5000, num_classes=num_classes)
    
    if model is not None:
        print("\nModel Architecture:")
        model.summary()
        
        # Step 8: Train
        print("\nSTEP 8: Training Model")
        print("-" * 70)
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose
        )
        
        # Step 9: Evaluate
        print("\nSTEP 9: Evaluating Model")
        print("-" * 70)
        
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        
        print(f"✓ Test Loss: {test_loss:.4f}")
        print(f"✓ Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
        
        # Step 10: Save
        print("\nSTEP 10: Saving Model")
        print("-" * 70)
        
        model.save(model_path)
        print(f"✓ Model saved to {model_path}")
        
        return {
            'model': model,
            'history': history.history,
            'test_loss': float(test_loss),
            'test_accuracy': float(test_acc),
            'num_classes': num_classes,
            'vocab_size': len(vocab),
            'categories': category_list,
        }
    else:
        print("\n⚠ TensorFlow not installed. Install with:")
        print("  pip install tensorflow")
        
        return {
            'model': None,
            'num_classes': num_classes,
            'vocab_size': len(vocab),
            'categories': category_list,
            'message': 'TensorFlow required for training'
        }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='End-to-End LSTM for Resume Classification')
    parser.add_argument('--csv-path', type=str, default=None,
                        help='Path to Resume.csv')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--sample-size', type=int, default=None)
    parser.add_argument('--model-path', type=str, default='lstm_resume_classifier.h5')
    args = parser.parse_args()
    
    results = train_lstm_resume_classifier(
        csv_path=args.csv_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        sample_size=args.sample_size,
        model_path=args.model_path
    )
    
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    if results['model'] is not None:
        print(f"✓ Test Accuracy: {results['test_accuracy']:.4f} ({results['test_accuracy']*100:.2f}%)")
        print(f"✓ Test Loss: {results['test_loss']:.4f}")
    print(f"✓ Classes: {results['num_classes']}")
    print(f"✓ Vocabulary: {results['vocab_size']} words")
    print("="*70 + "\n")
