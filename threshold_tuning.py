import torch
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
from tqdm import tqdm
import os
import logging
import sys
import json

# Import all necessary components from your gs.py file
from gs import (
    TextProcessor, 
    TextDataset, 
    EfficientTextClassifier, 
    TextClassificationTrainer, 
    device
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def find_optimal_thresholds(model, data_loader, label_names, device):
    """
    Evaluate model on a dataset and find optimal thresholds for each class
    """
    model.eval()
    all_probs = []
    all_labels = []
    
    logger.info("Collecting predictions for threshold optimization...")
    with torch.no_grad():
        for batch in tqdm(data_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            
            all_probs.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_probs = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)
    
    # Initialize thresholds dictionary
    optimal_thresholds = {}
    
    logger.info("Finding optimal thresholds for each class...")
    # For each class
    for i, label in enumerate(label_names):
        # Get precision, recall, thresholds
        precision, recall, thresholds = precision_recall_curve(
            all_labels[:, i], all_probs[:, i]
        )
        
        # Calculate F1 scores
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        # Find threshold with best F1 score
        if len(thresholds) > 0:
            best_idx = np.argmax(f1_scores[:-1])  # Last element doesn't have threshold
            best_threshold = thresholds[best_idx]
        else:
            best_threshold = 0.5
        
        optimal_thresholds[label] = float(best_threshold)  # Convert to Python float
        logger.info(f"Optimal threshold for {label}: {best_threshold:.3f}")
    
    # Apply manual overrides based on your observations
    manual_overrides = {
        'very_toxic': 0.700,  # Reduce false positives
        'threat': 0.550,      # Improve recall
    }
    
    for label, threshold in manual_overrides.items():
        if label in optimal_thresholds:
            logger.info(f"Manually overriding {label} threshold: {optimal_thresholds[label]:.3f} → {threshold:.3f}")
            optimal_thresholds[label] = threshold
    
    return optimal_thresholds


class ThresholdTunedPredictor:
    """Predictor that uses class-specific thresholds"""
    
    def __init__(self, model, processor, thresholds, label_names, device):
        self.model = model
        self.processor = processor
        self.thresholds = thresholds
        self.label_names = label_names
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def predict(self, text):
        """Predict with class-specific thresholds"""
        # Preprocess
        preprocessed = self.processor.preprocess(text)
        
        # Tokenize
        encoding = self.processor.tokenize(preprocessed, return_tensors='pt')
        
        # Get predictions
        with torch.no_grad():
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            outputs = self.model(input_ids, attention_mask)
            probabilities = outputs.cpu().numpy()[0]
            
            # Apply thresholds
            predictions = np.zeros_like(probabilities)
            for i, label in enumerate(self.label_names):
                threshold = self.thresholds.get(label, 0.5)
                predictions[i] = 1 if probabilities[i] >= threshold else 0
        
        return predictions, probabilities


def main():
    # Parameters - adjust as needed
    model_path = "text_classifier_model.pt"  # Path to your saved model
    data_path = "FF.csv"  # Path to your dataset
    max_length = 128
    batch_size = 32
    embedding_dim = 128
    hidden_dim = 128
    
    # Device is already imported from gs.py
    logger.info(f"Using device: {device}")
    
    # Load data
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Get label names
    label_names = df.iloc[:, 1:].columns.tolist()
    logger.info(f"Classes: {label_names}")
    
    # Initialize processor
    processor = TextProcessor(max_length=max_length)
    
    # Initialize model
    vocab_size = len(processor.tokenizer.vocab)
    num_classes = len(label_names)
    
    model = EfficientTextClassifier(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes
    )
    
    # Initialize optimizer (needed for trainer but won't be used for prediction)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    # Create trainer to load the model
    trainer = TextClassificationTrainer(
        model=model,
        train_loader=None,  # Not needed for loading
        val_loader=None,    # Not needed for loading
        criterion=None,     # Not needed for loading
        optimizer=optimizer,
        device=device,
        model_path=model_path
    )
    
    # Load model using the trainer's load_model method
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        logger.info(f"Model loaded from {model_path}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return
    
    # Create a small validation set for tuning thresholds
    from sklearn.model_selection import train_test_split
    
    # Prepare data
    X = df['comment']
    y = df.iloc[:, 1:]
    
    # Create validation set (10% of data)
    _, X_val, _, y_val = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y.sum(axis=1)
    )
    
    logger.info(f"Using {len(X_val)} samples for threshold tuning")
    
    # Create dataset and dataloader
    from torch.utils.data import DataLoader
    val_dataset = TextDataset(X_val, y_val, processor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Find optimal thresholds
    thresholds = find_optimal_thresholds(model, val_loader, label_names, device)
    
    # Save thresholds to file for future use
    # Convert any NumPy values to Python native types
    serializable_thresholds = {k: float(v) for k, v in thresholds.items()}
    with open('optimal_thresholds.json', 'w') as f:
        json.dump(serializable_thresholds, f, indent=2)
    logger.info(f"Thresholds saved to optimal_thresholds.json")
    
    # Create predictor with tuned thresholds
    predictor = ThresholdTunedPredictor(
        model=model,
        processor=processor,
        thresholds=thresholds,
        label_names=label_names,
        device=device
    )
    
    # Interactive prediction
    print("\nInteractive prediction mode with tuned thresholds (type 'exit' to quit):")
    while True:
        user_input = input("Enter a comment for prediction: ")
        if user_input.lower() == 'exit':
            break
            
        predictions, probabilities = predictor.predict(user_input)
        
        print("\nPredicted labels:")
        has_labels = False
        for i, label in enumerate(label_names):
            if predictions[i] > 0:
                has_labels = True
                print(f"- {label} (confidence: {probabilities[i]:.3f})")
        
        if not has_labels:
            print("- No labels predicted")
        
        print("\nRaw probabilities:")
        for i, label in enumerate(label_names):
            threshold = thresholds.get(label, 0.5)
            print(f"  {label}: {probabilities[i]:.3f} (threshold: {threshold:.3f})")
        print("")


if __name__ == "__main__":
    main()