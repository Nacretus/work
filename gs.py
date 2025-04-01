import warnings
warnings.filterwarnings("ignore")
import torch.nn.functional as F
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib import style
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score
import logging
import os
import random
from tqdm import tqdm  # For progress bars
from stopwordsiso import stopwords as stopwords_iso

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set manual seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Get device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

class TextProcessor:
    """Text preprocessing and tokenization class"""
    
    def __init__(self, max_length=128):
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.stop_words = set(stopwords_iso("en")).union(stopwords_iso("tl"))
        
    def preprocess(self, text):
        """Clean and preprocess text data"""
        # Convert to lowercase
        text = text.lower()
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Remove stopwords
        tokens = text.split()
        filtered_tokens = [word for word in tokens if word not in self.stop_words]
        return ' '.join(filtered_tokens)
    
    def tokenize(self, text, return_tensors=None):
        """Tokenize text using BERT tokenizer"""
        return self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors=return_tensors
        )
    
    def transform_text_batch(self, texts):
        """Preprocess and tokenize a batch of texts"""
        preprocessed_texts = [self.preprocess(text) for text in texts]
        encoding = self.tokenize(
            preprocessed_texts, 
            return_tensors='pt'
        )
        return encoding


class TextDataset(Dataset):
    """Dataset for text classification"""
    
    def __init__(self, texts, labels, processor):
        self.texts = texts
        self.labels = torch.tensor(labels.values, dtype=torch.float32)
        self.processor = processor
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts.iloc[idx]
        preprocessed = self.processor.preprocess(text)
        encoding = self.processor.tokenize(preprocessed, return_tensors=None)
        
        # Convert to tensors
        input_ids = torch.tensor(encoding['input_ids'], dtype=torch.long)
        attention_mask = torch.tensor(encoding['attention_mask'], dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': self.labels[idx]
        }


class EfficientTextClassifier(nn.Module):
    """
    Optimized model architecture with BiLSTM, CNN and Attention
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, dropout_rate=0.3):
        super(EfficientTextClassifier, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Bidirectional LSTM - using a single layer for efficiency
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            batch_first=True,
            bidirectional=True
        )
        
        # Single efficient CNN layer instead of multiple parallel ones
        self.conv = nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1)
        
        # Simple but effective attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes),
            nn.Sigmoid()
        )
    
    def forward(self, input_ids, attention_mask=None):
        # Get embeddings
        embedded = self.embedding(input_ids)
        
        # Apply LSTM - using packed sequence for efficiency when needed
        if attention_mask is not None:
            # Create a packed sequence
            lstm_out, _ = self.lstm(embedded)
            lstm_out = lstm_out * attention_mask.unsqueeze(-1)
        else:
            lstm_out, _ = self.lstm(embedded)
        
        # Apply CNN (permute to get channels dimension in middle)
        lstm_out = lstm_out.permute(0, 2, 1)
        cnn_out = self.conv(lstm_out)
        cnn_out = cnn_out.permute(0, 2, 1)  # back to [batch, seq, features]
        
        # Apply attention
        attn_weights = self.attention(cnn_out)
        context = torch.sum(attn_weights * cnn_out, dim=1)
        
        # Classification
        output = self.classifier(context)
        return output


class TextClassificationTrainer:
    """Class to handle training and evaluation"""
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler=None,
        device=torch.device("cpu"),
        model_path="text_classifier_model.pt"
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.model_path = model_path
        self.best_val_loss = float('inf')
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        train_loss = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(input_ids, attention_mask)
            
            # Calculate loss
            loss = self.criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # Gradient clipping
            self.optimizer.step()
            
            # Update tracking
            train_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        return train_loss / len(self.train_loader)
    
    def evaluate(self):
        """Evaluate the model"""
        self.model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask)
                
                # Calculate loss
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()
                
                # Store predictions and labels
                preds = (outputs > 0.5).float()
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        # Concatenate batches
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels.flatten(), all_preds.flatten())
        f1 = f1_score(all_labels, all_preds, average='micro')
        
        return val_loss / len(self.val_loader), accuracy, f1
    
    def train(self, num_epochs, patience=3):
        """Full training loop with early stopping"""
        logger.info(f"Starting training for {num_epochs} epochs")
        
        best_val_loss = float('inf')
        no_improvement = 0
        
        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch()
            
            # Evaluate
            val_loss, accuracy, f1 = self.evaluate()
            
            # Step the scheduler with validation loss if it exists
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Log progress
            logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                        f"Train Loss: {train_loss:.4f}, "
                        f"Val Loss: {val_loss:.4f}, "
                        f"Accuracy: {accuracy:.4f}, "
                        f"F1: {f1:.4f}")
            
            # Check if this is the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improvement = 0
                
                # Save the model
                self.save_model()
                logger.info(f"New best model saved with validation loss: {val_loss:.4f}")
            else:
                no_improvement += 1
                
            # Early stopping
            if no_improvement >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
                
        logger.info("Training completed")
        
    def save_model(self):
        """Save the model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, self.model_path)
    
    def load_model(self):
        """Load the model"""
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info(f"Model loaded from {self.model_path}")
        else:
            logger.error(f"No model found at {self.model_path}")


class Predictor:
    """Class for making predictions with a trained model"""
    
    def __init__(self, model, processor, device, threshold=0.5):
        self.model = model
        self.processor = processor
        self.device = device
        self.threshold = threshold
        self.model.to(device)
        self.model.eval()
        
    def predict(self, text):
        """Predict labels for a single text"""
        # Preprocess and tokenize
        preprocessed = self.processor.preprocess(text)
        encoding = self.processor.tokenize(preprocessed, return_tensors='pt')
        
        # Move to device
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            predictions = (outputs > self.threshold).float().cpu().numpy()
            
        return predictions
    
    def predict_batch(self, texts):
        """Predict labels for a batch of texts"""
        # Preprocess and tokenize
        encodings = self.processor.transform_text_batch(texts)
        
        # Move to device
        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            predictions = (outputs > self.threshold).float().cpu().numpy()
            
        return predictions


def visualize_data(df):
    """Create and save visualizations of the data"""
    # Set style
    style.use('ggplot')
    
    # Label distribution
    plt.figure(figsize=(10, 6))
    x = df.iloc[:, 1:].sum()
    plt.bar(x.index, x.values, alpha=0.8, color='tab:blue')
    plt.title("Label Distribution", fontsize=14)
    plt.ylabel("Count", fontsize=12)
    plt.xlabel("Labels", fontsize=12)
    plt.savefig("label_distribution.png", dpi=300, bbox_inches='tight')
    
    # Label per comment
    plt.figure(figsize=(10, 6))
    rowsum = df.iloc[:, 1:].sum(axis=1)
    plt.hist(rowsum.values, bins=range(0, max(rowsum) + 2), alpha=0.8, color='tab:orange')
    plt.title("Labels per Comment", fontsize=14)
    plt.ylabel("Number of Comments", fontsize=12)
    plt.xlabel("Number of Labels", fontsize=12)
    plt.xticks(range(0, max(rowsum) + 1))
    plt.savefig("labels_per_comment.png", dpi=300, bbox_inches='tight')
    
    # Log some statistics
    no_label = (rowsum == 0).sum()
    logger.info(f"Number of rows with no label: {no_label}")
    logger.info(f"Number of comments: {len(df)}")
    logger.info(f"Total labels: {x.sum()}")

def calculate_class_weights(df, label_columns):
    """
    Calculate class weights inversely proportional to class frequencies.
    More frequent classes get lower weights, less frequent classes get higher weights.
    
    Args:
        df: DataFrame containing the dataset
        label_columns: List or Index of column names for the labels
        
    Returns:
        Dictionary mapping class names to their weights
    """
    logger.info("Calculating class weights...")
    
    # Count number of samples in each class
    class_counts = df[label_columns].sum().to_dict()
    total_samples = len(df)
    
    # Calculate weights: less frequent classes get higher weights
    class_weights = {}
    for label, count in class_counts.items():
        # Add a small epsilon to avoid division by zero
        weight = total_samples / (count + 1e-5)
        class_weights[label] = weight
    
    # Normalize weights
    weight_sum = sum(class_weights.values())
    for label in class_weights:
        class_weights[label] = class_weights[label] / weight_sum * len(class_weights)
        
    # Log the weights
    for label, weight in class_weights.items():
        logger.info(f"Weight for {label}: {weight:.4f}")
        
    return class_weights

# Add this class before the main function
class WeightedBCELoss(nn.Module):
    """
    Binary Cross-Entropy Loss with class weights.
    Each class can have a different weight to address class imbalance.
    """
    def __init__(self, class_weights):
        super(WeightedBCELoss, self).__init__()
        self.class_weights = class_weights
        self.weight_keys = list(class_weights.keys())
        
    def forward(self, outputs, targets):
        # Create weight tensor from the class_weights dictionary
        weights = torch.ones(len(self.weight_keys), device=outputs.device)
        for i, label in enumerate(self.weight_keys):
            weights[i] = self.class_weights[label]
            
        # Apply weighted BCE loss
        # First calculate BCE loss for each output/target pair
        loss = F.binary_cross_entropy(outputs, targets, reduction='none')
        
        # Then multiply by the weights for each class
        weighted_loss = loss * weights.unsqueeze(0)
        
        # Return mean over all elements
        return weighted_loss.mean()

def main():
    # Configuration
    DATA_PATH = "FF.csv"
    MODEL_PATH = "text_classifier_model.pt"
    MAX_LENGTH = 128
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LEARNING_RATE = 2e-5  # Lower learning rate for more stability
    EMBEDDING_DIM = 128
    HIDDEN_DIM = 128
    VALIDATION_SPLIT = 0.1
    
    # Load data
    logger.info(f"Loading data from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    
    # Visualize data (optional)
    visualize_data(df)
    
    # Initialize text processor
    processor = TextProcessor(max_length=MAX_LENGTH)
    
    # Prepare data
    X = df['comment']
    y = df.iloc[:, 1:]  # Assuming labels start from column 1
    
    # Create train/val/test split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=SEED, stratify=df.iloc[:, 1:].sum(axis=1)
    )
    
    # Further split temp into validation and test
    val_ratio = VALIDATION_SPLIT / 0.3  # Adjust to get correct final ratio
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=SEED, 
        stratify=y_temp.sum(axis=1)
    )
    
    logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Create datasets
    train_dataset = TextDataset(X_train, y_train, processor)
    val_dataset = TextDataset(X_val, y_val, processor)
    test_dataset = TextDataset(X_test, y_test, processor)
    
    # Create data loaders with num_workers for parallel data loading
    num_workers = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Initialize model
    vocab_size = len(processor.tokenizer.vocab)
    num_classes = y.shape[1]
    
    model = EfficientTextClassifier(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=num_classes
    ).to(device)
    
    # Initialize optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    criterion = nn.BCELoss()
    
    # Initialize trainer
    trainer = TextClassificationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        model_path=MODEL_PATH
    )
    
    # Train the model
    trainer.train(NUM_EPOCHS, patience=3)
    
    # Load the best model for evaluation
    trainer.load_model()
    
    # Evaluate on test set
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            preds = (outputs > 0.5).float()
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    # Calculate and log metrics
    accuracy = accuracy_score(all_labels.flatten(), all_preds.flatten())
    f1_micro = f1_score(all_labels, all_preds, average='micro')
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    
    logger.info(f"Test Results - Accuracy: {accuracy:.4f}, F1 (micro): {f1_micro:.4f}, F1 (macro): {f1_macro:.4f}")
    logger.info("\nClassification Report:")
    logger.info(classification_report(all_labels, all_preds, target_names=y.columns))
    
    # Initialize predictor for inference
    predictor = Predictor(model, processor, device)
    
    # Demo prediction
    logger.info("\nDemo Prediction:")
    sample_text = "This is a sample text for prediction"
    prediction = predictor.predict(sample_text)
    logger.info(f"Text: {sample_text}")
    logger.info(f"Prediction: {prediction}")
    
    # Interactive prediction loop
    print("\nInteractive prediction mode (type 'exit' to quit):")
    while True:
        user_input = input("Enter a comment for prediction: ")
        if user_input.lower() == 'exit':
            break
            
        prediction = predictor.predict(user_input)
        class_predictions = {y.columns[i]: pred for i, pred in enumerate(prediction[0])}
        print("Predicted labels:")
        for label, value in class_predictions.items():
            if value > 0:
                print(f"- {label}")
        
        if all(value == 0 for value in class_predictions.values()):
            print("- No labels predicted")


if __name__ == "__main__":
    main()