from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import json
import torch
import pandas as pd
import re
from typing import Dict, List
import os

# Import components from the existing gs.py
from gs import TextProcessor, EfficientTextClassifier, Predictor

# Define request and response models
class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    original_text: str
    censored_text: str
    predictions: Dict[str, float]
    censored_words: List[str]

# Global variables
model = None
predictor = None
processor = None
optimal_thresholds = None
profanity_words = set()
label_columns = None
device = None

def load_profanity_words(file_path="extended_profanity_list.csv"):
    """Load profanity words from CSV file"""
    try:
        df = pd.read_csv(file_path)
        # Assuming the CSV has a column with profanity words
        word_column = df.columns[0]  # Get first column name
        return set(df[word_column].str.lower().tolist())
    except FileNotFoundError:
        print(f"Warning: {file_path} not found")
        return set()
    except Exception as e:
        print(f"Error loading profanity words: {e}")
        return set()

def censor_word(word, length):
    """Return a censored version of the word with asterisks matching the length"""
    return '*' * length

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load model and resources
    global model, predictor, processor, optimal_thresholds, profanity_words, label_columns, device
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load processor
    processor = TextProcessor(max_length=128)
    
    # Load optimal thresholds
    try:
        with open("optimal_thresholds.json", "r") as f:
            optimal_thresholds = json.load(f)
            label_columns = list(optimal_thresholds.keys())
            print(f"Loaded thresholds for {len(label_columns)} classes")
    except FileNotFoundError:
        print("Warning: optimal_threshold.JSON not found")
        optimal_thresholds = {}
    
    # Load model
    try:
        # Get vocab size
        vocab_size = len(processor.tokenizer.vocab)
        
        # Model parameters
        embedding_dim = 128
        hidden_dim = 128
        
        # Load model weights to determine num_classes if not available from thresholds
        checkpoint = torch.load("text_classifier_model.pt", map_location=device)
        
        # Determine number of classes
        if not label_columns:
            # Try to determine from model architecture
            if 'model_state_dict' in checkpoint:
                for key, value in checkpoint['model_state_dict'].items():
                    if key.endswith('classifier.3.weight'):
                        num_classes = value.shape[0]
                        break
                else:
                    # If we can't find the expected layer, use the last layer with weights
                    for key, value in checkpoint['model_state_dict'].items():
                        if 'weight' in key and len(value.shape) == 2:
                            last_layer_shape = value.shape
                    
                    # The output dimension is likely the first dimension of the weight matrix
                    num_classes = last_layer_shape[0]
                
                print(f"Inferred {num_classes} classes from model weights")
                label_columns = [f"class_{i}" for i in range(num_classes)]
                optimal_thresholds = {label: 0.5 for label in label_columns}
            else:
                raise ValueError("Could not determine number of classes from model")
        else:
            num_classes = len(label_columns)
        
        # Initialize model
        model = EfficientTextClassifier(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes
        ).to(device)
        
        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Initialize predictor
        predictor = Predictor(model, processor, device)
        print("Model loaded successfully")
    
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    
    # Load profanity words
    profanity_words = load_profanity_words("extended_profanity_list.csv")
    print(f"Loaded {len(profanity_words)} profanity words")
    
    yield
    
    # Cleanup (if needed)
    # Any cleanup code would go here

# Initialize FastAPI app with Swagger UI as default
app = FastAPI(
    title="Text Classification and Censoring API",
    description="API for predicting and censoring text based on model predictions and profanity list",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# Redirect root to Swagger UI
@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict text classification and censor if needed
    
    Args:
        request: PredictionRequest with text field
        
    Returns:
        PredictionResponse containing original text, censored text, and predictions
    """
    global model, predictor, optimal_thresholds, profanity_words, label_columns
    
    # Check if model is loaded
    if not model or not predictor:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Get input text
    text = request.text
    
    # Get model predictions
    try:
        with torch.no_grad():
            # Preprocess and tokenize
            preprocessed = predictor.processor.preprocess(text)
            encoding = predictor.processor.tokenize(preprocessed, return_tensors='pt')
            
            # Move to device
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            
            # Get raw predictions
            outputs = model(input_ids, attention_mask)
            probabilities = outputs.cpu().numpy()[0]
            
            # Round to 2 decimal places
            probabilities = [round(float(p), 2) for p in probabilities]
            
            # Create dictionary of predictions with rounded values
            predictions = {
                label: prob 
                for label, prob in zip(label_columns, probabilities)
            }
            
            # Apply thresholds to determine if censoring is needed
            needs_censoring = False
            for label, prob in predictions.items():
                threshold = optimal_thresholds.get(label, 0.5)
                if prob >= threshold:
                    needs_censoring = True
                    break
    
    except Exception as e:
        print(f"Error making prediction: {e}")
        raise HTTPException(status_code=500, detail="Prediction error")
    
    # Find words to censor
    censored_words = []
    
    # Extract all words from the text
    words = re.findall(r'\b\w+\b', text)
    
    # Check each word against profanity list
    for word in words:
        if word.lower() in profanity_words:
            censored_words.append(word)
    
    # Create censored text
    censored_text = text
    for word in censored_words:
        # Replace word with asterisks of the same length
        pattern = r'\b' + re.escape(word) + r'\b'
        censored_text = re.sub(pattern, '*' * len(word), censored_text, flags=re.IGNORECASE)
    
    return PredictionResponse(
        original_text=text,
        censored_text=censored_text,
        predictions=predictions,
        censored_words=censored_words
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)