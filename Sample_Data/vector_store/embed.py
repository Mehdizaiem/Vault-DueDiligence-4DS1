import os
from transformers import BertTokenizer, BertModel
import torch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get the embedding model name from environment variables
# Default to 'bert-base-uncased' if not specified
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bert-base-uncased")

# Initialize the tokenizer and model
tokenizer = BertTokenizer.from_pretrained(EMBEDDING_MODEL)
model = BertModel.from_pretrained(EMBEDDING_MODEL)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()  # Set model to evaluation mode

def generate_embedding(text):
    """
    Generate an embedding vector for the given text using BERT.
    
    Args:
        text (str): The text to embed
        
    Returns:
        list: The embedding vector as a list of floats
    """
    # Tokenize the text and prepare for model
    inputs = tokenizer(text, 
                      padding=True, 
                      truncation=True, 
                      max_length=512, 
                      return_tensors="pt")
    
    # Move inputs to the same device as model
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate embeddings
    with torch.no_grad():  # Disable gradient calculations
        outputs = model(**inputs)
        
        # Use the [CLS] token embedding as the document embedding
        embeddings = outputs.last_hidden_state[:, 0, :].squeeze()
        
        # Convert to list and move back to CPU if necessary
        if device.type == "cuda":
            embeddings = embeddings.cpu()
            
        return embeddings.numpy().tolist()

if __name__ == "__main__":
    # Test the embedding generation
    test_text = "This is a test document about cryptocurrency regulations."
    embedding = generate_embedding(test_text)
    print(f"Generated embedding with {len(embedding)} dimensions")
    print(f"First few values: {embedding[:5]}")