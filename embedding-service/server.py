from flask import Flask, request, jsonify
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MPNET_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
FINBERT_MODEL_NAME = "yiyanghkust/finbert-tone"
MPNET_DIMENSION = 768
FINBERT_DIMENSION = 768

app = Flask(__name__)

# Load models at startup
logger.info(f"Loading all-MPNet model: {MPNET_MODEL_NAME}")
mpnet_model = SentenceTransformer(MPNET_MODEL_NAME)

logger.info(f"Loading FinBERT model: {FINBERT_MODEL_NAME}")
finbert_tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL_NAME)
finbert_model = AutoModel.from_pretrained(FINBERT_MODEL_NAME)

logger.info("All embedding models loaded successfully")

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "models": ["mpnet", "finbert"]})

@app.route('/generate_mpnet_embedding', methods=['POST'])
def generate_mpnet():
    data = request.json
    text = data.get('text', '')
    
    if not text or not isinstance(text, str):
        logger.warning("Invalid text provided for embedding")
        return jsonify({"error": "Invalid text provided"}), 400
    
    try:
        # Truncate text to avoid potential memory issues
        text = text[:100000]  # Limit to 100K characters
        
        # Generate embeddings
        with torch.no_grad():  # Disable gradient calculation for inference
            embeddings = mpnet_model.encode(text, show_progress_bar=False)
        
        # Convert to list if it's a numpy array
        result = embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings
        return jsonify({"embedding": result})
    except Exception as e:
        logger.error(f"Error generating MPNet embedding: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/generate_finbert_embedding', methods=['POST'])
def generate_finbert():
    data = request.json
    text = data.get('text', '')
    
    if not text or not isinstance(text, str):
        logger.warning("Invalid text provided for FinBERT embedding")
        return jsonify({"error": "Invalid text provided"}), 400
    
    try:
        # Truncate text if needed
        text = text[:500]  # FinBERT has a more limited context window
        
        # Tokenize and get input tensors
        inputs = finbert_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        # Generate embeddings
        with torch.no_grad():
            outputs = finbert_model(**inputs)
            
        # Use the CLS token embedding as the document representation
        cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()
        
        # Normalize the embedding
        embedding = cls_embedding[0]
        embedding_norm = np.linalg.norm(embedding)
        if embedding_norm > 0:
            embedding = embedding / embedding_norm
            
        return jsonify({"embedding": embedding.tolist()})
    except Exception as e:
        logger.error(f"Error generating FinBERT embedding: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)