# Sample_Data/src/feature_selection.py
import numpy as np
import weaviate
import os
from sklearn.preprocessing import StandardScaler

def fetch_embeddings_with_metadata(client):
    """
    Retrieves embeddings and basic metadata from Weaviate using v4 API.
    """
    try:
        collection = client.collections.get("CryptoDueDiligenceDocuments")
        response = collection.query.fetch_objects(
            include_vector=True,
            return_properties=["source", "title", "document_type"],
            limit=10000
        )
        
        embeddings = []
        metadata = []
        for obj in response.objects:
            if obj.vector and "default" in obj.vector:
                embeddings.append(obj.vector["default"])
                metadata.append({
                    "source": obj.properties.get("source", "Unknown"),
                    "title": obj.properties.get("title", "Untitled"),
                    "document_type": obj.properties.get("document_type", "Unknown"),
                })
        
        return np.array(embeddings), metadata
    
    except Exception as e:
        print(f"Error fetching embeddings: {e}")
        return np.array([]), []

def extract_meaningful_features(embeddings, metadata):
    """
    Extract features for crypto due diligence predictions from embeddings.
    """
    # Risk-related features
    volatility_index = np.std(np.abs(embeddings), axis=1)  # High variability = riskier signals
    anomaly_spike = np.max(np.abs(embeddings), axis=1) - np.mean(np.abs(embeddings), axis=1)  # Extreme peaks (hype or red flags)
    negative_depth = np.sum(embeddings < -0.5, axis=1) / embeddings.shape[1]  # Negative signal density (warnings)

    # Regulatory and compliance signals
    stability_score = 1 / (np.var(embeddings, axis=1) + 1e-6)  # Low variance = steady, compliant vibe
    authority_weight = np.mean(embeddings[:, 200:300], axis=1)  # Arbitrary slice for authority (BERT often clusters regs here)

    # Project legitimacy and hype
    hype_factor = np.sum(embeddings > 0.5, axis=1) / embeddings.shape[1]  # Positive density (hype or confidence)
    credibility_gap = np.mean(np.abs(embeddings - np.median(embeddings, axis=1)[:, None]), axis=1)  # Deviation from norm (trust issues)
    tokenomics_signal = np.mean(embeddings[:, 400:500], axis=1)  # Slice for financial/tech (tweakable)

    # Fraud potential
    fraud_energy = np.linalg.norm(embeddings * (embeddings < 0), axis=1)  # Negative-weighted magnitude (fraud energy)

    # Stack features
    features = np.column_stack([
        volatility_index,
        anomaly_spike,
        negative_depth,
        stability_score,
        authority_weight,
        hype_factor,
        credibility_gap,
        tokenomics_signal,
        fraud_energy
    ])
    
    # Normalize for model use
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    
    feature_names = [
        "volatility_index",    # Variability in embedding (risk signal)
        "anomaly_spike",       # Extreme deviation (hype or fraud spike)
        "negative_depth",      # Density of negative signals (red flags)
        "stability_score",     # Inverse variance (compliance stability)
        "authority_weight",    # Regulatory/authority signal from embedding slice
        "hype_factor",         # Density of positive signals (over-hype risk)
        "credibility_gap",     # Deviation from median (legitimacy proxy)
        "tokenomics_signal",   # Financial/tech signal from embedding slice
        "fraud_energy"         # Negative-weighted magnitude (fraud potential)
    ]
    
    return features, feature_names

def view_features(features, feature_names, metadata, num_samples=5):
    """
    Pretty print the extracted features with metadata.
    """
    print(f"\nTotal samples: {features.shape[0]}")
    print(f"Number of features per sample: {features.shape[1]}")
    print(f"Feature names: {feature_names}")
    print(f"\nFirst {min(num_samples, len(metadata))} samples:")
    for i, (sample, meta) in enumerate(zip(features[:num_samples], metadata[:num_samples])):
        print(f"Sample {i + 1}:")
        print(f"  Source: {meta['source']}")
        print(f"  Title: {meta['title']}")
        print(f"  Document Type: {meta['document_type']}")
        formatted_sample = {name: round(val, 4) for name, val in zip(feature_names, sample)}
        print(f"  Features: {formatted_sample}")

def feature_selection_pipeline(client):
    """
    Fetch embeddings, extract predictive features, and save them.
    """
    embeddings, metadata = fetch_embeddings_with_metadata(client)

    if embeddings.size == 0:
        print("No embeddings found.")
        return

    # Extract meaningful features
    features, feature_names = extract_meaningful_features(embeddings, metadata)

    # Save everything
    os.makedirs("Sample_Data", exist_ok=True)
    np.save("Sample_Data/raw_embeddings.npy", embeddings)
    np.save("Sample_Data/extracted_features.npy", features)
    np.save("Sample_Data/feature_names.npy", np.array(feature_names))
    np.save("Sample_Data/metadata.npy", metadata)
    
    print(f"✅ Raw embeddings saved: {embeddings.shape}")
    print(f"✅ Extracted features saved: {features.shape} with names: {feature_names}")
    
    # Show a sample
    view_features(features, feature_names, metadata)