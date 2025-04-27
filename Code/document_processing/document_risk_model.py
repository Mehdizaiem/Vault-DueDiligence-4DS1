# File: document_risk_model_final.py

from typing import Tuple, List
import re

def extract_features(text: str) -> dict:
    """
    Extracts both risk and positive mitigation features from text.
    """
    text_lower = text.lower()

    risk_keywords = [
        "fraud", "rug pull", "money laundering", "ponzi", "regulatory breach",
        "compliance issue", "illegal", "hacking", "vulnerability",
        "pump and dump", "manipulation", "spoofing", "fake volume",
        "lack of audit", "undisclosed team", "centralized control"
    ]

    positive_keywords = [
        "regulated", "licensed", "audited", "compliance certified", "fully registered",
        "security audit", "iso certified", "approved by authority"
    ]

    feature_dict = {k: 0 for k in risk_keywords + positive_keywords}

    for keyword in feature_dict.keys():
        if re.search(rf"\b{re.escape(keyword)}\b", text_lower):
            feature_dict[keyword] = 1

    return feature_dict

def calculate_risk_score(features: dict) -> float:
    if not features:
        return 50.0  # Default neutral risk

    # Weights for critical features
    risk_weights = {
        "fraud": 30, "rug pull": 30, "money laundering": 30, "ponzi": 30,
        "regulatory breach": 25, "compliance issue": 25, "illegal": 25, "hacking": 25,
        "vulnerability": 20, "pump and dump": 20, "manipulation": 20,
        "spoofing": 15, "fake volume": 15, "lack of audit": 20,
        "undisclosed team": 20, "centralized control": 20
    }

    positive_weights = {
        "regulated": -20, "licensed": -20, "audited": -25,
        "compliance certified": -30, "fully registered": -30,
        "security audit": -25, "iso certified": -20, "approved by authority": -25
    }

    # Sum weighted risk and positive scores
    risk_score = 0.0

    for feature, weight in risk_weights.items():
        risk_score += features.get(feature, 0) * weight

    for feature, weight in positive_weights.items():
        risk_score += features.get(feature, 0) * weight

    # Normalize and clip
    final_score = min(100.0, max(0.0, 10 + risk_score))

    return final_score

def classify_risk(score: float) -> str:
    if score < 20:
        return "Very Low"
    elif score < 40:
        return "Low"
    elif score < 60:
        return "Moderate"
    elif score < 80:
        return "High"
    else:
        return "Very High"

def assess_document_risk(text: str) -> Tuple[float, str, List[str]]:
    features = extract_features(text)
    score = calculate_risk_score(features)
    category = classify_risk(score)
    triggered_keywords = [k for k, v in features.items() if v == 1]
    return score, category, triggered_keywords

# Example usage
if __name__ == "__main__":
    example_text = (
        "The project is fully regulated, audited, and compliance certified."
        " There are no fraud or money laundering allegations."
    )
    score, category, factors = assess_document_risk(example_text)
    print(f"Score: {score:.2f} | Category: {category} | Factors: {factors}")