from spacy.pipeline import EntityRuler
import json
import logging

logger = logging.getLogger(__name__)

def create_crypto_entity_ruler(nlp):
    """
    Create and configure a custom entity ruler for cryptocurrency terms.
    
    Args:
        nlp: spaCy language model
        
    Returns:
        EntityRuler: Configured entity ruler
    """
    # Create the entity ruler and add it to the pipeline
    ruler = EntityRuler(nlp, overwrite_ents=True)
    
    # Define patterns for different crypto entity types
    patterns = []
    
    # ---- CRYPTOCURRENCIES ----
    crypto_currencies = [
        # Major cryptocurrencies
        {"label": "CRYPTO", "pattern": "Bitcoin", "id": "BTC"},
        {"label": "CRYPTO", "pattern": "BTC", "id": "BTC"},
        {"label": "CRYPTO", "pattern": "Ethereum", "id": "ETH"},
        {"label": "CRYPTO", "pattern": "ETH", "id": "ETH"},
        {"label": "CRYPTO", "pattern": "Ripple", "id": "XRP"},
        {"label": "CRYPTO", "pattern": "XRP", "id": "XRP"},
        {"label": "CRYPTO", "pattern": "Litecoin", "id": "LTC"},
        {"label": "CRYPTO", "pattern": "Cardano", "id": "ADA"},
        {"label": "CRYPTO", "pattern": "ADA", "id": "ADA"},
        {"label": "CRYPTO", "pattern": "Solana", "id": "SOL"},
        {"label": "CRYPTO", "pattern": "SOL", "id": "SOL"},
        {"label": "CRYPTO", "pattern": "Polkadot", "id": "DOT"},
        {"label": "CRYPTO", "pattern": "DOT", "id": "DOT"},
        {"label": "CRYPTO", "pattern": "Binance Coin", "id": "BNB"},
        {"label": "CRYPTO", "pattern": "BNB", "id": "BNB"},
        
        # Stablecoins
        {"label": "STABLECOIN", "pattern": "Tether", "id": "USDT"},
        {"label": "STABLECOIN", "pattern": "USDT", "id": "USDT"},
        {"label": "STABLECOIN", "pattern": "USD Coin", "id": "USDC"},
        {"label": "STABLECOIN", "pattern": "USDC", "id": "USDC"},
        {"label": "STABLECOIN", "pattern": "Dai", "id": "DAI"},
        {"label": "STABLECOIN", "pattern": "DAI", "id": "DAI"},
        {"label": "STABLECOIN", "pattern": "TerraUSD", "id": "UST"},
        {"label": "STABLECOIN", "pattern": "UST", "id": "UST"},
    ]
    
    # ---- BLOCKCHAIN PROTOCOLS ----
    blockchain_protocols = [
        {"label": "PROTOCOL", "pattern": "Proof of Work", "id": "PoW"},
        {"label": "PROTOCOL", "pattern": "PoW", "id": "PoW"},
        {"label": "PROTOCOL", "pattern": "Proof of Stake", "id": "PoS"},
        {"label": "PROTOCOL", "pattern": "PoS", "id": "PoS"},
        {"label": "PROTOCOL", "pattern": "Delegated Proof of Stake", "id": "DPoS"},
        {"label": "PROTOCOL", "pattern": "DPoS", "id": "DPoS"},
        {"label": "PROTOCOL", "pattern": "Proof of Authority", "id": "PoA"},
        {"label": "PROTOCOL", "pattern": "Proof of History", "id": "PoH"},
        {"label": "PROTOCOL", "pattern": "Byzantine Fault Tolerance", "id": "BFT"},
        {"label": "PROTOCOL", "pattern": "Delegated Byzantine Fault Tolerance", "id": "dBFT"},
        {"label": "PROTOCOL", "pattern": "Practical Byzantine Fault Tolerance", "id": "PBFT"},
    ]
    
    # ---- TOKEN STANDARDS ----
    token_standards = [
        {"label": "TOKEN_STANDARD", "pattern": "ERC-20", "id": "ERC20"},
        {"label": "TOKEN_STANDARD", "pattern": "ERC20", "id": "ERC20"},
        {"label": "TOKEN_STANDARD", "pattern": "ERC-721", "id": "ERC721"},
        {"label": "TOKEN_STANDARD", "pattern": "ERC721", "id": "ERC721"},
        {"label": "TOKEN_STANDARD", "pattern": "ERC-1155", "id": "ERC1155"},
        {"label": "TOKEN_STANDARD", "pattern": "ERC1155", "id": "ERC1155"},
        {"label": "TOKEN_STANDARD", "pattern": "BEP-20", "id": "BEP20"},
        {"label": "TOKEN_STANDARD", "pattern": "BEP20", "id": "BEP20"},
        {"label": "TOKEN_STANDARD", "pattern": "BEP-721", "id": "BEP721"},
        {"label": "TOKEN_STANDARD", "pattern": "TRC-20", "id": "TRC20"},
        {"label": "TOKEN_STANDARD", "pattern": "TRC20", "id": "TRC20"},
    ]
    
    # ---- CRYPTO EXCHANGES ----
    crypto_exchanges = [
        {"label": "EXCHANGE", "pattern": "Binance", "id": "Binance"},
        {"label": "EXCHANGE", "pattern": "Coinbase", "id": "Coinbase"},
        {"label": "EXCHANGE", "pattern": "Kraken", "id": "Kraken"},
        {"label": "EXCHANGE", "pattern": "FTX", "id": "FTX"},
        {"label": "EXCHANGE", "pattern": "Gemini", "id": "Gemini"},
        {"label": "EXCHANGE", "pattern": "Bitstamp", "id": "Bitstamp"},
        {"label": "EXCHANGE", "pattern": "BitFinex", "id": "BitFinex"},
        {"label": "EXCHANGE", "pattern": "Huobi", "id": "Huobi"},
        {"label": "EXCHANGE", "pattern": "KuCoin", "id": "KuCoin"},
        {"label": "EXCHANGE", "pattern": "BitMEX", "id": "BitMEX"},
        {"label": "EXCHANGE", "pattern": "Uniswap", "id": "Uniswap"},
        {"label": "EXCHANGE", "pattern": "PancakeSwap", "id": "PancakeSwap"},
        {"label": "EXCHANGE", "pattern": "SushiSwap", "id": "SushiSwap"},
    ]
    
    # ---- DEFI TERMS ----
    defi_terms = [
        {"label": "DEFI", "pattern": "DeFi", "id": "DeFi"},
        {"label": "DEFI", "pattern": "Decentralized Finance", "id": "DeFi"},
        {"label": "DEFI", "pattern": "Yield Farming", "id": "YieldFarming"},
        {"label": "DEFI", "pattern": "Liquidity Mining", "id": "LiquidityMining"},
        {"label": "DEFI", "pattern": "AMM", "id": "AMM"},
        {"label": "DEFI", "pattern": "Automated Market Maker", "id": "AMM"},
        {"label": "DEFI", "pattern": "DEX", "id": "DEX"},
        {"label": "DEFI", "pattern": "Decentralized Exchange", "id": "DEX"},
        {"label": "DEFI", "pattern": "Liquidity Pool", "id": "LiquidityPool"},
        {"label": "DEFI", "pattern": "Smart Contract", "id": "SmartContract"},
        {"label": "DEFI", "pattern": "DAO", "id": "DAO"},
        {"label": "DEFI", "pattern": "Decentralized Autonomous Organization", "id": "DAO"},
    ]
    
    # ---- REGULATORY BODIES ----
    regulatory_bodies = [
        {"label": "REGULATOR", "pattern": "SEC", "id": "SEC"},
        {"label": "REGULATOR", "pattern": "Securities and Exchange Commission", "id": "SEC"},
        {"label": "REGULATOR", "pattern": "CFTC", "id": "CFTC"},
        {"label": "REGULATOR", "pattern": "Commodity Futures Trading Commission", "id": "CFTC"},
        {"label": "REGULATOR", "pattern": "FinCEN", "id": "FinCEN"},
        {"label": "REGULATOR", "pattern": "Financial Crimes Enforcement Network", "id": "FinCEN"},
        {"label": "REGULATOR", "pattern": "FATF", "id": "FATF"},
        {"label": "REGULATOR", "pattern": "Financial Action Task Force", "id": "FATF"},
        {"label": "REGULATOR", "pattern": "OFAC", "id": "OFAC"},
        {"label": "REGULATOR", "pattern": "Office of Foreign Assets Control", "id": "OFAC"},
    ]
    
    # Combine all patterns
    patterns = crypto_currencies + blockchain_protocols + token_standards + crypto_exchanges + defi_terms + regulatory_bodies
    
    # Add patterns to ruler
    ruler.add_patterns(patterns)
    logger.info(f"Added {len(patterns)} crypto-specific entity patterns to entity ruler")
    
    return ruler

def add_crypto_entities_to_pipeline(nlp):
    """
    Add crypto entity ruler to an existing spaCy pipeline.
    
    Args:
        nlp: The spaCy language model
        
    Returns:
        The modified spaCy language model
    """
    # Check if entity ruler already exists in pipeline
    if "entity_ruler" in nlp.pipe_names:
        nlp.remove_pipe("entity_ruler")
        
    # Create and add entity ruler
    ruler = create_crypto_entity_ruler(nlp)
    nlp.add_pipe(ruler, before="ner")
    
    logger.info("Added crypto entity ruler to spaCy pipeline")
    return nlp