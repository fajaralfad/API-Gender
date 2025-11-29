# app/config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # API Configuration
    API_TITLE = "Age Gender Classification API"
    API_DESCRIPTION = "API for age and gender classification from images using Vision Transformer"
    API_VERSION = "1.0.0"
    
    # Security
    API_KEY = os.getenv("API_KEY", "your-default-api-key-here")
    API_KEY_NAME = "X-API-Key"
    
    # Model Configuration - HuggingFace ViT Model
    MODEL_NAME = os.getenv("MODEL_NAME", "abhilash88/age-gender-prediction")
    
    # Model input size for ViT Base (standard is 224x224)
    IMAGE_SIZE = (224, 224)
    
    # Age range constraints
    MIN_AGE = 0
    MAX_AGE = 100
    
    # Gender classes
    GENDER_CLASSES = ["Male", "Female"]
    
    # File upload settings
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp"]
    
    # Model cache settings
    CACHE_DIR = os.getenv("CACHE_DIR", "./model_cache")
    
    # Trust remote code setting (required for custom models)
    TRUST_REMOTE_CODE = os.getenv("TRUST_REMOTE_CODE", "True").lower() == "true"

settings = Settings()