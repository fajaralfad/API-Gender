import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # API Configuration
    API_TITLE = os.getenv("API_TITLE", "Gender Classification API")
    API_DESCRIPTION = os.getenv("API_DESCRIPTION", "API untuk klasifikasi gender menggunakan model BEiT")
    API_VERSION = os.getenv("API_VERSION", "1.0.0")
    
    # Security
    API_KEY = os.getenv("API_KEY", "your-default-api-key-here")
    API_KEY_NAME = os.getenv("API_KEY_NAME", "X-API-Key")
    
    # Model Configuration
    MODEL_PATH = os.getenv("MODEL_PATH", "models/gender.onnx")
    IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "224"))
    
    # Server Configuration
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))
    LOG_LEVEL = os.getenv("LOG_LEVEL", "info")
    
    # CORS Configuration
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
    
    # Class labels
    CLASS_NAMES = os.getenv("CLASS_NAMES", "female,male").split(",")

settings = Settings()