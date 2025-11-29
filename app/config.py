import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # API Configuration
    API_TITLE = "Gender Classification API"
    API_DESCRIPTION = "API untuk klasifikasi gender menggunakan model BEiT"
    API_VERSION = "1.0.0"
    
    # Security
    API_KEY = os.getenv("API_KEY", "silpi")
    API_KEY_NAME = "X-API-Key"
    
    # Model Configuration
    MODEL_PATH = os.getenv("MODEL_PATH", "models/gender.onnx")
    IMAGE_SIZE = (224, 224)
    
    # Class labels (sesuai dengan model gender)
    CLASS_NAMES = [
        "female",
        "male"
    ]

settings = Settings()