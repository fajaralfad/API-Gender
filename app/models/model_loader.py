import onnxruntime as ort
import numpy as np
from PIL import Image
import io
import logging
from app.config import settings

logger = logging.getLogger(__name__)

class GenderClassifier:
    def __init__(self):
        self.session = None
        self.input_name = None
        self.output_name = None
        self.is_loaded = False
       
        logger.info("GenderClassifier initialized (model will be loaded on first request)")
    
    def load_model(self):
        """Load ONNX model (lazy loading)"""
        if self.is_loaded:
            return
            
        try:
            logger.info(f"Loading ONNX model from: {settings.MODEL_PATH}")
            
            # Load ONNX model
            self.session = ort.InferenceSession(
                settings.MODEL_PATH,
                providers=['CPUExecutionProvider']
            )
            
            # Get input and output details
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            
            # Get input details untuk debugging
            input_details = self.session.get_inputs()[0]
            logger.info(f"Input name: {self.input_name}")
            logger.info(f"Input shape: {input_details.shape}")
            logger.info(f"Input type: {input_details.type}")
            logger.info(f"Output name: {self.output_name}")
            
            self.is_loaded = True
            logger.info("ONNX model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            self.is_loaded = False
            raise e
    
    def preprocess_image(self, image_bytes):
        """Preprocess image untuk ONNX model"""
        try:
            # Open image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize image sesuai input model dari environment variable
            target_size = (settings.IMAGE_SIZE, settings.IMAGE_SIZE)
            image = image.resize(target_size)
            
            # Convert to array dengan tipe data float32
            image_array = np.array(image, dtype=np.float32)
            
            # Normalize menggunakan mean dan std BEiT
            mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
            std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
            image_array = (image_array / 255.0 - mean) / std
            
            # Change from HWC to CHW format
            image_array = np.transpose(image_array, (2, 0, 1))
            
            # Add batch dimension
            image_batch = np.expand_dims(image_array, axis=0)
            
            # Pastikan tipe data adalah float32
            image_batch = image_batch.astype(np.float32)
            
            logger.info(f"Processed image shape: {image_batch.shape}")
            logger.info(f"Processed image dtype: {image_batch.dtype}")
            
            return image_batch
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise e
    
    def predict(self, image_bytes):
        """Make prediction menggunakan ONNX"""
        # Load model on first predict call (lazy loading)
        if not self.is_loaded:
            logger.info("Model not loaded yet, loading now...")
            try:
                self.load_model()
            except Exception as e:
                return {
                    "error": f"Failed to load model: {str(e)}",
                    "success": False
                }
        
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image_bytes)
            
            # Debug: print input details sebelum inference
            logger.info(f"Input tensor shape: {processed_image.shape}")
            logger.info(f"Input tensor dtype: {processed_image.dtype}")
            
            # Run inference
            outputs = self.session.run(
                [self.output_name], 
                {self.input_name: processed_image}
            )
            
            predictions = outputs[0][0]  # Get first batch predictions
            
            # Apply softmax to get probabilities
            exp_preds = np.exp(predictions - np.max(predictions))
            probabilities = exp_preds / np.sum(exp_preds)
            
            # Get top prediction
            predicted_class_idx = np.argmax(probabilities)
            confidence = float(probabilities[predicted_class_idx])
            
            # Get class name dari environment variable
            predicted_class = settings.CLASS_NAMES[predicted_class_idx]
            
            # Get all predictions with confidence
            all_predictions = [
                {
                    "class": settings.CLASS_NAMES[i],
                    "confidence": float(probabilities[i])
                }
                for i in range(len(settings.CLASS_NAMES))
            ]
            
            # Sort by confidence descending
            all_predictions.sort(key=lambda x: x["confidence"], reverse=True)
            
            logger.info(f"Prediction result: {predicted_class} ({confidence:.4f})")
            
            return {
                "success": True,
                "predicted_class": predicted_class,
                "confidence": confidence,
                "all_predictions": all_predictions
            }
            
        except Exception as e:
            logger.error(f"Error during ONNX prediction: {e}")
            return {
                "error": str(e),
                "success": False
            }

# Global ONNX model instance
classifier = GenderClassifier()