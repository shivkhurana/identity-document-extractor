import cv2
import logging
import numpy as np
from typing import Dict, Optional
from ultralytics import YOLO
from PIL import Image

logger = logging.getLogger(__name__)

class DocumentDetector:
    def __init__(self, weights_path: str = "yolov8n.pt"):
        """
        Initializes the YOLO model for document zone detection.
        In production, weights_path would point to a custom-trained model.
        """
        try:
            self.model = YOLO(weights_path)
            logger.info(f"Successfully loaded YOLO model from {weights_path}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise RuntimeError("Model initialization failed.")

    def crop_zones(self, image_path: str, confidence_threshold: float = 0.6) -> Optional[Dict[str, Image.Image]]:
        """
        Runs inference on the document and crops out detected zones (e.g., Name, ID).
        Returns a dictionary mapping the class name to the cropped PIL Image.
        """
        try:
            # Run YOLO inference
            results = self.model.predict(source=image_path, conf=confidence_threshold, save=False)
            
            # Read original image using OpenCV for cropping
            original_img = cv2.imread(image_path)
            if original_img is None:
                raise ValueError(f"Could not read image at {image_path}")

            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            cropped_zones = {}

            # Process bounding boxes
            for box in results[0].boxes:
                # Extract coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                class_id = int(box.cls[0].item())
                class_name = self.model.names[class_id]

                # Crop the image array
                cropped_array = original_img[y1:y2, x1:x2]
                
                # Convert back to PIL Image for the HuggingFace pipeline
                cropped_zones[class_name] = Image.fromarray(cropped_array)

            logger.info(f"Successfully cropped {len(cropped_zones)} zones.")
            return cropped_zones

        except Exception as e:
            logger.error(f"Error during zone cropping: {e}")
            return None