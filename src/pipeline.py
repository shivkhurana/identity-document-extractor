import logging
from src.detection.yolo_cropper import DocumentDetector
from src.extraction.vlm_extractor import StructuredExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentExtractionPipeline:
    def __init__(self):
        logger.info("Initializing YOLO detection model...")
        self.detector = DocumentDetector(weights_path="models/yolo_doc_weights.pt")
        
        logger.info("Initializing Hugging Face VLM extractor...")
        self.extractor = StructuredExtractor(model_name="microsoft/donut-base")

    def process_document(self, image_path: str) -> dict:
        """
        End-to-end pipeline: Detects the document, crops critical zones, 
        and extracts structured text (Name, Date, ID Number).
        """
        logger.info(f"Processing document: {image_path}")
        
        # Step 1: YOLO detects and crops the relevant bounding boxes
        cropped_zones = self.detector.crop_zones(image_path)
        
        if not cropped_zones:
            logger.error("No valid document zones detected.")
            return {"error": "Detection failed"}

        # Step 2: VLM extracts structured data from the cropped zones
        extracted_data = {}
        for zone_name, cropped_image in cropped_zones.items():
            logger.info(f"Extracting text from zone: {zone_name}")
            text = self.extractor.extract_fields(cropped_image)
            extracted_data[zone_name] = text

        return extracted_data

if __name__ == "__main__":
    # Test the pipeline execution
    pipeline = DocumentExtractionPipeline()
    result = pipeline.process_document("data/sample_passport_scan.jpg")
    print("Final Extracted Data:", result)