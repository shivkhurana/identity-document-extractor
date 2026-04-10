import logging
import torch
from typing import Any, Dict
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel

logger = logging.getLogger(__name__)

class StructuredExtractor:
    def __init__(self, model_name: str = "naver-clova-ix/donut-base-finetuned-cord-v2"):
        """
        Initializes the Vision-Language Model for OCR and layout parsing.
        Automatically maps to GPU if available.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading VLM Extractor onto {self.device}...")

        try:
            self.processor = DonutProcessor.from_pretrained(model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval() # Set to evaluation mode
        except Exception as e:
            logger.error(f"Failed to load Hugging Face model: {e}")
            raise RuntimeError("VLM initialization failed.")

    def extract_fields(self, image: Image.Image) -> str:
        """
        Processes a cropped PIL Image and extracts structured text.
        """
        try:
            # Prepare the image tensor
            pixel_values = self.processor(image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device)

            # Generate text autoregressively
            task_prompt = "<s_cord-v2>"
            decoder_input_ids = self.processor.tokenizer(
                task_prompt, add_special_tokens=False, return_tensors="pt"
            ).input_ids.to(self.device)

            outputs = self.model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=self.model.decoder.config.max_position_embeddings,
                early_stopping=True,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                use_cache=True,
                num_beams=1,
                bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
            )

            # Decode the generated sequence
            sequence = self.processor.batch_decode(outputs.sequences)[0]
            sequence = sequence.replace(self.processor.tokenizer.eos_token, "").replace(self.processor.tokenizer.pad_token, "")
            
            # Clean up the specific task tokens
            cleaned_text = sequence.split("<s_cord-v2>")[-1].strip()
            return cleaned_text

        except Exception as e:
            logger.error(f"VLM extraction failed: {e}")
            return "EXTRACTION_ERROR"