import os
import cv2
import logging
import albumentations as A
from glob import glob

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentAugmenter:
    def __init__(self):
        """
        Defines a robust augmentation pipeline specifically for simulating 
        real-world bad passport/ID uploads (glare, blur, bad angles).
        """
        self.transform = A.Compose([
            A.Perspective(scale=(0.05, 0.1), p=0.5),      # Simulates holding the phone at a weird angle
            A.RandomBrightnessContrast(p=0.5),            # Simulates bad lighting
            A.MotionBlur(blur_limit=5, p=0.3),            # Simulates shaky hands
            A.ISONoise(color_shift=(0.01, 0.05), p=0.2),  # Simulates cheap phone cameras
            A.CoarseDropout(max_holes=2, max_height=20, max_width=20, fill_value=255, p=0.2) # Simulates glare/flash spots
        ])

    def augment_dataset(self, input_dir: str, output_dir: str, num_variations: int = 3):
        """
        Automates the creation of a robust training dataset so the team 
        isn't doing the same tedious data processing twice.
        """
        os.makedirs(output_dir, exist_ok=True)
        image_paths = glob(os.path.join(input_dir, "*.jpg"))
        
        logger.info(f"Found {len(image_paths)} original images. Generating {num_variations} variations each...")

        for img_path in image_paths:
            filename = os.path.basename(img_path)
            image = cv2.imread(img_path)
            
            if image is None:
                logger.warning(f"Could not read {img_path}. Skipping.")
                continue
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Save original
            cv2.imwrite(os.path.join(output_dir, f"orig_{filename}"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            # Generate augmented variations
            for i in range(num_variations):
                augmented = self.transform(image=image)
                aug_img = augmented['image']
                
                out_path = os.path.join(output_dir, f"aug_{i}_{filename}")
                cv2.imwrite(out_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
                
        logger.info(f"Augmentation complete. Output saved to {output_dir}")

if __name__ == "__main__":
    augmenter = DocumentAugmenter()
    # Assuming you have a folder called 'data/raw' and 'data/processed'
    augmenter.augment_dataset("data/raw", "data/processed", num_variations=4)