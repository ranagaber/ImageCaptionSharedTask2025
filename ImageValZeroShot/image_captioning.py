#!/usr/bin/env python3
"""
Arabic Image Captioning using Qwen2.5-VL-7B-Instruct
This script processes images and generates Arabic captions for historical content.
"""

import os
import csv
import argparse
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


class ArabicImageCaptioner:
    """Class for generating Arabic captions for images using Qwen2.5-VL model."""

    def __init__(self, model_name="Qwen/Qwen2.5-VL-7B-Instruct", checkpoint_path=None):
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path

    def load_model(self):
        """Load the model and processor."""
        print(f"Loading model: {self.model_name}")
        print(f"Using device: {self.device}")

        loading_path = self.checkpoint_path if self.checkpoint_path else self.model_name

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            loading_path,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="eager"
        ).eval().half()  # optimize memory

        self.processor = AutoProcessor.from_pretrained(self.model_name, use_fast=True)
        print("Model and processor loaded successfully!")

    def generate_caption(self, image_path, max_new_tokens=128):
        """Generate Arabic caption for a single image."""
        try:
            image = Image.open(image_path).convert("RGB")
            image = image.resize((512, 512))  # prevent large memory spikes

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {
                            "type": "text",
                            "text": (
                                "You are an expert in visual scene understanding and multilingual caption generation."
                                "Analyze the content of this image, which is potentially related to the palestnian Nakba"
                                "and Israeli occupation of Palestine, and provide a concise and meaningful caption in Arabic - about 15 to 50 words."
                                "The caption should reflect the scene's content, emotional context, and should be natural and culturally appropriate."
                                " Do not include any English or metadata — The caption must be in Arabic."
                            ),
                        },
                    ],
                }
            ]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, _ = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device)

            torch.cuda.empty_cache()
            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs, max_new_tokens=max_new_tokens
                )

            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )

            return output_text[0].strip()

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return ""

    def process_folder(self, image_folder, output_csv, supported_formats=('.png', '.jpg', '.jpeg')):
        """Process all images in a folder and save captions to CSV."""
        if not os.path.exists(image_folder):
            raise FileNotFoundError(f"Image folder not found: {image_folder}")

        image_files = [
            f for f in os.listdir(image_folder)
            if f.lower().endswith(supported_formats)
        ]

        if not image_files:
            print(f"No supported image files found in {image_folder}")
            return

        print(f"Found {len(image_files)} images to process")

        os.makedirs(os.path.dirname(output_csv), exist_ok=True)

        with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["image_file", "arabic_caption"])

            for image_file in tqdm(image_files, desc="Processing images"):
                image_path = os.path.join(image_folder, image_file)
                caption = self.generate_caption(image_path)

                if caption:
                    print(f"{image_file}: {caption}")
                    writer.writerow([image_file, caption])
                else:
                    print(f"Failed to generate caption for {image_file}")

                # Free memory after each image
                del caption, image_path
                torch.cuda.empty_cache()

        print(f"Processing complete! Results saved to: {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Arabic captions for images using Qwen2.5-VL model"
    )
    parser.add_argument("--image_folder", type=str, required=True, help="Path to folder containing images")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to output CSV file")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="Model name")
    parser.add_argument("--max_tokens", type=int, default=128, help="Max tokens (default: 128)")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Optional local checkpoint path")
    args = parser.parse_args()

    captioner = ArabicImageCaptioner(model_name=args.model_name, checkpoint_path=args.checkpoint_path)
    captioner.load_model()
    captioner.process_folder(image_folder=args.image_folder, output_csv=args.output_csv)


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()

