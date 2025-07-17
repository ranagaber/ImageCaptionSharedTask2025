#!/usr/bin/env python3
"""
Arabic Image Captioning using Qwen2.5-VL-7B-Instruct
"""

import os
import csv
import argparse
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

from transformers import BitsAndBytesConfig  # ✅ Add if using quant
from qwen_vl_utils import process_vision_info


class ArabicImageCaptioner:
    def __init__(self, model_name="Qwen/Qwen2.5-VL-7B-Instruct", checkpoint_path=None,
                 use_cpu=False, use_quant=False):
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.processor = None

        # ✅ Flexible device
        self.use_cpu = use_cpu
        self.use_quant = use_quant

        if self.use_cpu:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self):
        print(f"Loading model: {self.model_name}")
        print(f"Using device: {self.device}")

        loading_path = self.checkpoint_path if self.checkpoint_path else self.model_name

        model_args = {
            "attn_implementation": "eager"
        }

        if self.use_cpu:
            model_args["device_map"] = {"": "cpu"}
            model_args["torch_dtype"] = torch.float32  # CPU = FP32
        elif self.use_quant:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            model_args["device_map"] = "auto"
            model_args["quantization_config"] = bnb_config
        else:
            model_args["device_map"] = "auto"
            model_args["torch_dtype"] = torch.float16  # ✅ Use FP16 for GPU

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            loading_path, **model_args
        )

        self.processor = AutoProcessor.from_pretrained(self.model_name)

        # ✅ Force processor to resize images if it supports it
        if hasattr(self.processor, "image_processor"):
            self.processor.image_processor.size = {"height": 512, "width": 512}

        print("Model and processor loaded successfully!")

    def generate_caption(self, image_path, max_new_tokens=128):
        try:
            image = Image.open(image_path).convert("RGB")
            image = image.resize((512, 512))

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {
                            "type": "text",
                            "text": (
                                "You are an expert in visual scene understanding and multilingual caption generation."
                                " Analyze the content of this image, which is potentially related to the Palestinian Nakba"
                                " and Israeli occupation of Palestine, and provide a concise and meaningful caption in Arabic"
                                " (15 to 50 words). The caption should reflect the scene's content, emotional context,"
                                " and be culturally appropriate. Do not include any English or metadata."
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

            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            # ✅ Clear up VRAM
            del image_inputs, inputs, generated_ids, generated_ids_trimmed
            torch.cuda.empty_cache()

            return output_text[0].strip()

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return ""

    def process_folder(self, image_folder, output_csv, supported_formats=('.png', '.jpg', '.jpeg')):
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

        print(f"Processing complete! Results saved to: {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Generate Arabic captions for images")
    parser.add_argument("--image_folder", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--checkpoint_path", type=str, default=None)

    # ✅ New args to easily toggle CPU or quantization
    parser.add_argument("--use_cpu", action="store_true")
    parser.add_argument("--use_quant", action="store_true")

    args = parser.parse_args()

    captioner = ArabicImageCaptioner(
        model_name=args.model_name,
        checkpoint_path=args.checkpoint_path,
        use_cpu=args.use_cpu,
        use_quant=args.use_quant
    )
    captioner.load_model()
    captioner.process_folder(
        image_folder=args.image_folder,
        output_csv=args.output_csv
    )


if __name__ == "__main__":
    main()
