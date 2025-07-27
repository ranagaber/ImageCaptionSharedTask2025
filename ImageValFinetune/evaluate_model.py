import os
import json
import pandas as pd
from tqdm import tqdm
from PIL import Image

def evaluate_model(self, checkpoint_path=None, test_images_dir=None, max_images=None):
    """Evaluate the fine-tuned model on test images and save generated captions."""
    if checkpoint_path:
        self.load_model(checkpoint_path)
    else:
        self.load_latest_model()

    if test_images_dir is None:
        test_images_dir = self.paths["test_images_dir"]

    image_files = sorted([
        os.path.join(test_images_dir, fname)
        for fname in os.listdir(test_images_dir)
        if fname.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    if max_images:
        image_files = image_files[:max_images]

    results = []
    print("Generating captions:")
    for image_path in tqdm(image_files):
        try:
            image = Image.open(image_path).convert("RGB")
            caption = self.generate_caption(image)
            results.append({
                "image_file": os.path.basename(image_path),
                "arabic_caption": caption
            })
        except Exception as e:
            print(f"❌ Error processing {os.path.basename(image_path)}: {e}")

    # ✅ Save results to JSON
    json_path = os.path.join(self.paths["output_dir"], "generated_captions.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"📝 Captions saved to {json_path}")

    # ✅ Save results to CSV
    csv_path = os.path.join(self.paths["output_dir"], "generated_captions.csv")
    pd.DataFrame(results).to_csv(csv_path, index=False)
    print(f"📝 Captions also saved to {csv_path}")

    return results
