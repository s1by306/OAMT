import argparse
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os
import csv
import torch

def generate_captions(img_dir, output_tsv, device="cuda"):
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    image_files = sorted([
        f for f in os.listdir(img_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
    ])

    if not image_files:
        print(f"No images found in directory: {img_dir}")
        return

    print(f"Found {len(image_files)} images. Starting caption generation on {device.upper()}...")

    with open(output_tsv, 'w', newline='', encoding='utf-8') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t')
        writer.writerow(['image_id', 'caption'])

        processed = 0
        for img_file in image_files:
            try:
                image_path = os.path.join(img_dir, img_file)
                image = Image.open(image_path).convert('RGB')
                inputs = processor(image, return_tensors="pt").to(device)
                generated_ids = model.generate(**inputs, max_length=50)
                caption = processor.decode(generated_ids[0], skip_special_tokens=True).strip()
                writer.writerow([img_file, caption])
                processed += 1
                if processed % 10 == 0:
                    print(f"Processed {processed}/{len(image_files)} images")

            except Exception as e:
                print(f"Error processing {img_file}: {str(e)}")
                writer.writerow([img_file, f"ERROR: {str(e)}"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate image captions using BLIP model')
    parser.add_argument('--img_dir', type=str, required=True, help='Directory containing input images')
    parser.add_argument('--output_tsv', type=str, required=True, help='Output TSV file path')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='Device to use (cuda or cpu)')
    args = parser.parse_args()
    output_dir = os.path.dirname(args.output_tsv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    generate_captions(
        img_dir=args.img_dir,
        output_tsv=args.output_tsv,
        device=args.device
    )
    print(f"\nCaption generation complete. Results saved to: {args.output_tsv}")