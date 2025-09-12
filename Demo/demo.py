import os
import torch
from PIL import Image
from image_cutting.object_detection import detection,processor_detection,model_detection
from image_cutting.cutline_searching import divide_image,split_image,scan_horizontal,filt
from OAA_extraction.caption_collecting import caption_generate
from OAA_extraction.triplet_extraction import *
from compatiblity_check.check_rules import *
from error_detection.metamorphic_rules import mr_1,mr_2

device = torch.device('cuda')
image_path = "example"

def get_image_files(directory):
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')
    return [(file,os.path.join(directory, file))
            for file in os.listdir(directory)
            if file.lower().endswith(image_extensions)]


def process_images(image_dir, model_name, batch_size=4):
    file_tuples = get_image_files(image_dir)

    image_paths = [file_tuple[1] for file_tuple in file_tuples]
    captions = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        with torch.no_grad():
            images = []
            batch_captions = []
            for img_path in batch_paths:
                try:
                    with Image.open(img_path) as img:
                        img_rgb = img.convert("RGB")
                        images.append(img_rgb)

                        try:
                            batch_captions.append(caption_generate(img_rgb, model_name))
                        except ValueError:
                            continue

                        OD_result = detection(img_rgb, processor_detection, model_detection)
                        box_filtered = filt(OD_result, img_rgb)
                        split = scan_horizontal(box_filtered, img_rgb)

                        if len(split) > 0:
                            for sp in split:
                                image1, image2 = split_image(img_rgb, sp)
                        else:
                            width, height = img_rgb.size
                            image1, image2 = divide_image(img_rgb, 0.25 * width, 0.75 * width)

                        for sub_img in (image1, image2):
                            try:
                                images.append(sub_img.convert("RGB"))
                                batch_captions.append(caption_generate(sub_img.convert("RGB"), model_name))
                            except ValueError:
                                continue
                            finally:
                                sub_img.close()

                except Exception as e:
                    print(f"Error processing image {img_path}: {e}")
                    continue

            captions.extend(batch_captions)

            with open(f"{model_name}_caption.txt", "a") as file:
                for idx, string in enumerate(captions):
                    file.write(string + '\n')
                    if (idx + 1) % 3 == 0:
                        file.write('\n')
            print(f"{i} completed")

    return captions

captions = process_images(image_path, model_name="GIT",batch_size=4)
with open('GIT_captions.txt', 'r') as file:
    lines = file.readlines()

lines = [line.strip() for line in lines]

tuples = [tuple(lines[i:i+3]) for i in range(0, len(lines), 4)]
print(tuples[0:10])
total_err = 0
filenames = get_image_files(image_path)

for index,sentence_tuple in enumerate(tuples):
    sentence1 = sentence_tuple[0]
    sentence2 = sentence_tuple[1]
    sentence3 = sentence_tuple[2]
    if sentence1 == sentence2 or sentence1 == sentence3:
        continue
    action_1 = extract_tuples_from_fragment(sentence1)
    action_2 = extract_tuples_from_fragment(sentence2)
    nouns_2 = extract_nouns(sentence2)
    action_2 = merge(nouns_2,action_2)
    action_3 = extract_tuples_from_fragment(sentence3)
    nouns_3 = extract_nouns(sentence3)
    action_3 = merge(nouns_3,action_3)

    if not check_whole(sentence1,sentence2,sentence3) :
        if not mr_1(action_1,action_2,action_3):
            print(sentence1)
            print(sentence2)
            print(sentence3)
            print("MR1 violate")
        if not mr_2(action_1,action_2,action_3):
            print(sentence1)
            print(sentence2)
            print(sentence3)
            print("MR2 violate")






