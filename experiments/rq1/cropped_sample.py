import os
import random


from image_cutting.cutline_searching import *

base_dir = 'data/coco_2017/train_2017'
random.seed(42)
def get_image_files(directory):
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')
    return [(file,os.path.join(directory, file))
            for file in os.listdir(directory)
            if file.lower().endswith(image_extensions)]


file_tuples = get_image_files(base_dir)
image_paths = [file_tuple[1] for file_tuple in file_tuples]
sampled_paths = random.sample(image_paths, 200)

for idx,img_path in sampled_paths:
    with Image.open(img_path) as img:
        img_rgb = img.convert("RGB")
        OD_result = detection(img_rgb, processor_detection, model_detection)
        box_filtered = filt(OD_result, img_rgb)
        split = scan_horizontal(box_filtered, img_rgb)
        if len(split) > 0:
            for sp in split:
                image1, image2 = split_image(img_rgb, sp)
        else:
            width, height = img_rgb.size
            image1, image2 = divide_image(img_rgb, 0.25 * width, 0.75 * width)
    selected_image = random.choice([image1, image2])
    selected_image_path = os.path.join("./sample", f"sample_{idx}.jpg")
    selected_image.save(selected_image_path)


    image1.close()
    image2.close()
    selected_image.close()



