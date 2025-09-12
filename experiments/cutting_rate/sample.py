import os
import random
import shutil

input_dir = "data/train2017/"
def sample_images(src_dir, dst_dir, k):
    os.makedirs(dst_dir, exist_ok=True)
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".gif")
    images = [f for f in os.listdir(src_dir) if f.lower().endswith(exts)]
    sampled = random.sample(images, k)
    for img in sampled:
        src_path = os.path.join(src_dir, img)
        dst_path = os.path.join(dst_dir, img)
        shutil.copy(src_path, dst_path)

sample_images(input_dir,'sample_200',200)
sample_images(input_dir,'sample_3000',3000)
