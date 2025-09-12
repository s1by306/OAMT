from image_cutting.cutline_searching import divide_image
from OAA_extraction.caption_collecting import caption_generate
from OAA_extraction.triplet_extraction import *
from compatiblity_check.check_rules import *
from error_detection.metamorphic_rules import mr_1,mr_2
import torch
import os
from PIL import Image
import requests

ratios = [0.15,0.20,0.25,0.30]
device = torch.device('cuda')
model_name = "GIT"

#cut 200 pic
images = [f for f in os.listdir('sample_200')]


for ratio in ratios:
    for idx,img in enumerate(images):
        img1, img2 = divide_image(img,ratio,1-ratio)
        img1.save("./ratio{}/{}_left".format(ratio,idx))
        img1.save("./ratio{}/{}_right".format(ratio,idx))

images = [f for f in os.listdir('sample_3000')]

for ratio in ratios:
    for idx,img in enumerate(images):
        img1, img2 = divide_image(img,ratio,1-ratio)
        img1.save("./ratio{}/{}_left".format(ratio,idx))
        img1.save("./ratio{}/{}_right".format(ratio,idx))
        caption_org = caption_generate(model_name=model_name,image=img)
        caption1 =caption_generate(model_name=model_name,image=img1)
        caption2 = caption_generate(model_name=model_name,image=img2)
        with open(f"{model_name}_caption.txt", "a") as file:
            file.write(caption_org+ '\n')
            file.write(caption1 + '\n')
            file.write(caption2 +'\n')
            file.write('\n')

base_url = "http://images.cocodataset.org/train2017/"

def get_image_files(directory):
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')
    return [(file,os.path.join(directory, file))
            for file in os.listdir(directory)
            if file.lower().endswith(image_extensions)]


with open(f'{model_name}_caption.txt', 'r') as file:
    lines = file.readlines()

lines = [line.strip() for line in lines]

tuples = [tuple(lines[i:i+3]) for i in range(0, len(lines), 4)]
total_err = 0

for index,sentence_tuple in enumerate(tuples):
    sentence1 = sentence_tuple[0]
    sentence2 = sentence_tuple[1]
    sentence3 = sentence_tuple[2]
    if sentence1 == sentence2 or sentence1 == sentence3:
        continue
    action_1 = extract_tuples_from_fragment(sentence1)
    nouns_1 = extract_nouns(sentence1)
    action_1 = merge(nouns_1,action_1)
    action_2 = extract_tuples_from_fragment(sentence2)
    nouns_2 = extract_nouns(sentence2)
    action_2 = merge(nouns_2,action_2)
    action_3 = extract_tuples_from_fragment(sentence3)
    nouns_3 = extract_nouns(sentence3)
    action_3 = merge(nouns_3,action_3)
    if not check_whole(sentence1,sentence2,sentence3)  :
        con1 = mr_1(action_1,action_2,action_3)
        con2 = mr_2(action_1,action_2,action_3)
        mr_1_total = 0
        mr_2_total = 0
        if con1:
            mr_1_total += 1
        if con2:
            mr_2_total += 1
        if con1 or con2:
            with open(f'output_{model_name}.txt','a') as f:
                print(sentence_tuple,file = f)
                print("",file = f)
            total_err += 1