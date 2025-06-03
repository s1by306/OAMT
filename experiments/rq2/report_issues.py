import os
from OAA_extraction.triplet_extraction import extract_tuples_from_fragment, extract_nouns,merge
from compatiblity_check.check_rules import check_whole
from error_detection.metamorphic_rules import mr_1, mr_2

model_list = ['BLIP','GIT', 'OFA', 'OSCAR']
base_url = "http://images.cocodataset.org/train2017/"
base_dir = "../../data/train2017/"

def get_image_files(directory):
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')
    return [(file,os.path.join(directory, file))
            for file in os.listdir(directory)
            if file.lower().endswith(image_extensions)]

for model_name in model_list:
    with open(f'{model_name}_caption.txt', 'r') as file:
        lines = file.readlines()

    lines = [line.strip() for line in lines]

    tuples = [tuple(lines[i:i+3]) for i in range(0, len(lines), 4)]
    total_err = 0
    filenames = get_image_files(base_dir)

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
                    print(base_url + filenames[index], file = f)
                    print("",file = f)
                total_err += 1