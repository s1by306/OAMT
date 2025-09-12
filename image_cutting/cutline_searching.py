from object_detection import detection

def filt(bbox_list,image):
    width, height = image.size
    s_total = width*height
    box_filtered = []
    for label,bbox in bbox_list:
        s_bbox = (bbox[2]-bbox[0])*(bbox[3]-bbox[1])
        if s_bbox>s_total*0.02:
            box_filtered.append(bbox)
    return box_filtered

import heapq
def scan_horizontal(box_filtered,image):
    splitline = []
    result = []
    width, height = image.size
    for i in range(width):
        if i<width*0.15 or i>width*0.85:
            continue
        pos = i
        vaild = True
        left = 0
        right = 0
        for box in box_filtered:
            if box[0] > pos:
                right += 1
            elif box[2] < pos:
                left += 1
            else:
                vaild = False
                break;
        if left==0 or right==0:
            vaild = False
        if vaild is True:
            splitline.append(pos)
    left = 0
    right = 0
    for index in range(len(splitline)):
        if index == 0:
            left = splitline[index]
            right = left
            continue
        if splitline[index]-splitline[index-1]>1.5 or index==len(splitline)-1:
            result.append((left+right)/2)
            left = splitline[index]
            right = left
        else:
            right = splitline[index]
    closest = heapq.nsmallest(1, result, key=lambda x: (abs(x - width/2), x))

    return closest
def split_image(image,split_point):
    width, height = image.size
    # Split image horizontally at split_point
    image1 = image.crop((0, 0, split_point, height))
    image2 = image.crop((split_point, 0, width, height))
    # Save the cropped images
    return image1,image2

def divide_image(image,split_point1,split_point2):
    width, height = image.size
    image1 = image.crop((0, 0, split_point2, height))
    image2 = image.crop((split_point1, 0, width, height))
    return image1,image2

from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import requests
if __name__ == "__main__":
    device = torch.device('cuda')
    url = "http://images.cocodataset.org/train2017/000000555299.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    processor_detection = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
    model_detection = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm").to(device)
    OD_result = detection(image.convert("RGB"),processor_detection,model_detection,device=d)
    box_filtered = filt(OD_result,image)

    split = scan_horizontal(box_filtered,image)
    if len(split)>0:
        for sp in split:
            image1,image2 = split_image(image,sp)
    else:
        width, height = image.size
        image1,image2 = divide_image(image,0.25*width,0.75*width)
        image1.show()
        image2.show()