from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import requests


def detection(image, processor, model, device):
    inputs = processor(images=image, return_tensors="pt").to(device)
    outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
    results_processed = [];
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        results_processed.append((model.config.id2label[label.item()],box))
    return results_processed

if __name__ == '__main__':
    device = torch.device('cuda')
    url = "http://images.cocodataset.org/train2017/000000555299.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    processor_detection = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
    model_detection = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm").to(device)
    detection(image, processor_detection, model_detection)