from transformers import AutoProcessor, AutoModelForCausalLM
from transformers import BlipProcessor, BlipForConditionalGeneration
from OAA_extraction.oscar_inference import generate_OSCAR_caption
from OAA_extraction.ofa_inference import generate_OFA_caption

device = 'cuda'
git_processor = AutoProcessor.from_pretrained("microsoft/git-large-coco")
git_model = AutoModelForCausalLM.from_pretrained("microsoft/git-large-coco").to(device)
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
def caption_generate(image, model_name):
    if model_name == "GIT":
        model = git_model
        inputs = git_processor(images=image, return_tensors="pt")
        outputs = model.generate(**inputs)
        caption = blip_processor.decode(outputs[0], skip_special_tokens=True)
    elif model_name == "BLIP":
        model = blip_model
        inputs = git_processor(images=image, return_tensors="pt")
        outputs = model.generate(**inputs)
        caption = blip_processor.decode(outputs[0], skip_special_tokens=True)
    elif model_name == "OFA":
        caption = generate_OFA_caption(image)
    elif model_name == "Oscar":
        caption = generate_OSCAR_caption(image)
    return caption