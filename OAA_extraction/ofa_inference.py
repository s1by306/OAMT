

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import sys
import os




current_dir = os.path.dirname(os.path.abspath(__file__))
OFA_ROOT = os.path.join(current_dir, "OFA")
sys.path.append(OFA_ROOT)

from fairseq import utils, tasks
from fairseq import checkpoint_utils
from tasks.mm_tasks.caption import CaptionTask
from models.ofa import OFAModel
from utils.eval_utils import eval_step

_initialized = {
    "model": None,
    "cfg": None,
    "task": None,
    "generator": None,
    "patch_resize_transform": None,
    "text_dict": None
}

def initialize_ofa():
    if _initialized["model"] is not None:
        return


    model_path = os.path.join(OFA_ROOT, "checkpoints/caption_base_best.pt")
    bpe_dir = os.path.join(OFA_ROOT, "utils/BPE")

    tasks.register_task('caption', CaptionTask)


    overrides = {
        "bpe_dir": bpe_dir,
        "eval_cider": False,
        "beam": 5,
        "max_len_b": 16,
        "no_repeat_ngram_size": 3,
        "seed": 7
    }


    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        utils.split_paths(model_path),
        arg_overrides=overrides
    )


    use_cuda = torch.cuda.is_available()
    use_fp16 = False
    for model in models:
        model.eval()
        if use_fp16:
            model.half()
        if use_cuda:
            model.cuda()
        model.prepare_for_inference_(cfg)


    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    patch_resize_transform = transforms.Compose([
        lambda image: image.convert("RGB"),
        transforms.Resize((cfg.task.patch_image_size, cfg.task.patch_image_size),
                          interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


    _initialized.update({
        "model": models[0],
        "cfg": cfg,
        "task": task,
        "generator": task.build_generator(models, cfg.generation),
        "patch_resize_transform": patch_resize_transform,
        "text_dict": task.src_dict
    })

def generate_OFA_caption(image: Image.Image) -> str:
    if _initialized["model"] is None:
        initialize_ofa()

    model = _initialized["model"]
    cfg = _initialized["cfg"]
    task = _initialized["task"]
    generator = _initialized["generator"]
    patch_resize_transform = _initialized["patch_resize_transform"]
    text_dict = _initialized["text_dict"]

    patch_image = patch_resize_transform(image).unsqueeze(0)
    patch_mask = torch.tensor([True])

    pad_idx = text_dict.pad()
    bos_item = torch.LongTensor([text_dict.bos()])
    eos_item = torch.LongTensor([text_dict.eos()])

    def encode_text(text, append_bos=False, append_eos=False):
        s = text_dict.encode_line(
            line=task.bpe.encode(text),
            add_if_not_exist=False,
            append_eos=False
        ).long()
        if append_bos:
            s = torch.cat([bos_item, s])
        if append_eos:
            s = torch.cat([s, eos_item])
        return s

    src_text = encode_text(" what does the image describe?", append_bos=True, append_eos=True).unsqueeze(0)
    src_length = torch.LongTensor([s.ne(pad_idx).long().sum() for s in src_text])

    sample = {
        "id": np.array(['42']),
        "net_input": {
            "src_tokens": src_text,
            "src_lengths": src_length,
            "patch_images": patch_image,
            "patch_masks": patch_mask
        }
    }

    use_cuda = torch.cuda.is_available()
    sample = utils.move_to_cuda(sample) if use_cuda else sample
    if hasattr(model, 'cfg') and model.cfg.common.fp16:
        sample = utils.apply_to_sample(apply_half, sample)

    with torch.no_grad():
        result, _ = eval_step(task, generator, [model], sample)

    return result[0]['caption']

def apply_half(t):
    if t.dtype is torch.float32:
        return t.to(dtype=torch.half)
    return t

if __name__ == "__main__":
    img = Image.open("test.jpg")
    print("Generated Caption:", generate_OFA_caption(img))