
from PIL import Image
import torch, numpy as np
from typing import List

from pytorch_pretrained_bert import BertTokenizer, BertConfig
from oscar.modeling.modeling_bert import BertForImageCaptioning
from oscar.wrappers import OscarTensorizer
from scene_graph_benchmark.wrappers import VinVLVisualBackbone

CKPT = "./vinvl-base-image-captioning"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg   = BertConfig.from_json_file(f"{CKPT}/config.json")
tok   = BertTokenizer.from_pretrained(CKPT)
model = BertForImageCaptioning.from_pretrained(CKPT, config=cfg).to(DEVICE).eval()
tensorizer = OscarTensorizer(tokenizer=tok, device=DEVICE)
detector   = VinVLVisualBackbone()


def generate_OSCAR_caption(img: Image.Image) -> str:
    dets = detector(img)
    feats = np.concatenate((dets["features"], dets["spatial_features"]), 1)  # [n, 2054]
    labels: List[str] = [detector.CLASSES[i] for i in dets["classes"]]
    vis_feats = torch.from_numpy(feats).unsqueeze(0).to(DEVICE)            # [1, n, 2054]
    inputs = tensorizer.encode(vis_feats, labels=[labels])
    with torch.no_grad():
        out = model(**inputs)
    caption = tensorizer.decode(out)[0][0]["caption"]
    return caption
