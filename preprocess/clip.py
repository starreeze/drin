# -*- coding: utf-8 -*-
# @Date    : 2023-02-02 10:22:49
# @Author  : Shangyu.Xing (starreeze@foxmail.com)

from __future__ import annotations
import json, torch, os, sys
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
from transformers import CLIPProcessor, CLIPModel

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from common.utils import pad_tokens, load_image
from common.args import *


batch_size = 1  # larger than 1 will result in clip errors
num_workers = 0
max_clip_len = 77
process_mention = True
process_entity_attr = True


def squeeze_dict(d):
    return {k: v.squeeze(0) for k, v in d.items()}


# old wikimel dataset
class MIETDataWM(Dataset):
    """mention image & entity text"""

    def __init__(self, processor, type: str):
        super().__init__()
        self.processor = processor
        with open(qid2attr_path, "r") as f:
            self.qid2attr = json.load(f)
        with open(qid2entity_path, "r") as f:
            self.qid2name = json.load(f)
        self.entity_qid = np.load(os.path.join(preprocess_dir, f"entity-name-raw_{type}.npy"))
        self.entity_qid = self.entity_qid.reshape((-1, num_candidates_model))
        with open(mention_text_path % type, "r") as f:
            mention_text = json.load(f)
        self.mention_id = [k.split("-")[0] for k, v in mention_text.items() if v["mentions"] in v["sentence"]]
        assert len(self.mention_id) == len(self.entity_qid)

    def __len__(self):
        return len(self.mention_id)

    def __getitem__(self, idx):
        mention_image = os.path.join(mention_image_dir, self.mention_id[idx])
        mention_image = load_image(mention_image, default_image)
        entity_text = [self.qid2name[qid] + ". " + self.qid2attr[qid].replace(".", ";") for qid in self.entity_qid[idx]]
        processed = self.processor(text=entity_text, images=mention_image, return_tensors="pt", padding=True)
        return squeeze_dict(pad_tokens(processed, max_clip_len))


class EIMTDataWM(Dataset):
    """entity image & mention text"""

    def __init__(self, processor, type: str):
        super().__init__()
        self.processor = processor
        self.entity_qid = np.load(os.path.join(preprocess_dir, f"entity-name-raw_{type}.npy"))
        self.entity_qid = self.entity_qid.reshape((-1, num_candidates_model))
        self.mention_text = np.load(os.path.join(preprocess_dir, f"mention-text-raw_{type}.npy"))
        assert len(self.mention_text) == len(self.entity_qid)

    def __len__(self):
        return len(self.mention_text)

    def __getitem__(self, idx):
        entity_images = []
        for qid in self.entity_qid[idx]:
            entity_image_name = os.path.join(entity_image_dir, qid)
            entity_image = load_image(entity_image_name, default_image)
            entity_images.append(entity_image)
        mention_text = self.mention_text[idx]
        processed = self.processor(text=mention_text, images=entity_images, return_tensors="pt", padding=True)
        return squeeze_dict(pad_tokens(processed, max_clip_len))


class MIETData(Dataset):
    """mention image & entity text"""

    def __init__(self, processor, type: str):
        super().__init__()
        self.processor = processor
        self.mention_image = np.load(os.path.join(preprocess_dir, f"mention-image-path_{type}.npy"))
        self.entity_text = np.load(os.path.join(preprocess_dir, f"entity-attr-raw_{type}.npy"))
        self.entity_text = self.entity_text.reshape((-1, num_candidates_model))
        assert len(self.mention_image) == len(self.entity_text)

    def __len__(self):
        return len(self.mention_image)

    def __getitem__(self, idx):
        mention_image = load_image(self.mention_image[idx], default_image)
        entity_text = list(self.entity_text[idx])
        processed = self.processor(text=entity_text, images=mention_image, return_tensors="pt", padding=True)
        return squeeze_dict(pad_tokens(processed, max_clip_len))


class EIMTData(Dataset):
    """entity image & mention text"""

    def __init__(self, processor, type: str):
        super().__init__()
        self.processor = processor
        self.entity_image = np.load(os.path.join(preprocess_dir, f"entity-image-path_{type}.npy"))
        self.entity_image = self.entity_image.reshape((-1, num_candidates_model))
        self.mention_text = np.load(os.path.join(preprocess_dir, f"mention-text-raw_{type}.npy"))
        assert len(self.mention_text) == len(self.entity_image)

    def __len__(self):
        return len(self.mention_text)

    def __getitem__(self, idx):
        entity_images = [load_image(entity_image, default_image) for entity_image in self.entity_image[idx]]
        mention_text = self.mention_text[idx]
        processed = self.processor(text=mention_text, images=entity_images, return_tensors="pt", padding=True)
        return squeeze_dict(pad_tokens(processed, max_clip_len))


class ClipInfer:
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda")  # type: ignore

    def infer(self, dataset, output_type):
        data = DataLoader(dataset, batch_size, shuffle=False, num_workers=num_workers)
        features = []
        with torch.no_grad():
            for batch in tqdm(data):
                if output_type == "logits_per_image":  # squeeze text
                    batch["input_ids"] = self.squeeze_candidates(batch["input_ids"])
                    batch["attention_mask"] = self.squeeze_candidates(batch["attention_mask"])
                else:  # squeeze image
                    batch["pixel_values"] = self.squeeze_candidates(batch["pixel_values"])
                inputs = {k: v.to("cuda") for k, v in batch.items()}
                outputs = self.unsqueeze_candidates(self.model(**inputs)[output_type]).to("cpu")
                features.append(outputs)
        res = torch.cat(features, dim=0).numpy()
        assert list(res.shape) == [len(dataset), num_candidates_model]
        return res

    @staticmethod
    def squeeze_candidates(x):
        s = x.shape
        assert s[1] == num_candidates_model
        return x.reshape([s[0] * s[1]] + list(s[2:]))

    @staticmethod
    def unsqueeze_candidates(x):
        assert len(x.shape) == 2
        return x.reshape([-1, num_candidates_model])


def main():
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip = ClipInfer()
    for type in ["valid", "train", "test"]:
        print(type)
        target_file = os.path.join(preprocess_dir, f"similarity-miet_{type}.npy")
        if not os.path.exists(target_file):
            miet = MIETData(processor, type)
            outputs = clip.infer(miet, "logits_per_image")
            np.save(target_file, outputs)
        target_file = os.path.join(preprocess_dir, f"similarity-eimt_{type}.npy")
        if not os.path.exists(target_file):
            eimt = EIMTData(processor, type)
            outputs = clip.infer(eimt, "logits_per_text")
            np.save(target_file, outputs)


if __name__ == "__main__":
    main()
