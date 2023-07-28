# -*- coding: utf-8 -*-
# @Date    : 2023-02-03 15:40:10
# @Author  : Shangyu.Xing (starreeze@foxmail.com)
"""extract image features(mention and entity) with resnet and rcnn"""

from __future__ import annotations
import json, torch, os, sys, torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoFeatureExtractor, ResNetModel
from tqdm import tqdm
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from common.utils import load_image, NpyWriter
from common.args import *

batch_size = 1
num_workers = 0
extract_feature = True
extract_object = True
reshape_image = True


class ImageData(Dataset):
    """read data from a list of images files; output [C,H,W]"""

    def __init__(self, processor, image_file_path, *args):
        super().__init__()
        self.processor = processor
        self.image_file_path = image_file_path

    def __len__(self):
        return len(self.image_file_path)

    def __getitem__(self, idx):
        image_file = load_image(self.image_file_path[idx], default_image)
        return self.processor(image_file)


class ImageRegionData(ImageData):
    """
    read data from a list of images files, extracting regions according to region tensor,
    which will result in Nx length. output: [C, H, W]
    """

    def __init__(self, processor, image_file_path, regions, *args):
        super().__init__(processor, image_file_path, *args)
        self.regions = regions
        self.N = regions.shape[1]

    def __len__(self):
        return super().__len__() * self.N

    def __getitem__(self, idx):
        image_idx, region_idx = idx // self.N, idx % self.N
        image_file = load_image(self.image_file_path[image_idx], default_image)
        region = self.regions[image_idx, region_idx]
        return self.processor(image_file.crop(region.tolist()))


def get_loader(data_type, *args, collate_fn=None):
    dataset = data_type(*args)
    return DataLoader(dataset, batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)


class FeatureProcessor:
    """to feed into resnet"""

    def __init__(self, resnet_processor):
        self.processor = resnet_processor

    def __call__(self, image: Image.Image):
        image = image.resize(image_input_size)
        return self.processor(image, return_tensors="pt")["pixel_values"].squeeze(0)


class ObjectProcessor:
    """to feed into object detector"""

    def __call__(self, image: Image.Image):
        image = image.resize(image_input_size)
        return torch.tensor(np.asarray(image, dtype=np.uint8).transpose((2, 0, 1)) / np.array(255.0, dtype=np.float32))


class FeatureExtractor:
    """to extract resnet features"""

    def __init__(self, model):
        self.model = model.to("cuda")

    def infer(self, data, output_type, save_path=None) -> np.ndarray | NpyWriter:
        features = NpyWriter(save_path) if save_path else []
        with torch.no_grad():
            for batch in tqdm(data):
                output = self.model(batch.to("cuda"))[output_type].to("cpu").numpy()
                s = output.shape
                features.extend(output.reshape(s[0], s[1], s[3] * s[2]).transpose((0, 2, 1)))
        if not save_path:
            features = np.concatenate(features, 0)  # type: ignore
        return features  # type: ignore


class ObjectExtractor:
    """to extract object boxes and scores"""

    def __init__(self, model, topk):
        self.model = model.to("cuda")
        self.topk = topk

    def infer(self, data) -> tuple[torch.Tensor, torch.Tensor]:
        boxes, scores = [], []
        with torch.no_grad():
            for batch in tqdm(data):
                for output in self.model([sample.to("cuda") for sample in batch]):
                    score = torch.zeros([self.topk])
                    score[: min(self.topk, len(output["scores"]))] = output["scores"][: self.topk].to("cpu")
                    box = torch.tensor([default_box]).tile([self.topk, 1])
                    box[: min(self.topk, len(output["boxes"]))] = output["boxes"][: self.topk].to("cpu")
                    boxes.append(box)
                    scores.append(score)
        return torch.stack(boxes), torch.stack(scores)


class Inferrer:
    """do model infer on type["valid", "train", "test"] and name["mention", "entity"]"""

    def __init__(self):
        self.image_processor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-152")
        resnet = ResNetModel.from_pretrained("microsoft/resnet-152")
        self.feature_extractor = FeatureExtractor(resnet)
        if extract_object:
            if drin_object_detector == "faster_rcnn":
                self.object_detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                    weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
                )
            elif drin_object_detector == "mask_rcnn":
                self.object_detector = torchvision.models.detection.maskrcnn_resnet50_fpn(
                    weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
                )
            self.object_detector.eval()

    def infer(self, type: str, name: str, feature_output: str, object_output: str, image_file_path: list[str]):
        if extract_feature:
            print(f"extracting {name} {type} features")
            image_data = get_loader(ImageData, FeatureProcessor(self.image_processor), image_file_path)
            output_path = os.path.join(preprocess_dir, f"{name}-image-feature_{type}.npy")
            features = self.feature_extractor.infer(image_data, feature_output, output_path)
            features.close()  # type: ignore

        if extract_object:
            print(f"extracting {name} {type} objects")
            image_data = get_loader(ImageData, ObjectProcessor(), image_file_path, collate_fn=lambda x: x)
            object_extractor = ObjectExtractor(self.object_detector, object_topk[name])
            boxes, scores = object_extractor.infer(image_data)
            np.save(os.path.join(preprocess_dir, f"{name}-object-score_{type}.npy"), scores.numpy())

            image_data = get_loader(ImageRegionData, FeatureProcessor(self.image_processor), image_file_path, boxes)
            output_path = os.path.join(preprocess_dir, f"{name}-object-feature_{type}.npy")
            features = self.feature_extractor.infer(image_data, object_output, output_path)
            features.reshape([-1, object_topk[name], *(features.shape[1:])]).close()  # type: ignore


def main():
    inferrer = Inferrer()
    for type in ["valid", "train", "test"]:
        with open(mention_text_path % type, "r") as f:
            mention_text = json.load(f)
        if dataset_name == "wikimel":
            image_file_path = [
                os.path.join(mention_image_dir, k.split("-")[0])
                for k, v in mention_text.items()
                if v["mentions"] in v["sentence"]
            ]
        elif dataset_name == "wikidiverse":
            image_file_path = np.load(os.path.join(preprocess_dir, f"entity-image-path_{type}.npy"))
            inferrer.infer(type, "entity", "pooler_output", "pooler_output", image_file_path)
            image_file_path = np.load(os.path.join(preprocess_dir, f"mention-image-path_{type}.npy"))
        inferrer.infer(type, "mention", "last_hidden_state", "pooler_output", image_file_path)  # type: ignore
    if dataset_name == "wikimel":
        with open(qid2entity_path, "r") as f:
            qid2name = json.load(f)
        image_file_path = [os.path.join(entity_image_dir, k) for k in qid2name.keys()]
        inferrer.infer("all", "entity", "pooler_output", "pooler_output", image_file_path)


if __name__ == "__main__":
    main()
