# -*- coding: utf-8 -*-
# @Date    : 2023-02-09 08:46:47
# @Author  : Shangyu.Xing (starreeze@foxmail.com)
"""
This script converts the 2 datasets to a (mostly) uniform intermediate format
Note that entity feature data is flattened into a single dim (not affecting entity idx)
outputs:
mention_text_raw, mention_image_paths, start/end_pos, entity_attr_raw, entity_image_paths, (entity_idx), answer
"""

from __future__ import annotations
import sys, os, json, hashlib, re
import numpy as np
from tqdm import tqdm
from PIL import Image
from urllib.parse import unquote
from transformers import BertTokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from common.args import *


def save_np(dir, type, **kwargs):
    print("storing data")
    for k, v in tqdm(kwargs.items()):
        data = np.asarray(v)
        np.save(os.path.join(dir, "%s_%s.npy" % (k.replace("_", "-"), type)), data)


class MentionPositionProcessor:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

    def __call__(self, sentences, starts, ends):
        before_mention = [sentence[:start] for sentence, start in zip(sentences, starts)]
        mentions = [sentence[start:end] for sentence, start, end in zip(sentences, starts, ends)]
        mention_starts = (
            np.sum(
                self.tokenizer(before_mention, return_tensors="np", padding=True, truncation=True)["attention_mask"],
                axis=-1,
            )
            - 2  # CLS and End Of Sentence
        )
        mention_ends = (
            np.sum(
                self.tokenizer(mentions, return_tensors="np", padding=True, truncation=True)["attention_mask"], axis=-1
            )
            - 2
            + mention_starts
        )
        return mention_starts, mention_ends


class WDProcess:
    def __init__(self, mention_position_processor):
        self.mention_position_processor = mention_position_processor
        self.entity2image: dict[str, list[str]] = {}
        print("loading image path dict")
        with open(entity2image_path, "r") as f:
            f.readline()
            for line in tqdm(f.readlines()):
                if line:
                    line = line.strip().split("@@@@")
                    self.entity2image[line[0]] = line[1].split("[AND]")

    def __call__(self, type):
        mention_text, mention_image, start_pos, end_pos, answer = [], [], [], [], []
        entity_image, entity_brief = [], []
        with open(mention_text_path % type, "r") as f:
            data: list[list] = json.load(f)
        with open(entity2brief_path % type, "r") as f:
            entity2brief = json.load(f)
        print(f"loading {type} data...")
        image_error_count, brief_missing_count, entry_missing_count, no_match_count = (0, 0, 0, 0)
        for item in tqdm(data):
            candidates = [unquote(candidate.split("/")[-1]) for candidate in item[7]]
            answer_name = unquote(item[6].split("/")[-1])
            try:
                answer.append(candidates.index(answer_name))
            except ValueError:
                no_match_count += 1
                answer.append(num_candidates_data)
            while len(candidates) < num_candidates_data:
                candidates.append("__nil__")
            candidates.append(answer_name)
            mention_text.append(item[0])
            mention_image.append(self.get_image_path(item[1]))
            start_pos.append(item[9])
            end_pos.append(item[10])
            for name in candidates:
                try:
                    brief = (name + ": " + entity2brief[name])[:max_entity_attr_char_len]
                except KeyError:
                    brief = "" if name == "__nil__" else name
                    brief_missing_count += 1
                entity_brief.append(brief)
                image = self.get_entity_image(name)
                entity_image.append(image)
                image_error_count += image == default_image
        print("=============== statistics =================")
        print("all data:", len(data))
        print("cleaned data:", len(mention_text))
        print("image errors:", image_error_count)
        print("brief missing:", brief_missing_count)
        print("entity missing:", entry_missing_count)
        print("no matching:", no_match_count)
        start_pos, end_pos = self.mention_position_processor(mention_text, start_pos, end_pos)
        save_np(
            preprocess_dir,
            type,
            mention_text_raw=mention_text,
            mention_image_path=mention_image,
            start_pos=start_pos,
            end_pos=end_pos,
            answer=answer,
            entity_image_path=entity_image,
            entity_attr_raw=entity_brief,
        )

    @staticmethod
    def get_image_path(url: str) -> str:
        image_path = url.split("/")[-1]
        prefix = hashlib.md5(image_path.encode()).hexdigest()
        suffix = re.sub(r"(\S+(?=\.(jpg|JPG|png|PNG|svg|SVG)))|(\S+(?=\.(jpeg|JPEG)))", "", image_path)
        image_path = os.path.join(image_dir, prefix + suffix)
        image_path = image_path.replace(".svg", ".png").replace(".SVG", ".png")
        # do some checks
        try:
            image = Image.open(image_path)
            if image.size[0] < min_image_size[0] or image.size[1] < min_image_size[1]:
                raise ValueError("Image is too small")
            image = image.resize((224, 224))
        except Exception as e:
            # print(f"{image_path} error: {e}")
            return default_image
        return image_path

    def get_entity_image(self, name: str) -> str:
        image = default_image
        try:
            for url in self.entity2image[name]:
                image = self.get_image_path(url)
                if image != default_image:
                    break
        except KeyError:
            pass
        return image


class WMProcess:
    def __init__(self, mention_position_processor) -> None:
        self.mention_position_processor = mention_position_processor
        print("building dict...")
        self.id2candidate: dict[str, list[str]] = {}
        with open(candidate_path, "r") as f:
            for line in tqdm(f.readlines()):
                items = line.strip().split("\t")
                self.id2candidate[items[0]] = items[1:]

    def __call__(self, type):
        with open(mention_text_path % type, "r") as f:
            data: dict[str, dict] = json.load(f)
        mention_text, start_pos, end_pos, answer, entity_name = [], [], [], [], []
        no_match_count, mention_not_found = (0, 0)
        for id, info in tqdm(data.items(), total=len(data)):
            candidate = self.id2candidate[id]
            try:
                start = info["sentence"].index(info["mentions"])
                start_pos.append(start)
                end_pos.append(start + len(info["mentions"]))
            except ValueError:
                mention_not_found += 1
                continue
            try:
                answer.append(candidate.index(info["answer"]))
            except ValueError:
                no_match_count += 1
                answer.append(num_candidates_data)
            mention_text.append(info["sentence"])
            entity_name.extend(candidate + [info["answer"]])  # append answer to the end

        print("=============== statistics =================")
        print("all data:", len(data))
        print("cleaned data:", len(mention_text))
        print("no matching:", no_match_count)
        print("mention not found:", mention_not_found)
        start_pos, end_pos = self.mention_position_processor(mention_text, start_pos, end_pos)
        save_np(
            preprocess_dir,
            type,
            mention_text_raw=mention_text,
            entity_name_raw=entity_name,
            start_pos=start_pos,
            end_pos=end_pos,
            answer=answer,
        )


def main():
    mpp = MentionPositionProcessor()
    if dataset_name == "wikidiverse":
        processor = WDProcess(mpp)
    elif dataset_name == "wikimel":
        processor = WMProcess(mpp)
    for type in ["valid", "train", "test"]:
        processor(type)  # type: ignore


if __name__ == "__main__":
    main()
