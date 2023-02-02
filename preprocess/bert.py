# -*- coding: utf-8 -*-
# @Date    : 2023-01-27 15:40:10
# @Author  : Shangyu.Xing (starreeze@foxmail.com)
"""extract text features(mention and entity) with bert"""

from __future__ import annotations
import json, torch, os
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import numpy as np

batch_size = 128
num_workers = 0
max_mention_token_len = 128
max_entity_attr_token_len = 64
max_bert_len = 512
qid2attr_path = "/home/data_91_c/xsy/mel-dataset/wikimel/entities/qid2abs.json"
qid2name_path = "/home/data_91_c/xsy/mel-dataset/wikimel/candidates/qid2ne.json"
text_preprocess_dir = "/home/data_91_c/xsy/mel-dataset/text_preprocessed"
output_dir = "/data0/xsy/mel"
process_mention = True
process_entity_attr = True


class TextArrayData(Dataset):
    """data stored as numpy arrays"""

    def __init__(self, tokenizer, file_path):
        super().__init__()
        self.tokenizer = tokenizer
        self.raw_data = np.load(file_path)

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        text = str(self.raw_data[idx])
        token = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)  # type: ignore
        token = {k: torch.constant_pad_nd(v[0], [0, max_bert_len - v.shape[-1]]) for k, v in token.items()}
        return token


class QidJsonData(Dataset):
    """data stored as json: qid -> text"""

    def __init__(self, tokenizer, file_path, qid2name):
        super().__init__()
        self.tokenizer = tokenizer
        self.qid2name = qid2name
        with open(file_path, "r") as f:
            self.raw_data = list(json.load(f).items())

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        qid, text = self.raw_data[idx]
        text = self.qid2name[qid] + ". " + str(self.raw_data[idx][1]).replace(".", ";")
        token = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)  # type: ignore
        token = {k: torch.constant_pad_nd(v[0], [0, max_bert_len - v.shape[-1]]) for k, v in token.items()}
        return token

    def write_mapping(self, file_path):
        d = {k[0]: i for i, k in enumerate(self.raw_data)}
        with open(file_path, "w") as f:
            json.dump(d, f)


class BertInfer:
    def __init__(self):
        self.model = BertModel.from_pretrained("bert-base-cased").to("cuda")  # type: ignore

    def infer(self, text_dataset, output_type, max_len=None):
        data = DataLoader(text_dataset, batch_size, shuffle=False, num_workers=num_workers)
        features, paddings = [], []
        with torch.no_grad():
            for batch in tqdm(data):
                inputs = {k: v.to("cuda") for k, v in batch.items()}
                outputs = self.model(**inputs)[output_type].to("cpu")
                if output_type == "last_hidden_state":
                    outputs = outputs[:, :max_len]
                    paddings.append(inputs["attention_mask"][:, :max_len].to("cpu"))
                features.append(outputs)
        if output_type == "last_hidden_state":
            return torch.cat(features, dim=0).numpy(), torch.cat(paddings, dim=0).numpy()
        return torch.cat(features, dim=0).numpy()


def main():
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    bert = BertInfer()
    if process_mention:
        for type in ["train", "valid", "test"]:
            print("mention " + type)
            mention_text = TextArrayData(tokenizer, os.path.join(text_preprocess_dir, "mention-text-raw_%s.npy" % type))
            features, paddings = bert.infer(mention_text, "last_hidden_state", max_mention_token_len)
            np.save(os.path.join(output_dir, f"mention-text-feature_{type}.npy"), features)
            np.save(os.path.join(output_dir, f"mention-text-mask_{type}.npy"), paddings)
    print("entity")
    with open(qid2name_path, "r") as f:
        qid2name = json.load(f)
    if process_entity_attr:
        entity_text = QidJsonData(tokenizer, qid2attr_path, qid2name)
        entity_text.write_mapping(os.path.join(output_dir, "qid2idx.json"))
        features, paddings = bert.infer(entity_text, "last_hidden_state", max_entity_attr_token_len)
        np.save(os.path.join(output_dir, f"entity-attr-feature.npy"), features)
        np.save(os.path.join(output_dir, f"entity-attr-mask.npy"), paddings)


if __name__ == "__main__":
    main()
