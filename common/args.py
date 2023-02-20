# -*- coding: utf-8 -*-
# @Date    : 2023-01-04 10:01:03
# @Author  : Shangyu.Xing (starreeze@foxmail.com)

### model
##base
model_type = "drgcn"  # baseline or drgcn
if model_type == "baseline":
    # if True, extract mention names into independent sentences before bert
    pre_extract_mention = False  # forced to False if online_bert is False or model type is drgcn
    mention_final_layer_name = "multimodal"  # linear, transformer, multimodal or none
    # max pool or avg extract, forced to avg extract if mention_final_layer_name is linear and max pool if multimodal bi
    mention_final_representation = "max pool"
    mention_final_output_dim = 768
    entity_final_layer_name = "linear"
    entity_final_pooling = "avg"  # max, avg, bert_default; forced to bert_default if dataset is wikidiverse
    entity_final_output_dim = 768
    multimodal_subspace_activation = "gelu"
    mention_multimodal_attention = "bi"  # text or bi
elif model_type == "drgcn":
    gcn_embed_dim = 768
    num_gcn_layers = 2
    # we reuse unimodal mention_encoder and entity_encoder from baseline as vertex feature extractors
    mention_final_layer_name = "linear"  # linear, transformer or none
    mention_final_representation = "avg extract"
    entity_final_layer_name = "linear"
    entity_final_pooling = "avg"
    gcn_vertex_activation = "gelu"
    gcn_edge_activation = "sigmoid"
    mention_final_output_dim = gcn_embed_dim
    entity_final_output_dim = gcn_embed_dim

## encoders
# bert
max_bert_len = 512
bert_embed_dim = 768
CLS = 101
SEP = 102
finetune_bert = False  # forced to False if online_bert is False
online_bert = False

# resnet
resnet_embed_dim = 2048
resnet_num_region = 49
image_input_size = (224, 224)
min_image_size = (50, 50)
default_box = [0, 0, 50, 50]
object_topk = {"mention": 3, "entity": 1}

# transformer encoder
transformer_num_layers = 8
transformer_num_heads = 8
transformer_ffn_hidden_size = 512
transformer_ffn_activation = "gelu"
transformer_dropout = 0.1


### data
## base
entity_text_type = "attr"  # name, brief, attr; only attr is currently supported if online_bert is False
num_entity_sentence = 12  # if 0, disable zipping: every entity is a sentence
max_mention_name_len = 32  # max token length of mention name
max_mention_sentence_len = 128  # max token length of mention sentence, used in online bert
mention_mmap = "r"
entity_mmap = "r"

## dataset
dataset_name = "wikimel"
dataset_root = f"/home/data_91_c/xsy/mel-dataset/{dataset_name}/"
preprocess_dir = f"/data0/xsy/mel/{dataset_name}/"
default_image = "/home/data_91_c/xsy/mel-dataset/default.jpg"
if dataset_name == "wikimel":
    num_candidates_data = 100
    max_entity_attr_char_len = 128  # max char length of entity attribute, used in online bert
    max_entity_attr_token_len = 64  # max token length of entity attribute, used in offline bert
    qid2entity_path = dataset_root + "candidates/qid2ne.json"
    qid2attr_path = dataset_root + "entities/qid2abs.json"
    mention_text_path = dataset_root + "mentions/WIKIMEL_%s.json"
    candidate_path = dataset_root + "candidates/top100/candidates-answer.tsv"
    mention_image_dir = dataset_root + "mentions/KVQAimgs"
    entity_image_dir = dataset_root + "entities/cleaned-images"
elif dataset_name == "wikidiverse":
    num_candidates_data = 10
    max_entity_attr_char_len = 512  # max char length of entity attribute (desc), used in online bert
    max_entity_attr_token_len = 128  # max token length of entity attribute, used in offline bert
    mention_text_path = dataset_root + "candidates/%s_w_10cands.json"
    entity2image_path = dataset_root + "entities/wikipedia_entity2imgs.tsv"
    entity2brief_path = dataset_root + "entities/entity2brief_%s.json"
    image_dir = dataset_root + "images"
    mention_image_dir = entity_image_dir = image_dir
num_candidates_model = num_candidates_data + 1  # as the last is reserved for answer # type: ignore


### train
dataloader_workers = 0
shuffle_train_data = True
batch_size = 64
seed = 0
triplet_margin = 0.25
learning_rate = 1e-3
num_epoch = 30
test_epoch_interval = 10
if dataset_name == "wikimel":
    metrics_topk = [1, 5, 10, 20, 50]
elif dataset_name == "wikidiverse":
    metrics_topk = [1, 3, 5]


### debug
debug = False

if debug:
    shuffle_train_data = False
    num_epoch = test_epoch_interval = 1
    dataloader_workers = 0
