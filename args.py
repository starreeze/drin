# -*- coding: utf-8 -*-
# @Date    : 2023-01-04 10:01:03
# @Author  : Shangyu.Xing (starreeze@foxmail.com)

# base encoders
## bert
max_bert_len = 512
bert_embed_dim = 768
CLS = 101
SEP = 102
## resnet
resnet_embed_dim = 2048
resnet_num_region = 49

# model
## basic
finetune_bert = False  # this will be set to False if online_bert is False
online_bert = True
# if True, extract mention names into independent sentences before bert; set to False if online_bert is False
pre_extract_mention = False
mention_final_layer_name = "multimodal"  # linear will force the next option to extract avg and multimodal max pool
mention_final_representation = "avg extract"  # 'max pool' or 'avg extract'
mention_final_output_dim = 768
entity_final_layer_name = "linear"
entity_final_pooling = "max"  # max, mean, bert_default; will be set to bert_default if online_bert is False
entity_text_type = "attr"  # name, brief, attr; only attr is currently supported if online_bert is False
entity_final_output_dim = 768
## args for different layers
transformer_num_layers = 8
transformer_num_heads = 8
transformer_ffn_hidden_size = 512
transformer_ffn_activation = "gelu"
transformer_dropout = 0.1
multimodal_subspace_activation = "gelu"
mention_multimodal_attention = "text"  # text or bi

# other data
num_entity_sentence = 12  # if 0, disable zipping: every entity is a sentence
num_candidates = 101
max_mention_name_len = 32  # max token length of mention name
max_mention_sentence_len = 128  # max token length of mention sentence, used in online bert
max_entity_attr_char_len = 128  # max char length of entity attribute, used in online bert
max_entity_attr_token_len = 64  # max token length of entity attribute, used in offline bert

# path
qid2entity_answer_path = "/home/data_91_c/xsy/mel-dataset/wikimel/candidates/qid2ne.json"
qid2attr_path = "/home/data_91_c/xsy/mel-dataset/wikimel/entities/qid2abs.json"
text_preprocess_dir = "/home/data_91_c/xsy/mel-dataset/text_preprocessed"
image_preprocess_dir = "/data0/xingsy/mel/processed"

# train
dataloader_workers = 3
shuffle_train_data = True
batch_size = 64
seed = 0
triplet_margin = 0.4
learning_rate = 1e-3
num_epoch = 40
test_epoch_interval = 10
metrics_topk = [1, 5, 10, 20, 50]

# debug
debug = False

if debug:
    shuffle_train_data = False
    num_epoch = test_epoch_interval = 1
    dataloader_workers = 0
