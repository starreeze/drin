# bert
max_bert_len = 512
bert_embed_dim = 768
CLS = 101
SEP = 102

# model
finetune_bert = False
pre_extract_mention = False
mention_linear_after_avg = True
entity_linears_after_avg = True
linear_output_dim = 512

# other data
num_entity_sentence = 3
num_candidates = 101

# path
qid2entity_answer_path = "/home/data_91_c/xsy/mel-dataset/wikimel/candidates/qid2ne.json"
preprocess_dir = "/home/data_91_c/xsy/mel-dataset/text_preprocessed/"

# train
dataloader_workers = 4
batch_size = 32
seed = 0
triplet_margin = 0.25
learning_rate = 1e-3
epoch = 5
