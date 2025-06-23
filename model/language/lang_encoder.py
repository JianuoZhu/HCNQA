from transformers import BertConfig, BertModel, BertTokenizer
import torch
from pipeline.registry import registry

LOCAL_MODEL_PATH = '/home/zhengf_lab/cse12210702/.cache/huggingface/transformers/bert-base-uncased'

@registry.register_language_model("bert_tokenizer")
def get_bert_tokenizer():
    tokenizer = BertTokenizer.from_pretrained(
        LOCAL_MODEL_PATH,
        do_lower_case=True)
    return tokenizer

@registry.register_language_model("bert_lang_encoder")
def get_bert_lang_encoder(num_hidden_layer=3):
    txt_bert_config = BertConfig(
        hidden_size=768,
        num_hidden_layers=num_hidden_layer,
        num_attention_heads=12, type_vocab_size=2
    )
    txt_encoder = BertModel.from_pretrained(
        LOCAL_MODEL_PATH, config=txt_bert_config
    )
    return txt_encoder
