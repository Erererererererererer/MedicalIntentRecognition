import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from torch import nn
from transformers import BertModel

from DataPrepare import save_path


pretrain_model = "google-bert/bert-base-chinese"
# pretrain_model = "nghuyong/ernie-3.0-base-zh"
# pretrain_model = "trueto/medbert-base-chinese"


class BERTClassifier(nn.Module):
    def __init__(self, d_output, dropout=0.1):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(pretrain_model, cache_dir=save_path)
        for param in self.bert.parameters():
            param.requires_grad = True  # 每个参数都要求梯度
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(self.bert.config.hidden_size, d_output)

    def forward(self, x, mask):
        bert_output = self.bert(input_ids=x, attention_mask=mask)
        bert_output = bert_output.pooler_output  # 获取 BERT 的输出特征，即[CLS]
        output = self.dropout(bert_output)
        output = self.linear(output)

        return output
