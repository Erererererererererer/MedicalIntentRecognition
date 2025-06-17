import torch
from torch import nn


class Attention(nn.Module):
    def __init__(self, d_model):
        super(Attention, self).__init__()
        self.w_qk = nn.Linear(d_model, d_model)
        self.dot_product = nn.Linear(d_model, 1, bias=False)  # 模拟QK点乘

    def forward(self, x):  # x: [batch_size, seq_len, d_emb]
        qk = self.w_qk(x)  # [batch_size, seq_len, d_emb]
        similarity = self.dot_product(qk).squeeze(2)  # [batch_size, seq_len, 1] -> [batch_size, seq_len]
        probability = torch.softmax(similarity, dim=1)  # [batch_size, seq_len]
        output = torch.sum(probability.unsqueeze(2) * x, dim=1)  # [batch_size, seq_len, d_emb] -> [batch_size, d_emb]

        return output  # output: [batch_size, d_emb]


class AttentionLSTM(nn.Module):
    def __init__(self,
                 d_emb,  # embedding维度
                 d_hidden,  # LSTM隐藏层维度
                 d_output,  # 输出维度
                 vocab,  # 词汇表大小
                 ):
        super(AttentionLSTM, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab, embedding_dim=d_emb)
        self.lstm = nn.LSTM(input_size=d_emb, hidden_size=d_hidden, num_layers=2, batch_first=True, bidirectional=True)
        self.attention = Attention(2 * d_hidden)
        self.mlp = nn.Sequential(
            nn.Linear(2 * d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_output)
        )

    def forward(self, x, mask):  # x: [batch_size, seq_len]
        emb = self.embedding(x)  # [batch_size, seq_len, d_emb]
        output, hidden = self.lstm(emb)  # [batch_size, seq_len, 32]
        output = self.attention(output)  # [batch_size, 32]
        output = self.mlp(output)  # [batch_size, d_output]

        return output  # output: [batch_size, d_output]
