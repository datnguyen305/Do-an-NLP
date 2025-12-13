"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
from torch import nn
from vocabs.vocab import Vocab

class PositionwiseFeedForward(nn.Module):

    def __init__(self, config, vocab: Vocab, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(config.d_model, config.ffn_hidden)
        self.linear2 = nn.Linear(config.ffn_hidden, config.d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x