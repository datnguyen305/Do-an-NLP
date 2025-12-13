"""
@author : Hyunwoong
@when : 2019-10-24
@homepage : https://github.com/gusdnd852
"""
from torch import nn
from vocabs.vocab import Vocab
from ..layers.layer_norm import LayerNorm
from ..layers.multi_head_attention import MultiHeadAttention
from ..layers.position_wise_feed_forward import PositionwiseFeedForward


class EncoderLayer(nn.Module):

    def __init__(self, config, vocab: Vocab):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=config.d_model, n_head=config.n_head)
        self.norm1 = LayerNorm(d_model=config.d_model)
        self.dropout1 = nn.Dropout(p=config.drop_prob)
        self.ffn = PositionwiseFeedForward(d_model=config.d_model, hidden=config.ffn_hidden, drop_prob=config.drop_prob)
        self.norm2 = LayerNorm(d_model=config.d_model)
        self.dropout2 = nn.Dropout(p=config.drop_prob)

    def forward(self, x, src_mask):
        # 1. compute self attention
        _x = x
        x = self.attention(q=x, k=x, v=x, mask=src_mask)
        
        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)
        
        # 3. positionwise feed forward network
        _x = x
        x = self.ffn(x)
      
        # 4. add and norm
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x