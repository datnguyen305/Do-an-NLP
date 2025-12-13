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


class DecoderLayer(nn.Module):

    def __init__(self, config, vocab: Vocab):
        super(DecoderLayer, self).__init__()
        self.config = config
        self.self_attention = MultiHeadAttention(config, vocab)
        self.norm1 = LayerNorm(config, vocab)
        self.dropout1 = nn.Dropout(p=config.drop_prob)

        self.enc_dec_attention = MultiHeadAttention(config, vocab)
        self.norm2 = LayerNorm(config, vocab)
        self.dropout2 = nn.Dropout(p=config.drop_prob)
        self.ffn = PositionwiseFeedForward(config, vocab)
        self.norm3 = LayerNorm(config, vocab)
        self.dropout3 = nn.Dropout(p=config.drop_prob)

    def forward(self, dec, enc, trg_mask, src_mask):
        # 1. compute self attention
        _x = dec
        x = self.self_attention(q=dec, k=dec, v=dec, mask=trg_mask)
        
        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        if enc is not None:
            # 3. compute encoder - decoder attention
            _x = x
            x = self.enc_dec_attention(q=x, k=enc, v=enc, mask=src_mask)
            
            # 4. add and norm
            x = self.dropout2(x)
            x = self.norm2(x + _x)

        # 5. positionwise feed forward network
        _x = x
        x = self.ffn(x)
        
        # 6. add and norm
        x = self.dropout3(x)
        x = self.norm3(x + _x)
        return x