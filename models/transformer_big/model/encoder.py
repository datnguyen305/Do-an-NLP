"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
from torch import nn
from vocabs.vocab import Vocab
from ..blocks.encoder_layer import EncoderLayer
from ..embedding.transformer_embedding import TransformerEmbedding


class Encoder(nn.Module):

    def __init__(self, config, vocab: Vocab):
        super().__init__()
        self.max_len = vocab.max_sentence_length + 2
        self.emb = TransformerEmbedding(config, vocab)

        self.layers = nn.ModuleList([EncoderLayer(config, vocab)
                                     for _ in range(config.n_layers)])

    def forward(self, x, src_mask):
        x = self.emb(x)

        for layer in self.layers:
            x = layer(x, src_mask)

        return x