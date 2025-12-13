"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
from torch import nn
from vocabs.vocab import Vocab
from models.blocks.encoder_layer import EncoderLayer
from models.embedding.transformer_embedding import TransformerEmbedding


class Encoder(nn.Module):

    def __init__(self, config, vocab: Vocab):
        super().__init__()
        self.max_len = vocab.max_sentence_length + 2
        self.emb = TransformerEmbedding(d_model=config.d_model,
                                        max_len=self.max_len,
                                        vocab_size=vocab.vocab_size,
                                        drop_prob=config.drop_prob,
                                        device=config.device)

        self.layers = nn.ModuleList([EncoderLayer(d_model=config.d_model,
                                                  ffn_hidden=config.ffn_hidden,
                                                  n_head=config.n_head,
                                                  drop_prob=config.drop_prob)
                                     for _ in range(config.n_layers)])

    def forward(self, x, src_mask):
        x = self.emb(x)

        for layer in self.layers:
            x = layer(x, src_mask)

        return x