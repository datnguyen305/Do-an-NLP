"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn
from vocabs.vocab import Vocab
from ..blocks.decoder_layer import DecoderLayer
from ..embedding.transformer_embedding import TransformerEmbedding


class Decoder(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super().__init__()
        self.max_len = vocab.max_sentence_length + 2
        self.emb = TransformerEmbedding(d_model=config.d_model,
                                        drop_prob=config.drop_prob,
                                        max_len=self.max_len,
                                        vocab_size=vocab.vocab_size,
                                        device=config.device)
        self.layers = nn.ModuleList([DecoderLayer(d_model=config.d_model,
                                                  ffn_hidden=config.ffn_hidden,
                                                  n_head=config.n_head,
                                                  drop_prob=config.drop_prob)
                                     for _ in range(config.n_layers)])

        self.linear = nn.Linear(config.d_model, vocab.vocab_size)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        trg = self.emb(trg)

        for layer in self.layers:
            trg = layer(trg, enc_src, trg_mask, src_mask)

        # pass to LM head
        output = self.linear(trg)
        return output