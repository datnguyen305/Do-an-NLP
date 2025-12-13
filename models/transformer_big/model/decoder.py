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
        self.emb = TransformerEmbedding(config, vocab)
        self.layers = nn.ModuleList([DecoderLayer(config, vocab)
                                     for _ in range(config.n_layers)])

        self.linear = nn.Linear(config.d_model, vocab.vocab_size)

    def forward(self, trg, enc_src, trg_mask, src_mask, enc_dec_cache, self_attn_cache):
        trg = self.emb(trg)
        new_self_attn_cache = []
        for i, layer in enumerate(self.layers):
        # Truyền các tham số cache vào DecoderLayer
            trg, new_layer_cache = layer(
                trg, 
                enc_src, 
                trg_mask, 
                src_mask,
                enc_dec_cache[i],        # Cache Encoder-Decoder
                self_attn_cache[i]       # Cache Self-Attention cũ
            )
        new_self_attn_cache.append(new_layer_cache)

        # pass to LM head
        output = self.linear(trg)
        return output, new_self_attn_cache

    def init_encoder_decoder_cache(self, enc_src, src_mask):
        cache = []
        for layer in self.layers: # Giả định self.layers là danh sách các DecoderLayer
            # Giả định DecoderLayer có hàm init_encoder_decoder_cache
            cache.append(layer.init_encoder_decoder_cache(enc_src, src_mask))
        return cache