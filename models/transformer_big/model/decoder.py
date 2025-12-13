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
        self.n_layers = config.n_layers
        self.layers = nn.ModuleList([DecoderLayer(config, vocab)
                                     for _ in range(config.n_layers)])

        self.linear = nn.Linear(config.d_model, vocab.vocab_size)

    def forward(self, trg, enc_src, trg_mask, src_mask, enc_dec_cache=None, self_attn_cache=None):
        trg = self.emb(trg)
        
        # Nếu đang ở chế độ Training (không có cache), khởi tạo new_self_attn_cache là None.
        # Nếu ở chế độ Inference (có cache), khởi tạo new_self_attn_cache là list
        new_self_attn_cache = [] if self_attn_cache is not None else None 
        
        for i, layer in enumerate(self.layers):
            
            # --- Xử lý tham số cache truyền cho DecoderLayer ---
            layer_enc_dec_cache = enc_dec_cache[i] if enc_dec_cache is not None else None
            layer_self_attn_cache = self_attn_cache[i] if self_attn_cache is not None else None
            
            # Giả định DecoderLayer đã được sửa để trả về (output, new_cache)
            output_from_layer, new_layer_cache = layer(
                trg, 
                enc_src, 
                trg_mask, 
                src_mask,
                layer_enc_dec_cache,    
                layer_self_attn_cache   
            )
            trg = output_from_layer # Cập nhật đầu vào cho lớp tiếp theo
            
            # Nếu đang ở chế độ Inference, lưu cache mới
            if new_self_attn_cache is not None:
                new_self_attn_cache.append(new_layer_cache)

        # pass to LM head
        output = self.linear(trg)
        
        # Ở chế độ Training, trả về output và None cho cache
        return output, new_self_attn_cache

    def init_encoder_decoder_cache(self, enc_src, src_mask):
        cache = []
        for layer in self.layers: # Giả định self.layers là danh sách các DecoderLayer
            # Giả định DecoderLayer có hàm init_encoder_decoder_cache
            cache.append(layer.init_encoder_decoder_cache(enc_src, src_mask))
        return cache