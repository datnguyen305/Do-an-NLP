"""
@author : Hyunwoong
@when : 2019-10-24
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn
from vocabs.vocab import Vocab
from ..layers.layer_norm import LayerNorm
from ..layers.multi_head_attention import MultiHeadAttention
from ..layers.position_wise_feed_forward import PositionwiseFeedForward


class DecoderLayer(nn.Module):

    def __init__(self, config, vocab: Vocab):
        super(DecoderLayer, self).__init__()
        self.config = config
        
        # Đổi tên để khớp với cách bạn đã gọi self.self_attention
        self.self_attention = MultiHeadAttention(config, vocab) 
        self.norm1 = LayerNorm(config, vocab)
        self.dropout1 = nn.Dropout(p=config.drop_prob)

        # Đổi tên để khớp với cách bạn đã gọi self.enc_dec_attention
        self.enc_dec_attention = MultiHeadAttention(config, vocab) 
        self.norm2 = LayerNorm(config, vocab)
        self.dropout2 = nn.Dropout(p=config.drop_prob)
        self.ffn = PositionwiseFeedForward(config, vocab)
        self.norm3 = LayerNorm(config, vocab)
        self.dropout3 = nn.Dropout(p=config.drop_prob)

    # ----------------------------------------------------
    # PHƯƠNG THỨC HỖ TRỢ CACHING: Khởi tạo Cache Encoder-Decoder
    # ----------------------------------------------------
    # QUAN TRỌNG: Hàm này phải nằm trong phạm vi lớp và gọi self.enc_dec_attention
    def init_encoder_decoder_cache(self, enc_src, src_mask):
        # Giả định MultiHeadAttention có hàm init_cache(K, V, mask)
        # Hàm này sẽ tính và trả về K, V của Encoder.
        return self.enc_dec_attention.init_cache(enc_src, enc_src, src_mask) 
        # Cần đảm bảo hàm init_cache trong MultiHeadAttention nhận K và V.
        # Ở đây K=V=enc_src
    
    # ----------------------------------------------------
    # PHƯƠNG THỨC FORWARD (Đã được sửa để nhận Cache)
    # ----------------------------------------------------
    def forward(self, dec, enc, trg_mask, src_mask, enc_dec_cache=None, self_attn_cache=None):
        # dec (decoder input) hiện tại chỉ là token mới nhất [1, 1, d_model]

        # 1. compute self attention (Sử dụng và cập nhật cache)
        _x = dec
        # Giả định self_attention nhận cache qua tham số 'cache'
        x, new_self_attn_cache = self.self_attention(
            q=dec, 
            k=dec, 
            v=dec, 
            mask=trg_mask,
            cache=self_attn_cache # <-- Truyền cache cũ vào
        ) 
        
        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        if enc is not None: # enc_src luôn có trong inference
            # 3. compute encoder - decoder attention (Sử dụng cache)
            _x = x
            # Chỉ cần truyền K, V của Encoder thông qua tham số 'cache'
            x, _ = self.enc_dec_attention(
                q=x, 
                k=enc, # K và V truyền vào chỉ là giữ chỗ, chúng ta dùng cache
                v=enc, 
                mask=src_mask,
                cache=enc_dec_cache # <-- Truyền cache Encoder đã tính toán trước
            ) 
            
            # 4. add and norm
            x = self.dropout2(x)
            x = self.norm2(x + _x)

        # 5. positionwise feed forward network
        _x = x
        x = self.ffn(x)
        
        # 6. add and norm
        x = self.dropout3(x)
        x = self.norm3(x + _x)
        
        # QUAN TRỌNG: Trả về đầu ra VÀ cache Self-Attention mới
        return x, new_self_attn_cache