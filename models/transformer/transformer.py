import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from builders.model_builder import META_ARCHITECTURE
from vocabs.vocab import Vocab

# Helper: Positional Encoding chuẩn của PyTorch
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0)) # [1, Max_Len, D]

    def forward(self, x):
        # x: [Batch, Seq_Len, D]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class CachedMultiHeadAttention(nn.Module):
    """
    Sử dụng F.scaled_dot_product_attention của PyTorch để thay thế tính toán thủ công.
    Hỗ trợ KV Cache.
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # Các lớp Linear chiếu Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout_p = dropout

    def forward(self, query, key, value, mask=None, cache=None, is_cross_attn=False):
        batch_size = query.size(0)
        
        # 1. Projection: [Batch, Seq, D_model]
        q = self.w_q(query)
        k = self.w_k(key)
        v = self.w_v(value)

        # 2. Reshape cho Multi-head: [Batch, N_Heads, Seq, Head_Dim]
        q = q.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)

        # 3. Xử lý KV Cache (Chỉ dùng cho Self-Attention khi Inference)
        if cache is not None:
            if is_cross_attn:
                # Cross Attention: K, V đến từ Encoder (cố định), không cần nối, chỉ cần lưu lại nếu chưa có
                # Nhưng thực tế Cross Attn thường tính lại K, V từ Encoder state mỗi lần (hoặc project 1 lần rồi lưu)
                # Để đơn giản và đúng logic cache chuẩn:
                if cache[0] is None: 
                     # Lần đầu tiên: lưu lại K, V của Encoder
                    new_cache = (k, v)
                else:
                    # Các lần sau: dùng lại K, V đã lưu
                    k, v = cache
                    new_cache = cache
            else:
                # Self Attention: Nối K, V hiện tại vào quá khứ
                k_past, v_past = cache
                k = torch.cat([k_past, k], dim=2)
                v = torch.cat([v_past, v], dim=2)
                new_cache = (k, v)
        else:
            new_cache = None

        # 4. Tính Attention dùng hàm tối ưu của PyTorch (FlashAttention nếu có GPU xịn)
        # Hàm này thay thế toàn bộ đoạn matmul + scale + softmax + dropout
        output = F.scaled_dot_product_attention(
            query=q, 
            key=k, 
            value=v, 
            attn_mask=mask, 
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False # Mask đã xử lý causal ở ngoài
        )

        # 5. Gộp heads và project output
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.out_proj(output), new_cache

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = CachedMultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = CachedMultiHeadAttention(d_model, n_heads, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Feed Forward Network
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU() # Hiện đại hơn ReLU

    def forward(self, trg, enc_src, trg_mask, src_mask, cache=None):
        # cache input: (self_attn_cache, cross_attn_cache)
        self_cache = cache[0] if cache is not None else None
        cross_cache = cache[1] if cache is not None else None
        
        # 1. Self Attention
        _trg, new_self_cache = self.self_attn(trg, trg, trg, trg_mask, cache=self_cache, is_cross_attn=False)
        trg = self.norm1(trg + _trg) # Residual + Norm
        
        # 2. Cross Attention
        _trg, new_cross_cache = self.cross_attn(trg, enc_src, enc_src, src_mask, cache=cross_cache, is_cross_attn=True)
        trg = self.norm2(trg + _trg)
        
        # 3. Feed Forward
        _trg = self.linear2(self.dropout(self.activation(self.linear1(trg))))
        trg = self.norm3(trg + _trg)
        
        return trg, (new_self_cache, new_cross_cache)

@META_ARCHITECTURE.register()
class TransformerModel(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super().__init__()
        self.config = config
        self.vocab = vocab
        self.device = config.device
        self.d_model = config.d_model
        self.max_len = config.max_len
        
        # Embedding & Positional Encoding
        self.src_embedding = nn.Embedding(vocab.vocab_size, self.d_model, padding_idx=vocab.pad_idx)
        self.trg_embedding = nn.Embedding(vocab.vocab_size, self.d_model, padding_idx=vocab.pad_idx)
        self.pos_encoding = PositionalEncoding(self.d_model, dropout=0.1)
        
        # --- ENCODER: Sử dụng module chuẩn của PyTorch ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=config.encoder.n_heads, 
            dim_feedforward=config.encoder.d_ff, 
            dropout=config.encoder.dropout,
            batch_first=True # Quan trọng: Input shape [Batch, Seq, D]
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.encoder.n_layers)
        
        # --- DECODER: Sử dụng custom layer để hỗ trợ Cache ---
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(
                self.d_model, 
                config.decoder.n_heads, 
                config.decoder.d_ff, 
                config.decoder.dropout
            ) for _ in range(config.decoder.n_layers)
        ])
        
        self.fc_out = nn.Linear(self.d_model, vocab.vocab_size)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)

    def make_src_mask(self, src):
        # PyTorch TransformerEncoder cần mask dạng [Batch, Seq] hoặc [Batch, 1, 1, Seq] cho padding
        # True = keep, False = mask? Tùy hàm. 
        # nn.TransformerEncoderLayer mặc định: True là ko mask, False là mask (nếu dùng key_padding_mask)
        # Nhưng ở đây ta truyền vào attn_mask dạng float (cộng vào attention score)
        mask = (src == self.vocab.pad_idx) # [Batch, Seq]
        return mask # Trả về Boolean Mask (True ở vị trí Pad)

    def make_trg_mask(self, trg):
        # Causal Mask (Che tương lai)
        sz = trg.size(1)
        # Hàm có sẵn của PyTorch tạo mask tam giác
        mask = torch.triu(torch.ones(sz, sz, device=self.device) * float('-inf'), diagonal=1)
        return mask

    def forward(self, src, trg):
        # 1. Prepare Masks
        src_pad_mask = (src == self.vocab.pad_idx) 
        
        # trg mask: Dùng trg gốc (indices) để tạo mask là đúng rồi
        trg_causal_mask = self.make_trg_mask(trg)
        
        # 2. Embedding + Positional
        # SỬA LỖI Ở ĐÂY: Không ghi đè lên biến trg gốc
        src_emb = self.pos_encoding(self.src_embedding(src))
        trg_emb = self.pos_encoding(self.trg_embedding(trg)) # Đổi tên thành trg_emb
        
        # 3. Encoder 
        enc_src = self.encoder(src_emb, src_key_padding_mask=src_pad_mask)
        
        # 4. Decoder
        out = trg_emb # Dùng trg_emb để đưa vào mạng
        
        for layer in self.decoder_layers:
            # Lưu ý: trg_causal_mask vẫn dùng được
            out, _ = layer(out, enc_src, trg_causal_mask, None)
            
        output = self.fc_out(out)
        
        # 5. Tính loss
        output_flat = output.view(-1, output.size(-1))
        
        # SỬA LỖI Ở ĐÂY: Dùng trg gốc (indices) để tính loss
        trg_flat = trg.contiguous().view(-1) 
        
        loss = self.loss_fn(output_flat, trg_flat)
        return output, loss

    def predict(self, src: torch.Tensor):
        """
        Hàm Predict tối ưu với KV Cache và Batch Processing
        """
        batch_size = src.size(0)
        
        # 1. Encode (Chạy 1 lần)
        if src.size(1) > self.max_len: src = src[:, :self.max_len]
        src_pad_mask = (src == self.vocab.pad_idx)
        
        src_emb = self.pos_encoding(self.src_embedding(src))
        enc_src = self.encoder(src_emb, src_key_padding_mask=src_pad_mask)
        
        # 2. Init Decoder
        decoder_input = torch.full((batch_size, 1), self.vocab.bos_idx, dtype=torch.long, device=self.device)
        
        # Khởi tạo Cache: List[Tuple(Self_K, Self_V), Tuple(Cross_K, Cross_V)]
        caches = [None] * len(self.decoder_layers)
        
        finished_sentences = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        generated_tokens = []

        # 3. Loop sinh từ
        for i in range(self.max_len):
            # Embedding cho token hiện tại (chỉ 1 token)
            trg_emb = self.trg_embedding(decoder_input)
            # Positional encoding cho vị trí i (quan trọng vì seq_len = 1)
            pos = torch.arange(i, i + 1, dtype=torch.float, device=self.device).unsqueeze(0) # [1, 1]
            # Tính toán PE thủ công một chút để khớp shape hoặc dùng lại class PositionalEncoding nếu sửa lại
            # Để đơn giản, ta cộng PE theo đúng index
            trg_emb = trg_emb + self.pos_encoding.pe[:, i:i+1, :]
            
            out = trg_emb
            new_caches = []
            
            # Forward qua các lớp Decoder với Cache
            for layer_idx, layer in enumerate(self.decoder_layers):
                # trg_mask là None vì ta chỉ đưa vào 1 token, không cần mask tương lai
                out, layer_cache = layer(out, enc_src, trg_mask=None, src_mask=None, cache=caches[layer_idx])
                new_caches.append(layer_cache)
            
            caches = new_caches # Cập nhật cache cho vòng lặp sau
            
            # Prediction
            logits = self.fc_out(out[:, -1, :]) # [Batch, Vocab]
            next_token = logits.argmax(dim=-1, keepdim=True) # [Batch, 1]
            
            generated_tokens.append(next_token)
            
            # Check EOS
            current_eos = (next_token == self.vocab.eos_idx).squeeze()
            finished_sentences = finished_sentences | current_eos
            if finished_sentences.all():
                break
                
            decoder_input = next_token

        return torch.cat(generated_tokens, dim=1)