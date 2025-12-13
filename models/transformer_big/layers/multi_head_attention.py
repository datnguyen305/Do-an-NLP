"""
@author : Hyunwoong
@when : 2019-10-25
@homepage : https://github.com/gusdnd852
"""
from torch import nn
import torch
from vocabs.vocab import Vocab
from .scale_dot_product_attention import ScaleDotProductAttention


class MultiHeadAttention(nn.Module):

    def __init__(self, config, vocab: Vocab):
        super(MultiHeadAttention, self).__init__()
        self.n_head = config.n_head
        self.d_model = config.d_model
        self.d_k = self.d_model // self.n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(config.d_model, config.d_model)
        self.w_k = nn.Linear(config.d_model, config.d_model)
        self.w_v = nn.Linear(config.d_model, config.d_model)
        self.w_concat = nn.Linear(config.d_model, config.d_model)
    def init_cache(self, k_in, v_in, mask):
        """
        Tính toán Key và Value của Encoder/Source để lưu cache. 
        Phương thức này chỉ được gọi MỘT LẦN duy nhất.
        """
        # 1. Linear projections
        K = self.w_k(k_in)
        V = self.w_v(v_in)
        
        # 2. Split heads và transpose để có dạng [B, H, Lk, Dk]
        K = self.split(K)
        V = self.split(V)
        
        # Trả về cache dưới dạng tuple (K_cache, V_cache, mask_cache)
        return K, V, mask
    
    def forward(self, q, k, v, mask=None, cache=None):
        
        # 1. dot product with weight matrices
        # Q luôn được tính toán từ đầu vào mới
        Q = self.w_q(q)
        
        # K, V TẠM THỜI (chỉ là K, V của token mới nhất)
        K_new = self.w_k(k)
        V_new = self.w_v(v)

        # 2. split tensor by number of heads
        Q = self.split(Q)
        K_new = self.split(K_new)
        V_new = self.split(V_new)
        
        # Khởi tạo cache mới (chỉ dùng cho Self-Attention)
        new_cache = None 
        
        # Xử lý Caching
        if cache is not None:
            K_cache, V_cache, mask_cache = cache
            
            # --- KIỂM TRA LOẠI ATTENTION VÀ XỬ LÝ CACHE ---
            
            # 1. ENCODER-DECODER ATTENTION (Kích thước K_cache LỚN hơn K_new: dùng cache)
            if K_cache.size(2) > K_new.size(2): 
                # Đây là Encoder-Decoder Attention: K, V của Encoder đã được cache
                K_final = K_cache
                V_final = V_cache
                # Sử dụng mask của Encoder đã được tính toán
                mask = mask_cache 
                
            # 2. SELF-ATTENTION (Kích thước K_cache BẰNG K_new: nối cache)
            else: 
                # Đây là Self-Attention trong quá trình suy luận: Nối token mới nhất với cache cũ
                K_final = torch.cat([K_cache, K_new], dim=2) # Nối trên chiều Seq Length (dim=2)
                V_final = torch.cat([V_cache, V_new], dim=2)
                
                # Cập nhật cache mới để trả về cho bước tiếp theo
                new_cache = (K_final, V_final, mask) # mask cũng thay đổi theo chiều dài chuỗi
        
        else:
            # Không có cache (Quá trình training hoặc Self-Attention bước đầu tiên)
            K_final = K_new
            V_final = V_new
            
            # Trong Self-Attention bước 0, ta cần khởi tạo cache để bắt đầu
            if q.size(1) == 1 and q.size(0) == 1: # Rất có thể là bước đầu tiên của inference
                 new_cache = (K_final, V_final, mask)


        # 3. do scale dot product to compute similarity
        # Q: [B, H, 1, Dk] (Token mới nhất)
        # K_final/V_final: [B, H, Lk_final, Dk]
        out, attention = self.attention(Q, K_final, V_final, mask=mask)

        # 4. concat and pass to linear layer
        out = self.concat(out)
        out = self.w_concat(out)

        # 5. visualize attention map (giữ nguyên)
        
        # QUAN TRỌNG: Trả về output VÀ cache mới đã được cập nhật
        return out, new_cache

    def split(self, tensor):
        """
        split tensor by number of head

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()
        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        return tensor

    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor
        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor