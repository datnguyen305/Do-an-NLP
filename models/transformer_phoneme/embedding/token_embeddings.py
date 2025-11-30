import torch
from torch import nn

class PhonemeEmbedding(nn.Module):
    """
    Embedding module for phoneme-level input of shape [B, S, 4].
    The 4 dimensions correspond to (onset, medial, nucleus, coda).
    This uses Summation Embedding, where the vectors for each component are summed up.
    """

    def __init__(self, 
                 vocab_size_onset, 
                 vocab_size_medial, 
                 vocab_size_nucleus, 
                 vocab_size_coda, 
                 d_model, 
                 padding_idx=0):
        """
        :param vocab_size_*: Kích thước từ vựng của từng thành phần âm vị.
        :param d_model: Kích thước chiều ẩn của mô hình (Embedding dimension).
        :param padding_idx: ID của token padding (thường là 0).
        """
        super().__init__()
        
        # Tạo 4 lớp nn.Embedding độc lập, tất cả có cùng kích thước d_model
        # Lưu ý: Ta sử dụng padding_idx để các thành phần PAD không đóng góp vào tổng
        self.onset_embed = nn.Embedding(vocab_size_onset, d_model, padding_idx=padding_idx)
        self.medial_embed = nn.Embedding(vocab_size_medial, d_model, padding_idx=padding_idx)
        self.nucleus_embed = nn.Embedding(vocab_size_nucleus, d_model, padding_idx=padding_idx)
        self.coda_embed = nn.Embedding(vocab_size_coda, d_model, padding_idx=padding_idx)
        
        self.d_model = d_model

    def forward(self, x):
        """
        Thực hiện phép nhúng.
        
        :param x: Tensor input có shape [B, S, 4] (với x là các chỉ mục/IDs)
        :return: Tensor output có shape [B, S, d_model]
        """
        # Đảm bảo input là kiểu long
        if x.dtype != torch.long:
             x = x.long()

        # 1. Tách input [B, S, 4] thành 4 tensor [B, S]
        onset_input = x[..., 0] 
        medial_input = x[..., 1]
        nucleus_input = x[..., 2]
        coda_input = x[..., 3]
        
        # 2. Tính nhúng cho từng thành phần (Output: 4 tensors, mỗi tensor [B, S, d_model])
        onset_vec = self.onset_embed(onset_input)
        medial_vec = self.medial_embed(medial_input)
        nucleus_vec = self.nucleus_embed(nucleus_input)
        coda_vec = self.coda_embed(coda_input)
        
        # 3. Cộng tổng các vector nhúng (Summation)
        # Kết quả là một tensor [B, S, d_model]
        final_embedding = onset_vec + medial_vec + nucleus_vec + coda_vec
        
        return final_embedding