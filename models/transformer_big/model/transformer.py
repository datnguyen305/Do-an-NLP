"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn
from vocabs.vocab import Vocab
from .decoder import Decoder
from .encoder import Encoder
from builders.model_builder import META_ARCHITECTURE

@META_ARCHITECTURE.register()
class TransformerModel(nn.Module):
    def __init__(self, config, vocab: Vocab):
        super().__init__()
        self.src_pad_idx = vocab.pad_idx
        self.trg_pad_idx = vocab.pad_idx
        self.trg_sos_idx = vocab.bos_idx
        self.device = config.device
        self.vocab = vocab
        self.max_len = vocab.max_sentence_length + 2
        self.encoder = Encoder(config, vocab)
        self.decoder = Decoder(config, vocab)
        self.loss = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
        self.d_model = config.d_model

    def forward(self, src, trg):
        
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        output = self.decoder(trg, enc_src, trg_mask, src_mask)
        loss = self.loss(output.reshape(-1, self.vocab.vocab_size), trg.reshape(-1))
        return output, loss
    
    def predict(self, src):
        """
        Thực hiện dự đoán (suy luận) bằng Greedy Decoding với kỹ thuật Key-Value Caching
        để tăng tốc độ.
        """
        # Đảm bảo batch size là 1
        if src.size(0) != 1:
             raise ValueError("Hàm predict hiện tại chỉ hỗ trợ batch_size = 1")

        # 1. KHỞI TẠO ENCODER VÀ CACHE ENCODER-DECODER (TÍNH TOÁN 1 LẦN)
        src_mask = self.make_src_mask(src)
        enc_src = self.encoder(src, src_mask) 
        
        # Giả định self.decoder có hàm init_encoder_decoder_cache để tính K, V của Encoder
        encoder_decoder_cache = self.decoder.init_encoder_decoder_cache(enc_src, src_mask)

        # 2. KHỞI TẠO CACHE DECODER SELF-ATTENTION (CACHE NÀY SẼ ĐƯỢC CẬP NHẬT)
        # Giả định self.decoder có thuộc tính n_layers
        decoder_self_attn_cache = [None] * self.decoder.n_layers 
        
        # 3. VÒNG LẶP DECODING
        max_len = self.max_len 
        trg_indexes = [self.trg_sos_idx] 

        for i in range(max_len):
            
            # ĐẦU VÀO DECODER CHỈ LÀ TOKEN MỚI NHẤT: [1, 1]
            current_token_idx = trg_indexes[-1] 
            
            trg_tensor = torch.tensor([[current_token_idx]], dtype=torch.long, device=self.device)

            # Tạo trg_mask cho token hiện tại [1, 1, 1, 1]
            # Mặc dù seq_len = 1, nhưng ta vẫn tạo mask để đảm bảo tính nhất quán
            trg_mask = self.make_trg_mask(trg_tensor) 

            # THỰC HIỆN DECODING VÀ NHẬN LẠI CACHE MỚI
            # output: [1, 1, vocab_size]
            # *LƯU Ý: HÀM DECODER PHẢI ĐƯỢC SỬA ĐỔI ĐỂ NHẬN 4 THAM SỐ CUỐI CÙNG NÀY
            output, decoder_self_attn_cache = self.decoder(
                trg_tensor, 
                enc_src, 
                trg_mask, 
                src_mask, 
                encoder_decoder_cache,       # Cache Encoder-Decoder (Không thay đổi)
                decoder_self_attn_cache      # Cache Self-Attention (Cũ)
            )

            # Lấy token dự đoán
            pred_token = output.argmax(2).item() 

            trg_indexes.append(pred_token)

            # KIỂM TRA ĐIỀU KIỆN DỪNG
            if pred_token == self.trg_eos_idx:
                break
                
        # 4. KẾT THÚC
        final_trg_tensor = torch.tensor(trg_indexes, dtype=torch.long, device=self.device).unsqueeze(0)
        
        return final_trg_tensor

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.ByteTensor).to(self.device)
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask