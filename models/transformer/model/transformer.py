import torch
from torch import nn

from models.transformer.model.decoder import Decoder
from models.transformer.model.encoder import Encoder

from vocabs.vocab import Vocab
from builders.model_builder import META_ARCHITECTURE


@META_ARCHITECTURE.register()
class TransformerModel(nn.Module):

    def __init__(self, config, vocab: Vocab):
        super().__init__()
        self.src_pad_idx = vocab.pad_idx
        self.trg_pad_idx = vocab.pad_idx
        self.trg_bos_idx = vocab.bos_idx
        self.trg_eos_idx = vocab.eos_idx
        
        self.encoder = Encoder(config.encoder, vocab)

        self.decoder = Decoder(config.decoder, vocab)

        self.d_model = config.d_model
        self.device = config.device 
        self.config = config
        self.vocab = vocab
        self.MAX_LEN = vocab.max_sentence_length + 2 

        self.loss = nn.CrossEntropyLoss(ignore_index=self.trg_pad_idx)


    def forward(self, src, trg):
        config = self.config
        
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        output = self.decoder(trg, enc_src, trg_mask, src_mask)

         # Tính loss
        output_flat = output.contiguous().view(-1, output.size(-1))  # [B*T, Vocab]
        trg_flat = trg.contiguous().view(-1)                         # [B*T]

        loss = self.loss(output_flat, trg_flat)


        return output, loss

    def make_src_mask(self, src):
        # max_seq_len = src.shape[1]  # Lấy độ dài thực tế
        # src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # src_mask = src_mask[:, :, :max_seq_len, :max_seq_len]  # Cắt mask phù hợp
        # return src_mask

        """src: [B, src_len]
        return: [B,1,1,src_len] bool mask (True = not pad)"""
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask


    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)
        trg_len = trg.shape[1]
        # trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.ByteTensor).to(self.device)
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len, device=self.device)).bool()
        trg_mask = trg_pad_mask & trg_sub_mask # [B,1,T,T]
        return trg_mask
    
    def predict(self, src: torch.Tensor) -> torch.Tensor:
        config = self.config

        # 1. Tạo mask cho chuỗi nguồn và mã hóa
        src_mask = self.make_src_mask(src)
        enc_src = self.encoder(src, src_mask)   # [B, src_len, d_model]
        
        # 2. Chuẩn bị đầu vào ban đầu cho bộ giải mã là token BOS
        batch_size = src.size(0)
        # Sử dụng device của src để đảm bảo tính nhất quán
        decoder_input = torch.full((batch_size, 1), self.trg_bos_idx, dtype=torch.long, device=src.device)
        
        # outputs dùng để lưu kết quả dự đoán (trừ BOS)
        
        # 3. Tạo dự đoán từng bước (Autoregressive Decoding)
        for _ in range(self.MAX_LEN):
            
            # Tạo mask cho chuỗi đích (Trg mask sẽ tự động mở rộng theo trg_len)
            trg_mask = self.make_trg_mask(decoder_input)
            
            # Lấy đầu ra của bộ giải mã
            # decoder_output: [B, current_trg_len, vocab_size]
            decoder_output = self.decoder(decoder_input, enc_src, trg_mask, src_mask)
            
            # Lấy từ mới nhất (từ cuối cùng của chuỗi)
            # next_token_logits: [B, vocab_size]
            next_token_logits = decoder_output[:, -1, :] 
            
            # Chọn token có xác suất cao nhất (Greedy Decoding)
            # next_token: [B, 1]
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)

            # --- KIỂM TRA ĐIỀU KIỆN DỪNG ---
            # Nếu tất cả các chuỗi trong batch đã dự đoán ra EOS, hoặc nếu độ dài vượt quá max_len
            
            # Dùng nó làm input cho bước tiếp theo
            decoder_input = torch.cat([decoder_input, next_token], dim=1)
            
            # Kiểm tra xem token EOS đã được dự đoán ở tất cả các chuỗi chưa
            # Ta chỉ cần kiểm tra token mới nhất được thêm vào
            if (next_token == self.trg_eos_idx).all():
                break

        # 4. Loại bỏ token BOS ban đầu
        # Outputs: [B, trg_len] (chứa BOS, các token dự đoán, và có thể là EOS)
        # Ta cần loại bỏ BOS token ở vị trí đầu tiên
        final_outputs = decoder_input[:, 1:]
        
        return final_outputs
