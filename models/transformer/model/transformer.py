from logging import config
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
        self.max_len = config.max_len
        self.loss = nn.CrossEntropyLoss(ignore_index=self.trg_pad_idx)


    def forward(self, src, trg):
        # Cắt src nếu quá dài
        if src.shape[1] > self.max_len:
            src = src[:, :self.max_len]

        # Cắt trg nếu quá dài
        if trg.shape[1] > self.max_len:
            trg = trg[:, :self.max_len]

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
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.ByteTensor).to(self.device)
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask
    
    def predict(self, src: torch.Tensor) -> torch.Tensor:
        """
        Thực hiện Greedy Decoding để dự đoán chuỗi đích.

        Args:
            src (torch.Tensor): Chuỗi nguồn, shape [batch_size, src_len] (chứa token IDs).
            
        Returns:
            torch.Tensor: Chuỗi đích dự đoán, shape [batch_size, max_decoded_len] (không bao gồm token SOS).
        """
        # Cắt src nếu quá dài
        if src.shape[1] > self.max_len:
            src = src[:, :self.max_len]

        # Cắt trg nếu quá dài
        if trg.shape[1] > self.max_len:
            trg = trg[:, :self.max_len]
        # 1. Mã hóa chuỗi nguồn một lần
        src_mask = self.make_src_mask(src)
        enc_src = self.encoder(src, src_mask)   # [B, src_len, d_model]
        
        # 2. Khởi tạo đầu vào giải mã bằng token SOS
        batch_size = src.size(0)
        # Khởi tạo chuỗi đích là [B, 1] với token SOS
        trg_tokens = torch.full((batch_size, 1), 
                                self.trg_bos_idx, 
                                dtype=torch.long, 
                                device=self.device)
        
        # 3. Vòng lặp giải mã từng bước
        for _ in range(self.max_len):
            
            # Tạo mask cho chuỗi đích hiện tại
            trg_mask = self.make_trg_mask(trg_tokens)
            
            # Giải mã
            # output: [B, current_trg_len, dec_voc_size]
            output = self.decoder(trg_tokens, enc_src, trg_mask, src_mask)
            
            # Lấy logits của token tiếp theo (token cuối cùng của chuỗi)
            # next_token_logits: [B, dec_voc_size]
            next_token_logits = output[:, -1, :] 
            
            # Chọn token có xác suất cao nhất (Greedy)
            # next_token: [B, 1]
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            
            # Nối token mới vào chuỗi đích
            # trg_tokens: [B, current_trg_len + 1]
            trg_tokens = torch.cat([trg_tokens, next_token], dim=1)
            
            # Kiểm tra điều kiện dừng: nếu tất cả các chuỗi trong batch đã dự đoán EOS
            if (next_token == self.trg_eos_idx).all():
                break

        # 4. Trả về kết quả (loại bỏ token SOS ban đầu)
        # 
        return trg_tokens[:, 1:]
