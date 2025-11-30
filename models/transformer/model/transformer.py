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
        
        # SỬA: Tạo trực tiếp trên device, tránh copy từ CPU -> GPU
        trg_sub_mask = torch.tril(
            torch.ones((trg_len, trg_len), device=self.device)
        ).bool() # Dùng bool nhẹ hơn ByteTensor
        
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask
    
    def predict(self, src: torch.Tensor) -> torch.Tensor:
        if src.shape[1] > self.max_len:
            src = src[:, :self.max_len]
            
        src_mask = self.make_src_mask(src)
        enc_src = self.encoder(src, src_mask)
        
        batch_size = src.size(0)
        trg_tokens = torch.full((batch_size, 1), 
                                self.trg_bos_idx, 
                                dtype=torch.long, 
                                device=self.device)
        
        # Tạo mask để theo dõi câu nào đã xong (gặp EOS)
        finished_sentences = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        for _ in range(self.max_len):
            trg_mask = self.make_trg_mask(trg_tokens)
            
            # Optimization: Sử dụng mixed precision (AMP) nếu GPU hỗ trợ để nhanh hơn
            with torch.cuda.amp.autocast(enabled=True): 
                output = self.decoder(trg_tokens, enc_src, trg_mask, src_mask)
                next_token_logits = output[:, -1, :]
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            
            # Cập nhật trạng thái các câu đã xong
            # Nếu câu đã xong trước đó, token tiếp theo vẫn giữ nguyên (hoặc padding),
            # nhưng logic này giúp ta biết khi nào break sớm.
            current_eos = (next_token == self.trg_eos_idx).squeeze()
            finished_sentences = finished_sentences | current_eos
            
            # Chỉ nối token mới vào. 
            # (Lưu ý: Logic đơn giản này vẫn nối token sau EOS, ta sẽ cắt bỏ sau)
            trg_tokens = torch.cat([trg_tokens, next_token], dim=1)
            
            # Điều kiện dừng: Khi TẤT CẢ các câu đều đã từng gặp EOS
            if finished_sentences.all():
                break

        return trg_tokens[:, 1:]
