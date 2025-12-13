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
        self.max_len = self.vocab.max_sentence_length + 2
        max_len = self.max_len


        src_mask = self.make_src_mask(src)
        enc_src = self.encoder(src, src_mask)

        trg_indexes = [self.vocab.bos_idx]

        for i in range(max_len):
            trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(self.device)

            trg_mask = self.make_trg_mask(trg_tensor)

            output = self.decoder(trg_tensor, enc_src, trg_mask, src_mask)

            pred_token = output.argmax(2)[:, -1].item()

            trg_indexes.append(pred_token)

            if pred_token == self.vocab.eos_idx:
                break
        final_trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(self.device)      
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