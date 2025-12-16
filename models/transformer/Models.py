''' Define the Transformer model '''
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .Layers import EncoderLayer, DecoderLayer
from vocabs.vocab import Vocab
from builders.model_builder import META_ARCHITECTURE

__author__ = "Yu-Hsiang Huang"


def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)
   # Shape (B, 1, Seq_len)


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask

# subsequent_mask:
# 1,0,0,0,0,0,0,0,0,0,
# 1,1,0,0,0,0,0,0,0,0,
# 1,1,1,0,0,0,0,0,0,0,
# 1,1,1,1,0,0,0,0,0,0,
# 1,1,1,1,1,0,0,0,0,0,
# 1,1,1,1,1,1,0,0,0,0,
# 1,1,1,1,1,1,1,0,0,0,
# 1,1,1,1,1,1,1,1,0,0,
# 1,1,1,1,1,1,1,1,1,0,
# 1,1,1,1,1,1,1,1,1,1
# Các vị trí được cho là 1 thì mô hình sẽ chú ý đến, bao gồm các từ đã dự đoán và các từ trước đó

class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]
            # return a list of d_hid dimension [0.2134123, 0.5435345, ....] # (d_hid,)

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)]) # (n_position, d_hid)
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, n_src_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, dropout=0.1, n_position=200, scale_emb=False):

        super().__init__()

        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, src_seq, src_mask, return_attns=False):

        enc_slf_attn_list = []

        # -- Forward
        enc_output = self.src_word_emb(src_seq)
        if self.scale_emb:
            enc_output *= self.d_model ** 0.5
        enc_output = self.dropout(self.position_enc(enc_output))
        enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self, n_trg_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, n_position=200, dropout=0.1, scale_emb=False):

        super().__init__()

        self.trg_word_emb = nn.Embedding(n_trg_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model
        

    def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Forward
        dec_output = self.trg_word_emb(trg_seq)
        if self.scale_emb:
            dec_output *= self.d_model ** 0.5
        dec_output = self.dropout(self.position_enc(dec_output))
        dec_output = self.layer_norm(dec_output)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,

@META_ARCHITECTURE.register()
class TransformerModel(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self, config, vocab, src_pad_idx = 0, trg_pad_idx=0,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=4096,
            trg_emb_prj_weight_sharing=True, emb_src_trg_weight_sharing=True,
            scale_emb_or_prj='prj'):

        super().__init__()

        self.src_pad_idx, self.trg_pad_idx = src_pad_idx, trg_pad_idx
        self.n_trg_vocab = vocab.vocab_size
        self.n_src_vocab = vocab.vocab_size
        # In section 3.4 of paper "Attention Is All You Need", there is such detail:
        # "In our model, we share the same weight matrix between the two
        # embedding layers and the pre-softmax linear transformation...
        # In the embedding layers, we multiply those weights by \sqrt{d_model}".
        #
        # Options here:
        #   'emb': multiply \sqrt{d_model} to embedding output
        #   'prj': multiply (\sqrt{d_model} ^ -1) to linear projection output
        #   'none': no multiplication
        self.vocab = vocab
        self.n_src_vocab =  vocab.vocab_size
        self.n_src_vocab = vocab.vocab_size
        assert scale_emb_or_prj in ['emb', 'prj', 'none']
        scale_emb = (scale_emb_or_prj == 'emb') if trg_emb_prj_weight_sharing else False
        self.scale_prj = (scale_emb_or_prj == 'prj') if trg_emb_prj_weight_sharing else False
        self.d_model = d_model
        self.loss = nn.CrossEntropyLoss()
        self.encoder = Encoder(
            n_src_vocab=self.n_src_vocab, n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=src_pad_idx, dropout=dropout, scale_emb=scale_emb)

        self.decoder = Decoder(
            n_trg_vocab=self.n_trg_vocab, n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=trg_pad_idx, dropout=dropout, scale_emb=scale_emb)

        self.trg_word_prj = nn.Linear(d_model, self.n_trg_vocab, bias=False)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

        if trg_emb_prj_weight_sharing:
            # Share the weight between target word embedding & last dense layer
            self.trg_word_prj.weight = self.decoder.trg_word_emb.weight

        if emb_src_trg_weight_sharing:
            self.encoder.src_word_emb.weight = self.decoder.trg_word_emb.weight


    def forward(self, src_seq, trg_seq):

        src_mask = get_pad_mask(src_seq, self.src_pad_idx)
        trg_mask = get_pad_mask(trg_seq, self.trg_pad_idx) & get_subsequent_mask(trg_seq)

        enc_output, *_ = self.encoder(src_seq, src_mask)
        dec_output, *_ = self.decoder(trg_seq, trg_mask, enc_output, src_mask)
        seq_logit = self.trg_word_prj(dec_output)
        # seq_logit: [batch_size, trg_len, n_trg_vocab]

        if self.scale_prj:
            seq_logit *= self.d_model ** -0.5
        
        loss = self.loss(
            seq_logit.contiguous().view(-1, seq_logit.size(-1)),
            trg_seq.contiguous().view(-1)
        )

        return _, loss # Flatten for the loss function
    
    def _model_decode(self, trg_seq, enc_output, src_mask):
        trg_mask = get_subsequent_mask(trg_seq)
        dec_output, *_ = self.decoder(trg_seq, trg_mask, enc_output, src_mask)
        return self.trg_word_prj(dec_output)
    
    def predict(self, src_seq):
        ''' 
        Dự đoán đầu ra cho chuỗi đầu vào src_seq (chỉ hỗ trợ batch size = 1) 
        sử dụng Greedy Decoding.
        '''
        # Chỉ chấp nhận batch size bằng 1
        assert src_seq.size(0) == 1

        src_pad_idx, trg_eos_idx = self.vocab.pad_idx, self.vocab.eos_idx 
        max_seq_len = self.vocab.max_sentence_length + 2  # +2 for BOS and EOS
        device = src_seq.device
        
        with torch.no_grad():
            # 1. Mã hóa chuỗi nguồn (Encoder)
            src_mask = get_pad_mask(src_seq, src_pad_idx)
            # enc_output có shape (1, L_src, D)
            enc_output, *_ = self.encoder(src_seq, src_mask)
            B = src_seq.size(0)
            # Khởi tạo chuỗi đích với token <BOS> (B=1, L=1)
            # gen_seq có shape (1, current_len)
            gen_seq = torch.empty(B, 1, dtype=torch.long, device=enc_output.device).fill_(self.vocab.bos_idx)
            outputs = []

            # 2. Vòng lặp giải mã Tham Lam (Decoder)
            for step in range(max_seq_len):
                # dec_output_probs có shape (1, current_len, vocab_size)
                dec_output_probs = self._model_decode(gen_seq, enc_output, src_mask)
                # dec_output_probs: [B, current_len, vocab_size]
            
                next_word_probs = dec_output_probs[:, -1, :] 
                
                # Chọn token có xác suất cao nhất (Tham Lam)
                # next_word_id có shape (1, 1)
                _, next_word_id = torch.max(next_word_probs, dim=-1)
                outputs.append(next_word_id)
                # Lấy giá trị token (dạng scalar)
                next_word_id = next_word_id.squeeze().item()
                
                # 3. Cập nhật chuỗi và Kiểm tra điều kiện dừng
                
                # Nối token mới vào chuỗi
                next_token = torch.LongTensor([[next_word_id]]).to(device)
                gen_seq = torch.cat([gen_seq, next_token], dim=1)

                # Kiểm tra token kết thúc chuỗi
                if next_word_id == trg_eos_idx:
                    break
            
            outputs = torch.cat(outputs, dim=1)
            return outputs
        