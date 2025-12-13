"""
@author : Hyunwoong
@when : 2019-10-24
@homepage : https://github.com/gusdnd852
"""
from torch import nn
from vocabs.vocab import Vocab

class TokenEmbedding(nn.Embedding):
    """
    Token Embedding using torch.nn
    they will dense representation of word using weighted matrix
    """

    def __init__(self, config, vocab: Vocab):
        """
        class for token embedding that included positional information

        :param vocab_size: size of vocabulary
        :param d_model: dimensions of model
        """
        self.max_len = vocab.max_sentence_length + 2
        super(TokenEmbedding, self).__init__(self.max_len, config.d_model, padding_idx=vocab.pad_idx)