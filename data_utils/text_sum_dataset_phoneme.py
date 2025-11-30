from torch.utils.data import Dataset
import json
from typing import List

from builders.dataset_builder import META_DATASET
from utils.instance import Instance
from vocabs.vocab import Vocab
from vocabs.utils import preprocess_sentence # Bắt buộc import hàm này

@META_DATASET.register()
class TextSumDatasetPhoneme(Dataset):
    def __init__(self, config, vocab: Vocab) -> None:
        super().__init__()

        path: str = config.path
        self._data = json.load(open(path, encoding='utf-8'))
        self._keys = list(self._data.keys())
        self._vocab = vocab

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> Instance:
        key = self._keys[index]
        item = self._data[key]
        
        # --- 1. XỬ LÝ SOURCE ---
        paragraphs = item["source"]
        # Join các câu trong đoạn, sau đó join các đoạn bằng <nl>
        paragraphs = [" ".join(paragraph) for _, paragraph in paragraphs.items()]
        source_str = " <nl> ".join(paragraphs) 
        
        # QUAN TRỌNG: Tách từ trước khi encode (vì ViWordVocab xử lý theo list từ)
        source_tokens: List[str] = preprocess_sentence(source_str)
        
        # Encode: [Src_Len, 4]
        encoded_source = self._vocab.encode_caption(source_tokens)

        # --- 2. XỬ LÝ TARGET ---
        target_str = item["target"]
        target_tokens: List[str] = preprocess_sentence(target_str)

        # Encode: [Tgt_Len, 4] (Có chứa BOS và EOS)
        encoded_target = self._vocab.encode_caption(target_tokens)

        # --- 3. TẠO SHIFTED RIGHT LABEL ---
        # Cắt bỏ token đầu tiên (BOS) -> [Tgt_Len - 1, 4]
        # Đây chính là ground truth để tính loss
        shifted_right_label = encoded_target[1:]
       
        return Instance(
            id = key,
            input_ids = encoded_source,
            label = encoded_target,         # Full target (Có BOS, có EOS)
            shifted_right_label = shifted_right_label # Target (Mất BOS, có EOS)
        )