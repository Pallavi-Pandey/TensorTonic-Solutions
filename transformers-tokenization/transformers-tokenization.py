import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        # Add special tokens first
        special_tokens = [
            self.pad_token,
            self.unk_token,
            self.bos_token,
            self.eos_token
        ]
        
        for token in special_tokens:
            idx = self.vocab_size
            self.word_to_id[token] = idx
            self.id_to_word[idx] = token
            self.vocab_size += 1
        
        # Add unique words from texts
        for text in texts:
            for word in text.split():
                if word not in self.word_to_id:
                    idx = self.vocab_size
                    self.word_to_id[word] = idx
                    self.id_to_word[idx] = word
                    self.vocab_size += 1
    
    def encode(self, text: str) -> List[int]:
        ids = []
        for word in text.split():
            if word in self.word_to_id:
                ids.append(self.word_to_id[word])
            else:
                ids.append(self.word_to_id[self.unk_token])
        return ids
    
    def decode(self, ids: List[int]) -> str:
        words = []
        for i in ids:
            if i in self.id_to_word:
                words.append(self.id_to_word[i])
            else:
                words.append(self.unk_token)
        return " ".join(words)