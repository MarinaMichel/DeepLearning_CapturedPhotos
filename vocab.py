"""
Shared Vocabulary class.
Used identically by train.py (to build the vocab) and by the backend
(to decode model predictions back into words), so it lives in one place
and is pickled/unpickled rather than duplicated.
"""
from collections import Counter


class Vocabulary:
    def __init__(self, freq_threshold=1):
        self.word2idx = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.freq_threshold = freq_threshold
        self.word_counts = Counter()

    def build_vocabulary(self, sentence_list):
        for sent in sentence_list:
            self.word_counts.update(sent.lower().split())

        for word, count in self.word_counts.items():
            if count >= self.freq_threshold:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def numericalize(self, text):
        tokens = text.lower().split()
        return [self.word2idx.get(t, self.word2idx["<unk>"]) for t in tokens]

    def __len__(self):
        return len(self.word2idx)
