from typing import Dict, List
import os
import sys
class CROHMEVocab:
    # print(sys.path)
    # print(sys.path[0])
    PAD_IDX = 0
    NO_RELATION = 0
    INNER = 1
    # SOS_IDX = 1
    # EOS_IDX = 2
    def __init__(
        self,
        dict_sym: str = sys.path[0] + '/vocab/dictionary.txt',
        dict_rel: str = sys.path[0] + '/vocab/relation.txt'
        # dict_rel: str = './vocab/relation.txt'
        ) -> None:
        self.word2idx = dict()
        # self.word2idx["<pad>"] = self.PAD_IDX
        # self.word2idx["<sos>"] = self.SOS_IDX
        # self.word2idx["<eos>"] = self.EOS_IDX
        
        self.rel2idx = dict()
        self.rel2idx["NO_RELATION"] = self.NO_RELATION
        self.rel2idx["INNER"] = self.INNER


        with open(dict_sym, "r") as f:
            for line in f.readlines():
                w = line.strip()
                self.word2idx[w] = len(self.word2idx)

        with open(dict_rel, "r") as f:
            for line in f.readlines():
                w = line.strip()
                self.rel2idx[w] = len(self.rel2idx)

        self.idx2word: Dict[int, str] = {v: k for k, v in self.word2idx.items()}

        self.idx2rel: Dict[int, str] = {v: k for k, v in self.rel2idx.items()}

        # print(f"Init vocab with size: {len(self.word2idx)}")

    def words2indices(self, words: List[str]) -> List[int]:
        return [self.word2idx[w] for w in words]

    def indices2words(self, id_list: List[int]) -> List[str]:
        return [self.idx2word[i] for i in id_list]

    def indices2label(self, id_list: List[int]) -> str:
        words = self.indices2words(id_list)
        return " ".join(words)

    def rel2indices(self, relation):
        return [self.rel2idx[r] for r in relation]

    def indices2rel(self, id_list):
        return [self.idx2rel[i] for i in id_list]

    def __len__(self):
        return len(self.word2idx)


vocab = CROHMEVocab()