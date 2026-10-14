"""Byte Pair Encoding reversible (UTF-8) con entrenamiento independiente."""
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import json


def merge_tokens(tokens, pair, replacement):
    out=[];i=0
    while i<len(tokens):
        if i+1<len(tokens) and (tokens[i],tokens[i+1])==pair:
            out.append(replacement);i+=2
        else:
            out.append(tokens[i]);i+=1
    return out


@dataclass
class ByteBPETokenizer:
    merges:list[tuple[int,int]]

    @classmethod
    def train(cls,text,vocab_size=512,min_frequency=2):
        if not 256<=vocab_size<=65535:raise ValueError('vocab_size debe estar entre 256 y 65535')
        sequences=[list(line.encode('utf-8')) for line in text.splitlines(keepends=True) if line]
        merges=[]
        for new_id in range(256,vocab_size):
            freq=Counter(pair for seq in sequences for pair in zip(seq,seq[1:]))
            if not freq:break
            pair,n=min(freq.items(),key=lambda x:(-x[1],x[0]))
            if n<min_frequency:break
            sequences=[merge_tokens(seq,pair,new_id) for seq in sequences]
            merges.append(pair)
        return cls(merges)

    @property
    def vocab_size(self):return 256+len(self.merges)

    def encode(self,text):
        ids=list(text.encode('utf-8'))
        for idx,pair in enumerate(self.merges,256):
            ids=merge_tokens(ids,pair,idx)
        return ids

    def decode_bytes(self,ids):
        lookup=[bytes([i]) for i in range(256)]
        for a,b in self.merges:lookup.append(lookup[a]+lookup[b])
        return b''.join(lookup[i] for i in ids)

    def decode(self,ids):return self.decode_bytes(ids).decode('utf-8',errors='replace')

    def to_dict(self):
        return {'type':'byte_bpe_v1','merges':[list(p) for p in self.merges],'vocab_size':self.vocab_size}

    @classmethod
    def from_dict(cls,obj):
        if obj.get('type')!='byte_bpe_v1':raise ValueError('Tokenizer incompatible')
        merges=[]
        for i,pair in enumerate(obj['merges'],256):
            if len(pair)!=2 or any(type(n)is not int or not 0<=n<i for n in pair):
                raise ValueError('Merge inválida')
            merges.append(tuple(pair))
        if obj.get('vocab_size')!=256+len(merges):raise ValueError('Vocabulario inconsistente')
        return cls(merges)

    def save(self,path):
        Path(path).write_text(json.dumps(self.to_dict(),ensure_ascii=False),encoding='utf-8')

    @classmethod
    def load(cls,path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding='utf-8')))
