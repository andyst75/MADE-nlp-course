from collections import Counter
from itertools import chain
import numpy as np

class Tfidf:
    
    def __init__(self, norm='l2', alpha:int=0):
        self.words = {}
        self.idf = {}
        self.norm = norm
        self.alpha = alpha
        if norm not in [None, 'l1', 'l2']:
            raise ValueError('"norm" should be either None, "l1" or "l2"')


    def compute_idf_(self, corpus):
        corpus_size = len(corpus)
        document_counts = Counter()
        
        for text in corpus:
            document_counts.update(set(text))
            
        self.idf = {term: np.log(corpus_size / (count + self.alpha)) 
                        for term, count in document_counts.items()}
        
        return self.idf
    
    
    def compute_tfidf_vector_(self, word_counter:Counter):
        vector = np.zeros(len(self.words), dtype=np.float32)
        
        for word, count in word_counter.items():
            if word in self.words:
                vector[self.words[word]] = word_counter[word] * self.idf[word]
        
        if self.norm is None:
            return vector
        elif self.norm == 'l1':
            return vector / (np.linalg.norm(vector, ord=1) + 1e-8)
        else:
            return vector / (np.linalg.norm(vector, ord=2) + 1e-8)
    
    
    def fit(self, corpus):
        tokenized = list(map(str.split, corpus))
        self.words = {w:i for i, w in enumerate(set(chain.from_iterable(tokenized)))}
        self.compute_idf_(tokenized)


    def transform(self, corpus):
        tokenized = list(map(Counter, map(str.split, corpus)))
        return np.stack(list(map(self.compute_tfidf_vector_, tokenized)))

    def fit_transform(self, corpus):
        self.fit(corpus)
        return self.transform(corpus)