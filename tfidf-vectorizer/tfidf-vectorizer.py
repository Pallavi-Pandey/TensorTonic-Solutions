import numpy as np
from collections import Counter
import math

def tfidf_vectorizer(documents):
    tokenized = [doc.split() for doc in documents]

    vocab = sorted(set(word for doc in tokenized for word in doc))
    vocab_index = {w: i for i, w in enumerate(vocab)}

    N = len(documents)

    # Document frequency
    df = {w: 0 for w in vocab}
    for doc in tokenized:
        for w in set(doc):
            df[w] += 1

    # IDF
    idf = {w: math.log(N / df[w]) for w in vocab}

    tfidf = np.zeros((N, len(vocab)))

    for d, doc in enumerate(tokenized):
        counts = Counter(doc)
        doc_len = len(doc)

        for w, count in counts.items():
            j = vocab_index[w]
            tf = count / doc_len          # 🔥 normalized TF
            tfidf[d, j] = tf * idf[w]

    return tfidf, vocab