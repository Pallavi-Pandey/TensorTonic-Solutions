import numpy as np
from collections import Counter
import math

def bm25_score(query_tokens, docs, k1=1.2, b=0.75):
    """
    Returns numpy array of BM25 scores for each document.
    docs are already tokenized (list of token lists).
    """
    tokenized_docs = docs
    N = len(tokenized_docs)

    doc_lens = [len(doc) for doc in tokenized_docs]
    avgdl = sum(doc_lens) / N if N > 0 else 0

    # Document frequency
    df = Counter()
    for doc in tokenized_docs:
        for term in set(doc):
            df[term] += 1

    # IDF (BM25)
    idf = {}
    for term in df:
        idf[term] = math.log((N - df[term] + 0.5) / (df[term] + 0.5) + 1)

    scores = np.zeros(N)

    for i, doc in enumerate(tokenized_docs):
        freq = Counter(doc)

        for t in query_tokens:
            if t not in freq:
                continue

            f = freq[t]
            numerator = f * (k1 + 1)
            denominator = f + k1 * (1 - b + b * doc_lens[i] / avgdl)

            scores[i] += idf.get(t, 0) * numerator / denominator

    return scores