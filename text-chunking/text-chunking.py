def text_chunking(tokens, chunk_size, overlap):
    """
    Split tokens into fixed-size chunks with optional overlap.
    """
    if chunk_size <= 0:
        return []

    step = chunk_size - overlap
    if step <= 0:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []

    for start in range(0, len(tokens), step):
        chunk = tokens[start:start + chunk_size]
        if not chunk:
            break
        chunks.append(chunk)

        if start + chunk_size >= len(tokens):
            break

    return chunks