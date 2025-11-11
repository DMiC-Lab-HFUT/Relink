from typing import List
import tiktoken
import re
class TextSplitter:
    def __init__(self, model_name: str = 'gpt-3.5-turbo'):
        self.encoding = tiktoken.encoding_for_model(model_name)

    def split_text(self, text: str,
                   overlap: int = 16,
                   max_chunk_size: int = 128,
                   min_chunk_size: int = 100,
                   padding: str = " ...") -> List[str]:
        tokens = self.encoding.encode(text)

        step_size = max_chunk_size - overlap
        pos = 0
        chunks = []

        while pos < len(tokens):
            end_pos = pos + max_chunk_size

            if end_pos >= len(tokens):
                chunk = tokens[pos:len(tokens)]
                if len(chunk) < min_chunk_size and chunks:
                    chunks[-1].extend(chunk)
                else:
                    chunks.append(chunk)
                break
            else:
                chunk = tokens[pos:end_pos]
                chunks.append(chunk)
                pos += step_size

        texts = [self.encoding.decode(chunk) for chunk in chunks]

        padded_texts = []
        num_chunks = len(texts)

        if num_chunks <= 1:
            return texts

        for i, chunk_text in enumerate(texts):
            if i == 0:
                padded_chunk = chunk_text + padding
            elif i == num_chunks - 1:
                padded_chunk = padding + chunk_text
            else:
                padded_chunk = padding + chunk_text + padding
            padded_texts.append(padded_chunk)
        return padded_texts



class SemanticTextSplitter:
    def __init__(self, model_name='gpt-3.5-turbo'):
        self.encoding = tiktoken.encoding_for_model(model_name)

    def split_text(self, text: str, max_tokens=128, overlap=16):
        sentences = re.split(r'(?<=[.!?。！？])\s+', text)
        chunks, current_chunk = [], []
        current_length = 0

        for sent in sentences:
            sent_len = len(self.encoding.encode(sent))
            if current_length + sent_len > max_tokens:
                chunks.append(" ".join(current_chunk))
                overlap_tokens = self.encoding.encode(" ".join(current_chunk))[-overlap:]
                overlap_text = self.encoding.decode(overlap_tokens)
                current_chunk = [overlap_text, sent]
                current_length = len(self.encoding.encode(overlap_text + sent))
            else:
                current_chunk.append(sent)
                current_length += sent_len

        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks
