from __future__ import annotations
from rag_reranker.ingestion.loader import Document

def _count_words(text: str) -> int:
    # Since counting a token requires loading a tokenizer to convert words first, then count which adds 200+ millisecond.
    # Hence calculating approx length of words which can be fed to AI Model. 1 token = ~0.75 words so 512 tokens = ~384 words.
    return len(text.split())

class FixedSizeChunker:
    # Initializing instance attributes
    def __init__(self, chunk_size: int = 512, overlap: int = 64):
        # Converting token targets to word targets
        self.words_per_chunk = int(chunk_size * 0.75)
        self.overlap_words = int(overlap * 0.75)
        
    # Methods for splitting the full document into smaller chunks which fits the model's input size/context window 
    def split(self, doc: Document) -> list[Document]:
        words = doc.content.split()
        
        # After splitting the document if length of words is less than words per chunk then return doc and exit as no splitting required
        # Else pass for chunking
        if len(words) <= self.words_per_chunk:
            return [doc]
        
        # Creating steps for iterating to all the splitted chunks in a document with overlap
        step = self.words_per_chunk - self.overlap_words
        chunks = []
        
        # Iterating through each chunks with overlap and saving the list inside the variable
        for start in range(0, len(words),step):
            chunk_words = words[start : start + self.words_per_chunk]
            # checks if chunk_words is empty, it breaks the loop and no empty list is added
            if not chunk_words:
                break
            
            chunks.append(
                Document(
                    title=doc.title,
                    content=" ".join(chunk_words),
                    topic=doc.topic,
                    source=doc.source,
                )
            )
            # Checks if the chunks passed the last word to avoid redundant chunks
            if start + self.words_per_chunk >= len(words):
                break
            
        return chunks
    
class RecursiveChunker:
    # \n\n - Paragraph break, . - sentence end, ! - exclaimation, ? - question mark, " " - word boundary
    # Ordered for splitting from start to end
    SEPARATORS = ["\n\n", ". ", "! ", "? ", " "]
    
    # Instance attributes as max characters and overlap chunks for creating small chunks for the documents
    def __init__(self, chunk_size: int = 512, overlap: int = 64):
        # 1 token ~ 4 characters for english text
        self.max_chars = chunk_size * 4
        self.overlap_chars = overlap * 4
        
    # Recursively splitting the text priortizing to retain the semantic meaning  
    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        
        # If text fits in one chunk - return text, else pass
        if len(text) <= self.max_chars:
            return[text]
        
        # If no seperators given, must hard-cut
        if not separators:
            return[
                text[i : i + self.max_chars]
                for i in range(0,len(text),self.max_chars)
            ]
        
        # Recursively using each separators from ordered list to retain the semantic meaning
        # Prioritizing paragraph break first, then sentence end, then exclaimation marks, then question marks, and lastly word boundry
        separator = separators[0]
        remaining_separators = separators[1:]
        
        # Using first separator to split the text and stripping the blank spaces from start and end
        parts = text.split(separator)
        parts = [p for p in parts if p.strip()]

        # Created empty variables to split chunks and save
        # current_buffer will try to add characters till less than max_chars.
        # Once current_buffer reaches <= max_chars constraint and no other character could be appended, it safely saves to merged_chunks.
        # After above action, it reset itself for new chunk
        merged_chunks: list[str] = []
        current_buffer =""
        
        for part in parts:
            candidate = current_buffer + separator + part if current_buffer else part
            
            if len(candidate) <= self.max_chars:
                current_buffer = candidate
            else:
                if current_buffer:
                    merged_chunks.append(current_buffer)
                    
            if len(part) > self.max_chars:
                sub_chunks = self._split_recursive(part, remaining_separators)
                merged_chunks.extend(sub_chunks)
                current_buffer = ""
            else: 
                current_buffer = part
                
        if current_buffer:
            merged_chunks.append(current_buffer)
            
        return merged_chunks
    
    def _add_overlap(self, chunks: list[str]) -> list[str]:
        """"
        Prepend the tail of the previous chunk to the current chunk.
        Overlap gives the context for understanding.
        """
        
        if self.overlap_chars <= 0 or len(chunks) <=1:
            return chunks
        
        overlapped = [chunks[0]]
        
        for i in range(1, len(chunks)):
            previous_tail = chunks[i-1][-self.overlap_chars : ]
            overlapped.append(previous_tail + " " + chunks[i])
            
        return overlapped

    def split(self, doc: Document) -> list[Document]:
        if len(doc.content) <= self.max_chars:
            return [doc]
        
        raw_chunks = self._split_recursive(doc.content, self.SEPARATORS)
        raw_chunks = self._add_overlap(raw_chunks)
        
        return[
            Document(
                title=doc.title,
                content=chunk.strip(),
                topic=doc.topic,
                source=doc.source,
                )
            for chunk in raw_chunks
            if chunk.strip()
        ]
        
def chunk_documents(docs: list[Document], strategy: str = "recursive", chunk_size: int = 512, overlap: int = 64) -> list[Document]:
    if strategy == "fixed":
        chunker = FixedSizeChunker(chunk_size, overlap)
    else:
        chunker = RecursiveChunker(chunk_size, overlap)
        
    all_chunks: list[Document] = []
    for doc in docs:
        chunks = chunker.split(doc)
        all_chunks.extend(chunks)
        
    return all_chunks