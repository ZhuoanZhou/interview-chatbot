# Python standard library imports
import os
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any

# Third-party imports
import faiss
import numpy as np

from src.content.memory_bank.memory_bank_base import MemoryBankBase
from src.content.memory_bank.memory import Memory, MemorySearchResult
from src.content.embeddings.embedding_service import EmbeddingService

# Load environment variables

class VectorMemoryBank(MemoryBankBase):
    """Vector database implementation of memory bank using FAISS and configurable embeddings.

    Supports multiple embedding backends via EMBEDDING_BACKEND env var:
    - openai: OpenAI embeddings (default)
    - vllm: Local vLLM server embeddings
    - noop: Disabled embeddings (no API calls)
    """

    def __init__(self, embedding_service: Optional[EmbeddingService] = None):
        super().__init__()
        self.embedding_service = embedding_service or EmbeddingService()
        self.embedding_dimension = self.embedding_service.get_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.embedding_dimension)
        self.embeddings: Dict[str, np.ndarray] = {}

    def _get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for the given text using configured backend."""
        return self.embedding_service.get_embedding(text)

    def add_memory(
        self,
        title: str,
        text: str,
        subtopic_links: List[Dict[str, Any]],
        source_interview_question: str,
        source_interview_response: str,
        metadata: Dict = None,
        question_ids: List[str] = None
    ) -> Memory:
        """Add a new memory to the vector database."""
        if metadata is None:
            metadata = {}
        if question_ids is None:
            question_ids = []
            
        memory_id = self.generate_memory_id()
        combined_text = f"{title}\n{text}"
        embedding = self._get_embedding(combined_text)
        
        memory = Memory(
            id=memory_id,
            title=title,
            text=text,
            subtopic_links=subtopic_links,
            metadata=metadata,
            timestamp=datetime.now(),
            source_interview_question=source_interview_question,
            source_interview_response=source_interview_response,
            question_ids=question_ids
        )
        
        self.memories.append(memory)
        self.embeddings[memory_id] = embedding
        self.index.add(embedding.reshape(1, -1))

        return memory

    def search_memories(self, query: str, k: int = 5) -> List[MemorySearchResult]:
        """Search for similar memories using the query text."""
        if not self.memories:
            return []
        
        query_embedding = self._get_embedding(query)
        
        # Adjust k to not exceed the number of available memories
        k = min(k, len(self.memories))
        
        # Perform similarity search
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1),
            k
        )
        
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx < len(self.memories):
                memory = self.memories[idx]
                similarity_score = float(1 / (1 + distance))
                results.append(MemorySearchResult.from_memory(
                    memory=memory,
                    similarity_score=similarity_score
                ))
        
        return results

    def _save_implementation_specific(self, path: str) -> None:
        """Save embeddings to file.
        
        Args:
            path: Path to save embeddings (either user_id or session path)
        """
        embedding_data = {
            'embeddings': [
                {'id': memory_id, 'embedding': embedding.tolist()}
                for memory_id, embedding in self.embeddings.items()
            ]
        }
        
        embedding_filepath = os.getenv("LOGS_DIR") + \
            f"/{path}/memory_bank_embeddings.json"
        os.makedirs(os.path.dirname(embedding_filepath), exist_ok=True)
        
        with open(embedding_filepath, 'w') as f:
            json.dump(embedding_data, f)

    def _load_implementation_specific(self, user_id: str, base_path: Optional[str] = None) -> None:
        """Load embeddings from file and reconstruct the FAISS index."""
        # Determine embedding filepath based on base_path
        if base_path:
            embedding_filepath = os.path.join(base_path, "memory_bank_embeddings.json")
        else:
            embedding_filepath = os.getenv("LOGS_DIR") + f"/{user_id}/memory_bank_embeddings.json"
        
        try:
            with open(embedding_filepath, 'r') as f:
                embedding_data = json.load(f)
                
            # Create embedding lookup dictionary
            self.embeddings = {
                e['id']: np.array(e['embedding'], dtype=np.float32)
                for e in embedding_data['embeddings']
            }
            
            # Reconstruct FAISS index
            for memory in self.memories:
                embedding = self.embeddings.get(memory.id)
                if embedding is not None:
                    self.index.add(embedding.reshape(1, -1))
                    
        except FileNotFoundError:
            pass  # No embeddings file exists yet
