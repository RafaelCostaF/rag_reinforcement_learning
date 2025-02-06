import gym
import numpy as np
from gym import spaces
from sentence_transformers import SentenceTransformer, util

class RAGImprovementEnv(gym.Env):
    def __init__(self, text_chunks, embedding_model_name="all-MiniLM-L6-v2"):
        super(RAGImprovementEnv, self).__init__()

        # Initialize documents and model
        self.documents = text_chunks  # List of text chunks
        self.model = SentenceTransformer(embedding_model_name)

        # Process documents into embeddings for fast similarity comparison
        self.document_embeddings = np.array([self.model.encode(chunk) for chunk in self.documents])
        
        # Get embedding dimensions
        self.embedding_dim = self.document_embeddings.shape[1]
        self.num_documents = len(self.documents)

        # Observation space: embeddings, query, similarities, and retrieval state
        self.observation_space = spaces.Dict({
            "document_embeddings": spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.num_documents, self.embedding_dim), dtype=np.float32
            ),  
            "query_embedding": spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.embedding_dim,), dtype=np.float32
            ),
            "similarities": spaces.Box(
                low=0.0, high=1.0, shape=(self.num_documents,), dtype=np.float32
            ),
            "retrieved_chunks_mask": spaces.MultiBinary(self.num_documents)  # Binary vector: 1 if selected, 0 otherwise
        })

        # Action space: Selecting chunks (indices) + stop action
        self.action_space = spaces.Discrete(self.num_documents + 1)

        # Placeholder for query and retrieval state
        self.query_embedding = None
        self.retrieved_chunks_mask = None

    def reset(self, query):
        """Reset environment with a new query."""
        self.step_count = 0
        self.query_embedding = self.model.encode(query)

        # Compute cosine similarity scores
        cosine_similarities = np.array([
            util.pytorch_cos_sim(self.query_embedding, doc_emb).item() for doc_emb in self.document_embeddings
        ])

        # Retrieval mask (no chunks selected at start)
        self.retrieved_chunks_mask = np.zeros(self.num_documents, dtype=np.int8)

        return {
            "document_embeddings": self.document_embeddings,
            "query_embedding": self.query_embedding,
            "similarities": cosine_similarities,
            "retrieved_chunks_mask": self.retrieved_chunks_mask
        }

    def step(self, action):
        """Take an action (select a document chunk or stop)."""
        self.step_count += 1

        if action == 0:  # Stop action
            done = True
            reward = self.compute_final_reward()
        else:
            chunk_idx = action - 1  # Adjust index (1-based to 0-based)
            if self.retrieved_chunks_mask[chunk_idx] == 1:
                reward = -1.0  # Penalize redundant selection
            else:
                self.retrieved_chunks_mask[chunk_idx] = 1
                reward = self.compute_reward(chunk_idx)

            done = False

        return {
            "document_embeddings": self.document_embeddings,
            "query_embedding": self.query_embedding,
            "similarities": self.similarities,
            "retrieved_chunks_mask": self.retrieved_chunks_mask
        }, reward, done, {}

    def compute_reward(self, action):
        """Reward based on relevance (cosine similarity) and step efficiency."""
        return self.similarities[action] * (10 - (self.step_count * 2))  

    def compute_final_reward(self):
        """Encourage stopping at the right time."""
        return 10.0  

    def render(self, mode="human"):
        print(f"Query Embedding: {self.query_embedding}")
        print(f"Similarities: {self.similarities}")
        print(f"Retrieved Chunks: {self.retrieved_chunks_mask}")
