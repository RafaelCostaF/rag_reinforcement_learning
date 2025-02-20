import gymnasium as gym
import numpy as np
from gymnasium import spaces
from sentence_transformers import SentenceTransformer, util

class RAGImprovementEnv(gym.Env):
    def __init__(self, embedding_model_name="all-MiniLM-L6-v2"):
        super(RAGImprovementEnv, self).__init__()

        self.model = SentenceTransformer(embedding_model_name)
        
        self.embedding_dim = self.model.get_sentence_embedding_dimension()  # Get actual dim
        self.max_num_documents = 300  

        self.observation_space = spaces.Dict({
            "document_embeddings": spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_num_documents, self.embedding_dim), dtype=np.float32),  
            "query_embedding": spaces.Box(low=-np.inf, high=np.inf, shape=(self.embedding_dim,), dtype=np.float32),  
            "similarities": spaces.Box(low=0.0, high=1.0, shape=(self.max_num_documents,), dtype=np.float32),  
            "retrieved_chunks_mask": spaces.MultiBinary(self.max_num_documents)  
        })

        
        self.action_space = spaces.Discrete(self.max_num_documents + 1)
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        text_chunks = options.get("text_chunks", [])  # Default to empty list if None
        query = options.get("query", "")

        self.documents = text_chunks
        self.num_documents = len(self.documents)

        # Check if text_chunks is empty
        if self.num_documents == 0:
            raise ValueError("text_chunks is empty. The environment requires at least one document.")

        # Initialize document embeddings with zero padding
        self.document_embeddings = np.zeros((self.max_num_documents, self.embedding_dim), dtype=np.float32)
        
        computed_embeddings = np.array([self.model.encode(chunk) for chunk in self.documents])

        if computed_embeddings.size == 0:  # Avoid shape mismatch
            raise ValueError("Computed embeddings are empty. Check if SentenceTransformer is working properly.")

        self.document_embeddings[:self.num_documents, :computed_embeddings.shape[1]] = computed_embeddings

        self.query_embedding = self.model.encode(query)

        # Compute similarities and pad if necessary
        self.similarities = np.zeros(self.max_num_documents, dtype=np.float32)
        computed_similarities = np.array([
            util.pytorch_cos_sim(self.query_embedding, doc_emb).item() for doc_emb in computed_embeddings
        ])
        self.similarities[:self.num_documents] = computed_similarities

        self.retrieved_chunks_mask = np.zeros(self.max_num_documents, dtype=np.int8)
        self.step_count = 0

        return {
            "document_embeddings": self.document_embeddings,
            "query_embedding": self.query_embedding,
            "similarities": self.similarities,
            "retrieved_chunks_mask": self.retrieved_chunks_mask
        }, {}

    def step(self, action):
        """Take an action (select a document chunk or stop)."""
        self.step_count += 1
        done = False
        reward = 0.0

        if action == 0:  # Stop action
            done = True
            reward = self.compute_final_reward()
        else:
            chunk_idx = action - 1
            if self.retrieved_chunks_mask[chunk_idx] == 1:
                reward = -1.0  # Penalize redundant selection
            else:
                self.retrieved_chunks_mask[chunk_idx] = 1
                reward = self.compute_reward(chunk_idx)

        return (
            {
                "document_embeddings": self.document_embeddings,
                "query_embedding": self.query_embedding,
                "similarities": self.similarities,
                "retrieved_chunks_mask": self.retrieved_chunks_mask
            },
            reward,
            done,
            False,  # No early termination
            {}      # ✅ Return an empty dictionary instead of ''
        )


    def compute_reward(self, action):
        """Reward based on relevance (cosine similarity) and step efficiency."""
        return self.similarities[action] * (10 - (self.step_count * 2))

    def compute_final_reward(self):
        """Encourage stopping at the right time."""
        return 10.0

    def render(self):
        print(f"Query Embedding: {self.query_embedding}")
        print(f"Similarities: {self.similarities}")
        print(f"Retrieved Chunks: {self.retrieved_chunks_mask}")
