import gymnasium as gym
import numpy as np
from gymnasium import spaces
from sentence_transformers import SentenceTransformer, util

class RAGImprovementEnv(gym.Env):
    def __init__(self, embedding_model_name="all-MiniLM-L6-v2"):
        super(RAGImprovementEnv, self).__init__()

        self.model = SentenceTransformer(embedding_model_name)

        # Placeholder attributes (set during reset)
        self.documents = []
        self.document_embeddings = None
        self.embedding_dim = 1  # Temporary value to be updated later
        self.num_documents = 1  # Temporary value

        # Define a default observation space (to be updated in reset)
        self.observation_space = spaces.Dict({
            "document_embeddings": spaces.Box(low=-np.inf, high=np.inf, shape=(1, 1), dtype=np.float32),
            "query_embedding": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            "similarities": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "retrieved_chunks_mask": spaces.MultiBinary(1)
        })
        self.action_space = spaces.Discrete(2)  # At least 1 document + 1 stop action

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        text_chunks = options["text_chunks"]
        query = options["query"]
        
        self.documents = text_chunks
        self.num_documents = len(self.documents)
        self.document_embeddings = np.array([self.model.encode(chunk) for chunk in self.documents])
        self.embedding_dim = self.document_embeddings.shape[1]

        # Update observation and action spaces dynamically
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
            "retrieved_chunks_mask": spaces.MultiBinary(self.num_documents)
        })
        self.action_space = spaces.Discrete(self.num_documents + 1)

        # Compute query embedding and similarities
        self.query_embedding = self.model.encode(query)
        # self.similarities = np.array([
        #     util.pytorch_cos_sim(self.query_embedding, doc_emb).item() for doc_emb in self.document_embeddings
        # ])
        
        self.similarities = np.array([
            # util.pytorch_cos_sim(self.query_embedding, doc_emb) for doc_emb in self.document_embeddings
            util.pytorch_cos_sim(self.query_embedding, doc_emb).squeeze().tolist() for doc_emb in self.document_embeddings
        ])

        # Reset retrieval state
        self.retrieved_chunks_mask = np.zeros(self.num_documents, dtype=np.int8)
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
            False
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
