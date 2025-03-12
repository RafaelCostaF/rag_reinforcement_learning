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

        if self.num_documents == 0:
            raise ValueError("text_chunks is empty. The environment requires at least one document.")

        computed_embeddings = np.array([self.model.encode(chunk) for chunk in self.documents])

        if computed_embeddings.size == 0:
            raise ValueError("Computed embeddings are empty. Check if SentenceTransformer is working properly.")

        self.document_embeddings = np.zeros((self.max_num_documents, self.embedding_dim), dtype=np.float32)
        self.document_embeddings[:self.num_documents] = computed_embeddings.reshape(self.num_documents, -1)

        self.query_embedding = self.model.encode(query)

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
            {}
        )

    def compute_reward(self, action):
        similarity = self.similarities[action]
        
        if similarity < 0.4:
            return -0.5  # Reduce penalty for bad chunks
        elif similarity < 0.6:
            return 0.5  # Small positive reward to encourage exploration
        else:
            return similarity * 10  # Good chunks still have strong rewards

    def compute_final_reward(self):
        good_chunks = sum(1 for i, selected in enumerate(self.retrieved_chunks_mask) if selected and self.similarities[i] >= 0.6)
    
        if good_chunks == 0:
            return -5.0  # Strong penalty if no good chunk was selected
        else:
            return 5 + good_chunks * 2 - self.step_count  # Subtract step count to reward efficiency


    # def compute_reward(self, action):
    #     """Reward based on similarity category."""
    #     similarity = self.similarities[action]

    #     if similarity < 0.4:
    #         return -2.0  # Bad chunk (Penalty)
    #     elif similarity < 0.6:
    #         return 1.0  # Neutral chunk (Small reward)
    #     else:
    #         return similarity * 10  # Good chunk (Scaled reward)

    # def compute_final_reward(self):
    #     """Encourage stopping at the right time based on good selections."""
    #     good_chunks = sum(1 for i, selected in enumerate(self.retrieved_chunks_mask) if selected and self.similarities[i] >= 0.6)
        
    #     if good_chunks == 0:
    #         return -5.0  # Strong penalty if no good chunk was selected
    #     else:
    #         return 5 + good_chunks * 2  # Reward based on good chunk count


    # def compute_reward(self, action):
    #     """Reward based on relevance (cosine similarity) and step efficiency."""
    #     return self.similarities[action] * (10 - (self.step_count * 2))

    # def compute_final_reward(self):
    #     """Encourage stopping at the right time."""
    #     return 10.0
    
    
    # def compute_reward(self, action):
    #     """Reward based on relevance (cosine similarity) and step efficiency."""
    #     relevance = self.similarities[action]  # How relevant the chunk is
    #     step_efficiency = 10 - (self.step_count * 2)  # Penalize actions after more steps
    #     return relevance * step_efficiency

    # def compute_final_reward(self):
    #     """Encourage stopping at the right time."""
    #     # If the model stops after selecting relevant chunks, reward more
    #     num_relevant_chunks = sum(self.retrieved_chunks_mask)  # Count relevant chunks selected
    #     return 5 + num_relevant_chunks * 2  # Increase reward based on the number of relevant chunks retrieved

    def render(self):
        print(f"Query Embedding: {self.query_embedding}")
        print(f"Similarities: {self.similarities}")
        print(f"Retrieved Chunks: {self.retrieved_chunks_mask}")
