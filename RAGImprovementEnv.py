import gym
import numpy as np
from gym import spaces
from sentence_transformers import SentenceTransformer, util

# State
# Means that you know everything knowable that could determine
# the response of the environment to a specific action

class RAGImprovementEnv(gym.Env):
    def __init__(self, text_chunks, embedding_model_name="all-MiniLM-L6-v2"):
        super(RAGImprovementEnv, self).__init__()

        # Initialize document and model
        self.documents = text_chunks  # List of text documents
        self.model = SentenceTransformer(embedding_model_name)
        
        # Process documents into embeddings for fast similarity comparison
        self.document_embeddings = [self.model.encode(chunk) for chunk in self.documents]
        
        # Assuming document_embeddings is already computed
        embedding_dim = len(self.document_embeddings[0])  # Dimensionality of each embedding
        num_documents = len(self.document_embeddings)  # Number of document chunks

        self.observation_space = spaces.Dict({
            "document_embeddings": spaces.Box(
                low=-np.inf, high=np.inf, shape=(num_documents, embedding_dim), dtype=np.float32
            ),  # Multiple document embeddings
            "query_embeddings": spaces.Box(
                low=-np.inf, high=np.inf, shape=(embedding_dim,), dtype=np.float32
            ),  # A single query embedding
        })

        # Action space is the index of document chunks PLUS stop action
        self.action_space = spaces.Discrete(len(self.documents)+1)
        
        # Placeholder for query
        self.query_embedding = None
        
    def reset(self, query):
        self.step_count = 0
        
        # Encode user query as an embedding
        self.query_embeddings = self.model.encode(query)  # Ensure consistency with observation_space
        
        # Compute cosine similarity between the question and all document chunks
        cosine_similarities = [util.pytorch_cos_sim(self.query_embedding, doc_emb).item() for doc_emb in self.document_embeddings]

        # Convert cosine similarity to a distance measure (1 - similarity)
        cosine_distances = [1 - sim for sim in cosine_similarities]

        self.similarities = np.array(cosine_distances.copy())

        # Return the initial observation
        return {
            "document_embeddings": self.document_embeddings,  # List of document embeddings
            "query_embeddings": self.query_embeddings  # Single query embedding
        }

    def step(self, action):
        self.step_count +=1
        if action == 0:  # Stop action
            done = True
            reward = self.compute_final_reward()  # Final reward when stopping
        else:
            done = False
            reward = self.compute_reward(action - 1)  # Adjust index (1-based to 0-based)
            
        # Assume action is the index of the selected document chunk
        selected_chunk = self.documents[action]
    
        return {"query_embedding": self.query_embedding, "similarities": self.similarities}, reward, done, {}

    def compute_reward(self, action):
        """
            Reward for selecting a document (e.g., cosine similarity with query).
            Reward better for less responses
        """
        return self.similarities[action] * (10 - (self.step_count*2))  # Higher similarity = better reward

    def compute_final_reward(self):
        """Final reward for stopping: Encourage stopping at the right time"""
        return 10.0  # Adjust this value as needed

    def render(self, mode="human"):
        print(f"Query Embedding: {self.query_embedding}")
        print(f"Similarities: {self.similarities}")
