import gym
import numpy as np
from gym import spaces
from sentence_transformers import SentenceTransformer, util
from Functions.documents import load_documents
from Functions.text import clear_text, remove_stopwords, split_text_into_chunks

class RAGImprovementEnv(gym.Env):
    def __init__(self, document_paths, chunk_size=500, model_name="all-MiniLM-L6-v2"):
        super(RAGImprovementEnv, self).__init__()
        
        texts = load_documents(document_paths)

        cleaned_texts = [clear_text(text) for text in texts]

        no_stopwords_text = [remove_stopwords(text) for text in cleaned_texts]

        joined_texts = " ".join(no_stopwords_text)

        text_chunks = split_text_into_chunks(joined_texts, chunk_size=chunk_size)

        # Initialize document and model
        self.documents = text_chunks  # List of text documents
        self.model = SentenceTransformer(model_name)
        
        # Process documents into embeddings for fast similarity comparison
        self.document_embeddings = [self.model.encode(chunk) for chunk in self.documents]
        
        # Define observation space (embedding space + similarity scores)
        self.observation_space = spaces.Dict({
            "similarities": spaces.Box(low=0.0, high=1.0, shape=(len(self.documents),), dtype=np.float32)
        })
        
        # Action space is the index of document chunks
        self.action_space = spaces.Discrete(len(documents))
        
        # Placeholder for query
        self.query_embedding = None

    def reset(self, query):
        # Encode user query as an embedding
        self.query_embedding = self.model.encode(query)
        
        # Calculate initial similarities to document chunks
        self.similarities = np.array([util.cos_sim(self.query_embedding, doc_emb).item() 
                                      for doc_emb in self.document_embeddings])
        
        # Return the initial observation
        return {"query_embedding": self.query_embedding, "similarities": self.similarities}

    def step(self, action):
        # Assume action is the index of the selected document chunk
        selected_chunk = self.documents[action]
        
        # Use an LLM (placeholder here) to check if the chunk meets the query needs
        llm_response = self._check_with_llm(selected_chunk)
        
        # Reward: 1 if the selected chunk answers the question, -1 otherwise
        reward = 1 if llm_response else -1
        done = True  # End episode after one choice for simplicity

        return {"query_embedding": self.query_embedding, "similarities": self.similarities}, reward, done, {}

    def _check_with_llm(self, chunk):
        # Placeholder function - replace with actual LLM call to validate chunk
        # Assume the LLM can validate if the chunk answers the query (True/False)
        return True  # Implement your own logic or call to LLM here

    def render(self, mode="human"):
        print(f"Query Embedding: {self.query_embedding}")
        print(f"Similarities: {self.similarities}")
