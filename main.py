from Functions.documents import load_documents
from Functions.text import clear_text, remove_stopwords, split_text_into_chunks, remove_portuguese_accents
from stable_baselines3 import PPO
from RAGImprovementEnv import RAGImprovementEnv
from QueryResetWrapper import QueryResetWrapper
import pandas as pd
import gymnasium as gym

CHUNK_SIZE = 500

# BEGIN = Setting texts and queries
dataset = pd.read_parquet("./tests/dataset/Crag Splitted/crag_dataset_0.parquet")  # Documents from dataset
texts = dataset['page_result'].copy()
queries = dataset['query'].copy()
# END = Setting texts and queries

# BEGIN = Processing documents 
cleaned_texts = [clear_text(text) for text in texts]
no_stopwords_text = [remove_stopwords(text, language='english') for text in cleaned_texts]
no_accents_portuguese = [remove_portuguese_accents(text) for text in no_stopwords_text]
lower_texts = [text.lower() for text in no_accents_portuguese]
text_chunks = [split_text_into_chunks(text, chunk_size=CHUNK_SIZE) for text in lower_texts]
# END = Processing documents 

# Initialize environment with wrapper
def make_env():
    env = RAGImprovementEnv()
    env = QueryResetWrapper(env, queries, text_chunks)  # Wrap to pass new query in reset
    return env

# env = gym.vector.SyncVectorEnv([make_env])  # Use SyncVectorEnv instead of DummyVecEnv
env = make_env()

# Train with PPO
model = PPO("MultiInputPolicy", env, verbose=1, learning_rate=0.0003, tensorboard_log='./PPO_tensorboard_20250602/')
model.learn(total_timesteps=100000)
