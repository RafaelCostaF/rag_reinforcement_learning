from Functions.documents import load_documents
from Functions.text import clear_text, remove_stopwords, split_text_into_chunks, remove_portuguese_accents
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from RAGImprovementEnv import RAGImprovementEnv
from QueryResetWrapper import QueryResetWrapper
import pandas as pd
import gymnasium as gym

CHUNK_SIZE = 500

# BEGIN = Setting texts and queries
dataset = pd.read_parquet("./Dataset/Crag Splitted/crag_dataset_0.parquet")  # Documents from dataset
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



# END = Processing documents 
# Check if queries or text chunks are empty and drop the same index from both
non_empty_indices = [i for i, (query, chunk) in enumerate(zip(queries, text_chunks)) if query.strip() and chunk]

# Filter out the empty values from both queries and text_chunks using the non_empty_indices
queries = [queries[i] for i in non_empty_indices]
text_chunks = [text_chunks[i] for i in non_empty_indices]

# Ensure indices exist in both queries and text_chunks by creating a set of valid indices
valid_indices = set(non_empty_indices)

# Create a new list that holds only the valid indices that exist in both lists
valid_queries = [queries[i] for i in range(len(queries)) if i in valid_indices]
valid_text_chunks = [text_chunks[i] for i in range(len(text_chunks)) if i in valid_indices]

# Now, use these valid queries and valid text_chunks
filtered_queries = valid_queries
filtered_text_chunks = valid_text_chunks

# Initialize environment with wrapper
def make_env():
    env = RAGImprovementEnv()
    env = QueryResetWrapper(env, filtered_queries, filtered_text_chunks)  # Wrap to pass new query in reset
    return env

# env = gym.vector.SyncVectorEnv([make_env])  # Use SyncVectorEnv instead of DummyVecEnv
env = make_env()


# PPO
# # Create a checkpoint callback that saves the model every 10,000 timesteps
# checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./ppo_checkpoints/', name_prefix='ppo_model')

# # # Train with PPO and use the callback
# model = PPO("MultiInputPolicy", env, verbose=1, learning_rate=0.0003, tensorboard_log='./PPO_tensorboard_20250602/')
# model.learn(total_timesteps=100000, callback=checkpoint_callback)


# DQN
# from stable_baselines3 import DQN
# from stable_baselines3.common.callbacks import CheckpointCallback

# # Create a checkpoint callback that saves the model every 10,000 timesteps
# checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./dqn_checkpoints/', name_prefix='dqn_model')

# # Train with DQN and use the callback
# # model = DQN("MultiInputPolicy", env, verbose=1, learning_rate=0.0003, tensorboard_log='./DQN_tensorboard_20250602/')
# model = DQN(
#     "MultiInputPolicy", env, verbose=1, learning_rate=0.0003,
#     buffer_size=10_000,  # Reduce from 1_000_000 to 50_000
#     tensorboard_log='./DQN_tensorboard_20250602/'
# )
# model.learn(total_timesteps=100000, callback=checkpoint_callback)


from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import CheckpointCallback

# Create a checkpoint callback that saves the model every 10,000 timesteps
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./a2c_checkpoints/', name_prefix='a2c_model')

# Train with A2C and use the callback
model = A2C("MultiInputPolicy", env, verbose=1, learning_rate=0.0003, tensorboard_log='./A2C_tensorboard_20250602/')
model.learn(total_timesteps=100000, callback=checkpoint_callback)
