import pandas as pd
df = pd.read_parquet("CRAG_grouped_0_1_2_with_retriever_rank_fixed.parquet")
df.columns
df.head(3)
df.retriever_rank.values[0]
import pandas as pd
import numpy as np
from itertools import permutations

def augment_retriever_ranks(df, num_permutations=5):
    augmented_rows = []

    for _, row in df.iterrows():
        rank = row['retriever_rank']
        triplet = list(zip(rank['distances'], rank['indices'], rank['texts']))
        
        # Generate all permutations except the original
        all_perms = list(permutations(triplet))
        original_tuple = tuple(triplet)
        all_perms.remove(original_tuple)

        # Select up to num_permutations unique permutations
        selected_perms = all_perms[:num_permutations]

        # Add the original row first
        augmented_rows.append(row.to_dict())

        # Add permuted rows
        for perm in selected_perms:
            distances, indices, texts = zip(*perm)
            new_rank = {
                'distances': np.array(distances),
                'indices': np.array(indices),
                'texts': np.array(texts)
            }
            new_row = row.to_dict()
            new_row['retriever_rank'] = new_rank
            augmented_rows.append(new_row)

    return pd.DataFrame(augmented_rows).reset_index(drop=True)

# Assuming df already has the retriever_rank column
df_augmented = augment_retriever_ranks(df, num_permutations=6)

df.shape
df_augmented.shape
# from RankThenStopEnv import RankThenStopEnv
# from Functions.llm import get_evaluation_from_llm as reward_fn
# from stable_baselines3 import PPO
# from stable_baselines3.common.callbacks import CheckpointCallback

# # Env setup
# env = RankThenStopEnv(df=df_augmented, reward_fn=reward_fn)

# # Callbacks
# checkpoint_callback = CheckpointCallback(save_freq=1000, save_path="./ppo_checkpoints/", name_prefix="ppo_model")

# # Train
# model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log="./ppo_tensorboard/", n_steps=128)
# model.learn(total_timesteps=5000, callback=checkpoint_callback)

from RankThenStopEnv import RankThenStopEnv
from Functions.llm import get_evaluation_from_llm as reward_fn
from stable_baselines3 import PPO, A2C, DQN
from sb3_contrib import QRDQN
from stable_baselines3.common.callbacks import CheckpointCallback

# Algorithms to run
algorithms = {
    # "ppo": PPO,
    "a2c": A2C,
    "dqn": DQN,
    "qrdqn": QRDQN,
}

# Train each model
for algo_name, AlgoClass in algorithms.items():
    print(f"🔁 Training {algo_name.upper()}...")

    # Env setup
    env = RankThenStopEnv(df=df_augmented, reward_fn=reward_fn)

    # Callback
    checkpoint_callback = CheckpointCallback(
        save_freq=1000,
        save_path=f"./{algo_name}_checkpoints/",
        name_prefix=f"{algo_name}_model"
    )

    # Model
    model = AlgoClass(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log=f"./{algo_name}_tensorboard/",
        n_steps=128 if algo_name in ["ppo", "a2c"] else None,  # only for on-policy algorithms
    )

    # Learn
    model.learn(total_timesteps=5000, callback=checkpoint_callback)

    # Optional: Save model
    model.save(f"./{algo_name}_final_model")

    print(f"✅ {algo_name.upper()} training done.\n")
