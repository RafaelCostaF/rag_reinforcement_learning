import gym
from gym import spaces
import numpy as np
import spacy

nlp = spacy.load("en_core_web_md")

MAX_CHUNKS = 3
STOP_ACTION = MAX_CHUNKS  # special action to stop

class RankThenStopEnv(gym.Env):
    def __init__(self, df, reward_fn, inference_mode=False):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.reward_fn = reward_fn
        self.inference_mode = inference_mode
        self.current_index = -1
        self.max_steps = MAX_CHUNKS + 1  # or whatever makes sense
        self.steps_taken = 0

        self.observation_space = spaces.Dict({
            "similarities": spaces.Box(low=-1, high=1, shape=(MAX_CHUNKS,), dtype=np.float32),
            "distances": spaces.Box(low=0, high=np.inf, shape=(MAX_CHUNKS,), dtype=np.float32),
            "selected_mask": spaces.MultiBinary(MAX_CHUNKS),
        })

        # Action: choose a chunk index or STOP
        self.action_space = spaces.Discrete(MAX_CHUNKS + 1)

    def reset(self):
        self.current_index += 1
        if self.current_index >= len(self.df):
            self.current_index = 0

        row = self.df.iloc[self.current_index]
        self.query = row['query']
        self.answer = row['answer']
        self.chunks = list(row['retriever_rank']['texts'][:MAX_CHUNKS])
        self.distances = row['retriever_rank']['distances'][:MAX_CHUNKS]

        query_doc = nlp(self.query)
        self.similarities = np.zeros(MAX_CHUNKS, dtype=np.float32)
        self.distances = np.zeros(MAX_CHUNKS, dtype=np.float32)

        for i, chunk in enumerate(self.chunks):
            self.similarities[i] = query_doc.similarity(nlp(str(chunk)))
            self.distances[i] = row['retriever_rank']['distances'][i]

        self.selected_indices = []
        self.selected_mask = np.zeros(MAX_CHUNKS, dtype=np.int8)
        self.done = False
        self.steps_taken = 0  # Reset step counter each episode

        return self._get_obs()

    def step(self, action):
        # if self.done:
        #     raise RuntimeError("Episode is done. Call reset() to start a new episode.")
        self.steps_taken += 1
        if action == STOP_ACTION or len(self.selected_indices) >= MAX_CHUNKS or self.steps_taken >= self.max_steps:
            self.done = True
            if self.inference_mode:
                return self._get_obs(), 0.0, self.done, {}  # no reward
            
            selected_chunks = [self.chunks[i] for i in self.selected_indices]

            llm_score = self.reward_fn(self.query, selected_chunks, self.answer) * 10
            usage_ratio = len(selected_chunks) / MAX_CHUNKS
            efficiency_bonus = (1 - usage_ratio)
            reward = llm_score + efficiency_bonus

            return self._get_obs(), reward, self.done, {}

        if action in self.selected_indices or action >= len(self.chunks):
            # Penalize duplicate or invalid action
            reward = -1.0
            return self._get_obs(), reward, self.done, {}

        # ✅ Valid chunk selection
        self.selected_indices.append(action)
        self.selected_mask[action] = 1

        similarity = float(self.similarities[action])
        distance = float(self.distances[action])

        # Reward = similarity / (1 + distance), scaled to approx 10–100
        reward = (similarity / (1 + distance)) * 10

        return self._get_obs(), reward, self.done, {}



    def _get_obs(self):
        return {
            "similarities": self.similarities.astype(np.float32),
            "distances": self.distances.astype(np.float32),
            "selected_mask": self.selected_mask.astype(np.int8),
        }
