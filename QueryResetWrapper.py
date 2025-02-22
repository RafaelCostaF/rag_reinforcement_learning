import gymnasium as gym

class QueryResetWrapper(gym.Wrapper):
    """Wrapper to pass a new query and new text chunks on each reset."""
    def __init__(self, env, queries, text_chunks_list):
        super().__init__(env)
        self.queries = queries  
        self.text_chunks_list = text_chunks_list  
        self.query_index = 0  

    def reset(self, **kwargs):
        text_chunks = self.text_chunks_list[self.query_index]
        query = self.queries[self.query_index]
        self.query_index += 1

        return self.env.reset(options={"text_chunks": text_chunks, "query": query}, **kwargs)