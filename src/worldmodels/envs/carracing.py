import gymnasium as gym


def make_env(seed: int):
    """Factory that creates ONE CarRacing env (called in each worker)."""

    def _thunk():
        env = gym.make("CarRacing-v3", render_mode="rgb_array", lap_complete_percent=0.95, domain_randomize=False)
        env.reset(seed=seed)
        return env

    return _thunk


def get_envs(seed, workers):
    env_fns = [make_env(seed + i) for i in range(workers)]
    envs = gym.vector.AsyncVectorEnv(env_fns)
    return envs
