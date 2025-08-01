import gymnasium as gym

from src.worldmodels.envs.utils import get_action_bounds


def make_env(render_mode="rgb_array"):
    """Factory that creates ONE CarRacing env (called in each worker)."""

    def _thunk():
        env = gym.make("CarRacing-v3", render_mode=render_mode, lap_complete_percent=0.95, domain_randomize=False)
        env.reset()
        return env

    return _thunk


def get_envs(workers):
    env_fns = [make_env() for _ in range(workers)]
    envs = gym.vector.AsyncVectorEnv(env_fns)
    return envs


action_space = [(-1.0, 1.0), (0.0, 1.0), (0.0, 1.0)]

if __name__ == "__main__":
    env = make_env()()
    print("Action space:", env.action_space)
    print("Action bounds:", get_action_bounds(env))
