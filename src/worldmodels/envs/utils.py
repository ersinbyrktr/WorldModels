from gymnasium.spaces import Box


def get_action_bounds(env):
    """
    Returns a list of (low, high) tuples for each action dimension,
    using standard Python float values.
    """
    if isinstance(env.action_space, Box):
        low = env.action_space.low
        high = env.action_space.high
        return [(float(l), float(h)) for l, h in zip(low, high)]
    else:
        raise TypeError("This function only supports environments with Box action spaces.")
