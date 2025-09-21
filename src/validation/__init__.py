# from .val_tmaze import get_returns_TMaze
# from .val_vizdoom_two_colors import get_returns_VizDoom
# from .val_minigridmemory import get_returns_MinigridMemory
# from .val_memory_maze import get_returns_MemoryMaze

try:
    from .val_tmaze import get_returns_TMaze
except Exception:
    get_returns_TMaze = None

try:
    from .val_vizdoom_two_colors import get_returns_VizDoom
except Exception:
    get_returns_VizDoom = None

try:
    from .val_minigridmemory import get_returns_MinigridMemory
except Exception:
    get_returns_MinigridMemory = None

try:
    from .val_memory_maze import get_returns_MemoryMaze
except Exception:
    get_returns_MemoryMaze = None

try:
    from .val_popgym import get_returns_POPGym
except Exception:
    get_returns_POPGym = None

try:
    from .val_mikasa_robo import get_returns_MIKASARobo
except Exception:
    get_returns_MIKASARobo = None

try:
    from .val_mdp import get_returns_MDP
except Exception:
    get_returns_MDP = None
