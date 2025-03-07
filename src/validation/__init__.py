# from .val_tmaze import get_returns_TMaze
# from .val_vizdoom_two_colors import get_returns_VizDoom
# from .val_minigridmemory import get_returns_MinigridMemory
# from .val_memory_maze import get_returns_MemoryMaze

try:
    from .val_tmaze import get_returns_TMaze
except ImportError:
    get_returns_TMaze = None

try:
    from .val_vizdoom_two_colors import get_returns_VizDoom
except ImportError:
    get_returns_VizDoom = None

try:
    from .val_minigridmemory import get_returns_MinigridMemory
except ImportError:
    get_returns_MinigridMemory = None

try:
    from .val_memory_maze import get_returns_MemoryMaze
except ImportError:
    get_returns_MemoryMaze = None