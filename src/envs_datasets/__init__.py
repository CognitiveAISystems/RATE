try:
    from .vizdoom_dataset import ViZDoomIterDataset
except ImportError:
    ViZDoomIterDataset = None

try:
    from .memory_maze_dataset import MemoryMazeDataset
except ImportError:
    MemoryMazeDataset = None

try:
    from .minigrid_memory_dataset import MinigridMemoryIterDataset
except ImportError:
    MinigridMemoryIterDataset = None

try:
    from .tmaze_dataset import TMazeCombinedDataLoader, TMaze_data_generator
except ImportError:
    TMazeCombinedDataLoader = None
    TMaze_data_generator = None

try:
    from .popgym_dataset import POPGymDataset
except ImportError:
    POPGymDataset = None

try:
    from .mikasa_robo_dataset import MIKASARoboIterDataset
except ImportError:
    MIKASARoboIterDataset = None