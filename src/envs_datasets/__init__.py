try:
    from .vizdoom_dataset import ViZDoomIterDataset
except ImportError:
    ViZDoomIterDataset = None
    raise ImportError("ViZDoomIterDataset is not found")

try:
    from .memory_maze_dataset import MemoryMazeDataset
except ImportError:
    MemoryMazeDataset = None
    raise ImportError("MemoryMazeDataset is not found")
try:
    from .minigrid_memory_dataset import MinigridMemoryIterDataset
except ImportError:
    MinigridMemoryIterDataset = None
    raise ImportError("MinigridMemoryIterDataset is not found")
try:
    from .tmaze_dataset import TMazeCombinedDataLoader, TMaze_data_generator
except ImportError:
    TMazeCombinedDataLoader = None
    TMaze_data_generator = None
    raise ImportError("TMazeCombinedDataLoader and TMaze_data_generator are not found")
try:
    from .popgym_dataset import POPGymDataset
except ImportError:
    POPGymDataset = None
    raise ImportError("POPGymDataset is not found")
try:
    from .mikasa_robo_dataset import MIKASARoboIterDataset
except ImportError:
    MIKASARoboIterDataset = None
    raise ImportError("MIKASARoboIterDataset is not found")

try:
    from .mdp_dataset import MDPDataset
except ImportError:
    MDPDataset = None
    raise ImportError("MDPDataset is not found")

try:
    from .arshot_dataset import ARShotDataset
except ImportError:
    ARShotDataset = None
    raise ImportError("ARShotDataset is not found")

try:
    from .mujoco_dataset import create_mujoco_dataloader
except ImportError:
    create_mujoco_dataloader = None
    raise ImportError("create_mujoco_dataloader is not found")