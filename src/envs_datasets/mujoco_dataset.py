# mujoco_dataset.py
from __future__ import annotations
import os
import pickle
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ---- Your precomputed statistics (used verbatim) ----
PRECOMPUTED_DATASET_STATISTICS = {
    "halfcheetah-expert-v2": {
        "obs_mean": np.array([
            -0.04489148, 0.03232588, 0.06034835, -0.17081226, -0.19480659, -0.05751596,
            0.09701628, 0.03239211, 11.047426, -0.07997331, -0.32363534, 0.36297753,
            0.42322603, 0.40836546, 1.1085187, -0.4874403, -0.0737481
        ]),
        "obs_std": np.array([
            0.04002118, 0.4107858, 0.54217845, 0.41522816, 0.23796624, 0.62036866,
            0.30100912, 0.21737163, 2.2105937, 0.572586, 1.7255033, 11.844218,
            12.06324, 7.0495934, 13.499867, 7.195647, 5.0264325,
        ]),
    },
    "halfcheetah-medium-expert-v2": {
        "obs_mean": np.array([
            -0.05667463,  0.02436997, -0.06167056, -0.22351515, -0.26751512, -0.07545716,
            -0.05809683, -0.02767508,  8.110626,   -0.06136331, -0.17986928,  0.25175223,
            0.24186333,  0.25193694,  0.5879553,  -0.24090636, -0.03018427
        ]),
        "obs_std": np.array([
           0.06103534,  0.36054105,  0.455444,    0.38476887,  0.22183637,  0.5667524,
           0.3196683,   0.28529236,  3.443822,    0.67281395,  1.8616977,   9.575808,
           10.029895,    5.90345,    12.128185,    6.4811788,   6.37862,
        ]),
    },
    "halfcheetah-medium-replay-v2": {
        "obs_mean": np.array([
            -0.12880704, 0.37381196, -0.14995988, -0.23479079, -0.28412786, -0.13096535,
            -0.20157982, -0.06517727, 3.4768248, -0.02785066, -0.01503525,  0.07697279,
            0.01266712, 0.0273253, 0.02316425, 0.01043872, -0.01583941
        ]),
        "obs_std": np.array([
            0.17019016, 1.2844249, 0.33442774, 0.36727592, 0.26092398, 0.4784107,
            0.31814206, 0.33552638, 2.0931616, 0.80374336, 1.9044334, 6.57321,
            7.5728636, 5.0697494, 9.105554, 6.0856543, 7.253004,
        ]),
    },
    "halfcheetah-medium-v2": {
        "obs_mean": np.array([
            -0.06845774, 0.01641455, -0.18354906, -0.27624607, -0.34061527, -0.09339716,
            -0.21321271, -0.08774239, 5.1730075, -0.04275195, -0.03610836, 0.14053793,
            0.06049833, 0.09550975, 0.067391, 0.00562739, 0.01338279,
        ]),
        "obs_std": np.array([
            0.07472999, 0.30234998, 0.3020731, 0.34417078, 0.17619242, 0.5072056,
            0.25670078, 0.32948127, 1.2574149, 0.7600542, 1.9800916, 6.5653625,
            7.4663677, 4.472223, 10.566964, 5.6719327, 7.498259,
        ]),
    },



    "hopper-expert-v2": {
        "obs_mean": np.array([
            1.3490015, -0.11208222, -0.5506444, -0.13188992, -0.00378754,  2.6071432,
            0.02322114, -0.01626922, -0.06840388, -0.05183131,  0.04272673
        ]),
        "obs_std": np.array([
            0.15980862, 0.0446214, 0.14307782, 0.17629202, 0.5912333, 0.5899924,
            1.5405099, 0.8152689, 2.0173461, 2.4107876, 5.8440027,
        ]),
    },
    "hopper-medium-expert-v2": {
        "obs_mean": np.array([
            1.3293816,  -0.09836531, -0.5444298,  -0.10201651,  0.02277466,  2.3577216,
            -0.06349576, -0.00374026, -0.17662701, -0.11862941, -0.1209782,
        ]),
        "obs_std": np.array([
            0.17012376, 0.05159067, 0.18141434, 0.16430604, 0.6023368,  0.7737285,
            1.4986556,  0.74833184, 1.795316,   2.0530026,  5.725033,
        ]),
    },
    "hopper-medium-replay-v2": {
        "obs_mean": np.array([
            1.2305138,  -0.04371411, -0.44542956, -0.09370098,  0.09094488,  1.3694725,
            -0.19992675, -0.02286135, -0.5287045,  -0.14465883, -0.19652697,
        ]),
        "obs_std": np.array([
            0.17565121, 0.06369286, 0.34383234, 0.19566889, 0.5547985, 1.0510299,
            1.1583077, 0.79631287, 1.4802359, 1.6540332, 5.108601,
        ]),
    },
    "hopper-medium-v2": {
        "obs_mean": np.array([
            1.311279,   -0.08469521, -0.5382719,  -0.07201576,  0.04932366,  2.1066856,
            -0.15017354,  0.00878345, -0.2848186,  -0.18540096, -0.28461286,
        ]),
        "obs_std": np.array([
            0.17790751, 0.05444621, 0.21297139, 0.14530419, 0.6124444, 0.85174465,
            1.4515252, 0.6751696, 1.536239, 1.6160746, 5.6072536,
        ]),
    },



    "walker2d-expert-v2": {
        "obs_mean": np.array([
            1.2384834e+00,  1.9578537e-01, -1.0475016e-01, -1.8579608e-01,
            2.3003316e-01,  2.2800924e-02, -3.7383768e-01,  3.3779100e-01,
            3.9250960e+00, -4.7428459e-03,  2.5267061e-02, -3.9287535e-03,
            -1.7367510e-02, -4.8212224e-01,  3.5432147e-04, -3.7124525e-03,
            2.6285544e-03
        ]),
        "obs_std": np.array([
            0.06664903, 0.16980624, 0.17309439, 0.21843709, 0.74599105, 0.02410989,
            0.3729872,  0.6226182,  0.9708009,  0.72936815, 1.504065,   2.495893,
            3.511518,   5.3656907,  0.79503316, 4.317483,   6.1784487
        ]),
    },
    "walker2d-medium-expert-v2": {
        "obs_mean": np.array([
            1.2294334e+00,  1.6869690e-01, -7.0890814e-02, -1.6197483e-01,
            3.7101927e-01, -1.2209027e-02, -4.2461398e-01,  1.8986578e-01,
            3.1624751e+00, -1.8092677e-02,  3.4969468e-02, -1.3921680e-02,
            -5.9370294e-02, -1.9549426e-01, -1.9200450e-03, -6.2483322e-02,
            -2.7366525e-01
        ]),
        "obs_std": np.array([
            0.09932825, 0.259814,   0.1506276,  0.24249177, 0.67587185, 0.16507415,
            0.38140664, 0.69623613, 1.350149,   0.76419914, 1.5345743,  2.1785972,
            3.2765827,  4.766194,   1.1716983,  4.0397825,  5.891614
        ]),
    },
    "walker2d-medium-replay-v2": {
        "obs_mean": np.array([
            1.2093647,   0.13264023, -0.14371201, -0.20465161,  0.55776125, -0.03231537,
            -0.2784661,   0.19130707,  1.4701707,  -0.12504704,  0.05649531, -0.09991033,
            -0.34034026,  0.03546293, -0.08934259, -0.2992438,  -0.5984178,
        ]),
        "obs_std": np.array([
            0.11929835, 0.3562574,  0.258522,   0.42075422, 0.5202291,  0.15685083,
            0.3677098,  0.7161388,  1.3763766,  0.8632222,  2.6364644,  3.0134118,
            3.720684,   4.867284,   2.6681626,  3.845187,   5.4768386,
        ]),
    },
    "walker2d-medium-v2": {
        "obs_mean": np.array([
            1.218966,  0.14163373, -0.03704914, -0.1381431,   0.51382244, -0.0471911,
            -0.47288352,  0.04225416,  2.3948874,  -0.03143199,  0.04466356, -0.02390724,
            -0.10134014,  0.09090938, -0.00419264, -0.12120572, -0.5497064,
        ]),
        "obs_std": np.array([
            0.12311358, 0.324188,   0.11456084, 0.26230657, 0.5640279,  0.22718786,
            0.38373196, 0.7373677,  1.2387927,  0.7980206,  1.5664079,  1.8092705,
            3.0256042,  4.062486,   1.4586568,  3.744569,   5.585129,
        ]),
    },
}


def _infer_key_from_path(pkl_path: str) -> str:
    """Extract dataset key 'env-dataset-v2' from '.../env-dataset-v2.pkl'."""
    base = os.path.basename(pkl_path)
    if base.endswith(".pkl"):
        base = base[:-4]
    return base  # expected to match keys in PRECOMPUTED_DATASET_STATISTICS


class D4RLMuJoCoSequenceDataset(Dataset):
    """
    PyTorch dataset for MuJoCo vector trajectories saved in Decision-Transformer format.

    Normalization:
      - If normalize_obs=True, use *precomputed* stats from PRECOMPUTED_DATASET_STATISTICS
        keyed by the .pkl filename (without extension). No runtime estimation.
      - If normalize_obs=False, return raw observations.

    Cropping:
      - "sliding": overlapping windows (stride=1)  [Default — recommended for DT].
      - "disjoint": non-overlapping windows (stride=L).
      - "stride": user-defined stride w (>0).
    """

    def __init__(
        self,
        pkl_path: str,
        sequence_length: int,
        gamma: float = 1.0,
        crop_mode: str = "sliding",
        stride: Optional[int] = None,
        drop_short: bool = True,
        normalize_obs: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert os.path.isfile(pkl_path), f"File not found: {pkl_path}"
        with open(pkl_path, "rb") as f:
            self.trajectories: List[Dict[str, Any]] = pickle.load(f)
        assert isinstance(self.trajectories, list) and len(self.trajectories) > 0
        assert sequence_length > 0

        self.pkl_path = pkl_path
        self.L = int(sequence_length)
        self.gamma = float(gamma)
        self.crop_mode = str(crop_mode).lower()
        self.drop_short = bool(drop_short)
        self.eps = float(eps)

        # stride policy
        if self.crop_mode not in {"sliding", "disjoint", "stride"}:
            raise ValueError(f"Unknown crop_mode: {crop_mode}")
        if self.crop_mode == "stride":
            assert stride is not None and stride > 0, "Provide a positive stride for crop_mode='stride'"
            self.stride = int(stride)
        elif self.crop_mode == "sliding":
            self.stride = 1
        else:
            self.stride = self.L

        # coerce arrays
        for ep in self.trajectories:
            for k in ["observations", "actions", "rewards", "terminals"]:
                assert k in ep, f"Missing key '{k}' in a trajectory"
            for k in list(ep.keys()):
                if isinstance(ep[k], list):
                    ep[k] = np.asarray(ep[k])

        self.state_dim = int(self.trajectories[0]["observations"].shape[-1])
        self.act_dim = int(self.trajectories[0]["actions"].shape[-1] if self.trajectories[0]["actions"].ndim > 1 else 1)

        # precomputed normalization (if requested)
        self.normalize = bool(normalize_obs)
        if self.normalize:
            key = _infer_key_from_path(pkl_path)
            if key not in PRECOMPUTED_DATASET_STATISTICS:
                raise KeyError(
                    f"No precomputed stats for key '{key}'. "
                    f"Available: {list(PRECOMPUTED_DATASET_STATISTICS.keys())}"
                )
            stats = PRECOMPUTED_DATASET_STATISTICS[key]
            obs_mean = np.asarray(stats["obs_mean"], dtype=np.float32)
            obs_std  = np.asarray(stats["obs_std"],  dtype=np.float32)
            if obs_mean.shape[0] != self.state_dim or obs_std.shape[0] != self.state_dim:
                raise ValueError(
                    f"Precomputed stats dimension mismatch for '{key}': "
                    f"obs_dim={self.state_dim}, mean_dim={obs_mean.shape[0]}, std_dim={obs_std.shape[0]}"
                )
            # ensure positive std
            obs_std = np.maximum(obs_std, self.eps)
            self.obs_mean = obs_mean
            self.obs_std  = obs_std
        else:
            self.obs_mean = np.zeros(self.state_dim, dtype=np.float32)
            self.obs_std  = np.ones(self.state_dim, dtype=np.float32)

        # index windows
        self.index: List[Tuple[int, int, int]] = []
        for ti, ep in enumerate(self.trajectories):
            # if ti > 1: break # TODO: [CRITICAL] comment after debugging!!!
            T = int(ep["observations"].shape[0])
            if self.drop_short:
                if T >= self.L:
                    for t in range(0, T - self.L + 1, self.stride):
                        self.index.append((ti, t, self.L))
            else:
                for t in range(0, T, self.stride):
                    le = min(self.L, T - t)
                    if le <= 0: break
                    self.index.append((ti, t, le))

    def __len__(self) -> int:
        return len(self.index)

    @staticmethod
    def _discount_cumsum(r: np.ndarray, gamma: float) -> np.ndarray:
        T = r.shape[0]
        out = np.zeros_like(r, dtype=np.float32)
        acc = 0.0
        for t in range(T - 1, -1, -1):
            acc = float(r[t]) + gamma * acc
            out[t] = acc
        return out

    def __getitem__(self, idx: int):
        ti, t0, le = self.index[idx]
        ep = self.trajectories[ti]

        s = ep["observations"][t0:t0+le]                # (le, state_dim)
        a = ep["actions"][t0:t0+le]                     # (le, act_dim) or (le,)
        r = ep["rewards"][t0:t0+le]                     # (le,)
        d = ep["terminals"][t0:t0+le].astype(np.int64)  # (le,)

        rtg = self._discount_cumsum(r, self.gamma).reshape(-1, 1)  # (le, 1)
        timesteps = np.arange(le, dtype=np.int64)

        L = self.L
        if le < L:
            pad = L - le
            s   = np.pad(s,   ((0, pad), (0, 0)), mode="constant")
            if a.ndim == 1:
                a = a[:, None]
            a   = np.pad(a,   ((0, pad), (0, 0)), mode="constant")
            rtg = np.pad(rtg, ((0, pad), (0, 0)), mode="constant")
            d   = np.pad(d,   (0, pad), mode="constant")
            timesteps = np.pad(timesteps, (0, pad), mode="constant")
            mask = np.zeros(L, dtype=np.float32); mask[:le] = 1.0
        else:
            if a.ndim == 1:
                a = a[:, None]
            mask = np.ones(L, dtype=np.float32)

        # apply precomputed normalization if enabled
        if self.normalize:
            s = (s.astype(np.float32) - self.obs_mean) / self.obs_std
        else:
            s = s.astype(np.float32)

        s   = torch.from_numpy(s)                    # (L, state_dim)
        a   = torch.from_numpy(a.astype(np.float32)) # (L, act_dim)
        rtg = torch.from_numpy(rtg.astype(np.float32))  # (L, 1)
        d   = torch.from_numpy(d.astype(np.int64))      # (L,)
        tms = torch.from_numpy(timesteps.astype(np.int64))
        msk = torch.from_numpy(mask.astype(np.float32))

        return s, a, rtg, d, tms, msk


def create_mujoco_dataloader(
    pkl_path: str,
    sequence_length: int,
    batch_size: int,
    gamma: float = 1.0,
    crop_mode: str = "sliding",       # "sliding" (2.1), "disjoint" (2.2), "stride" (2.3)
    stride: Optional[int] = None,     # only if crop_mode == "stride"
    drop_short: bool = True,
    normalize_obs: bool = False,      # use precomputed stats if True
    num_workers: int = 8,
    shuffle: bool = True,
    pin_memory: bool = True,
) -> DataLoader:
    ds = D4RLMuJoCoSequenceDataset(
        pkl_path=pkl_path,
        sequence_length=sequence_length,
        gamma=gamma,
        crop_mode=crop_mode,
        stride=stride,
        drop_short=drop_short,
        normalize_obs=normalize_obs,
    )
    key = _infer_key_from_path(pkl_path)
    print(
        f"[MuJoCoDataset] file={os.path.basename(pkl_path)} key={key} "
        f"| windows={len(ds)} | L={sequence_length} | crop={crop_mode}"
        f"{'' if crop_mode!='stride' else f' (stride={stride})'} "
        f"| gamma={gamma} | normalize={'precomputed' if normalize_obs else 'none'}"
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
