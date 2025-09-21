# pip install gymnasium
from __future__ import annotations
import random
import string
from itertools import product
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces


class ARShotEnv(gym.Env):
    """
    Associative Retrieval (AR) with a 'shot' query.

    Token stream:
      - shot_mode="after_pairs":
          [! k : v !] x P  +  [! query_key : shot]          → length T = 5*P + 4
      - shot_mode="after_any_colon":
          [! k : v !] x P  +  [! k : v !] x E + [! k : shot] → T = 5*(P+E) + 4
        (query_key was previously encountered as a complete pair)

    Observation — one token per step.
    Reward is given only when current token == 'shot':
        reward = 1, if action == correct value; otherwise 0. Episode terminates.

    Important flags:
      - deterministic_vocab=True  → universe order is fixed (doesn't depend on seed)
      - full_universe_vocab=True → env.vocab includes the entire token universe by lengths
      - randomize_pairs=True     → keys and values for EPISODE are random (but from fixed universe)
      - include_pass_token=True  → add 'pass' to special tokens (can be used as no-op)


    # Token to ID mapping and back
    obs_id = env.token_to_id["zA"]   # for example 1287
    print(obs_id) # 1287
    tok = env.id_to_token[obs_id]    # returns "zA"
    print(tok) # "zA"
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        n_pairs: int = 6,
        rng_seed: Optional[int] = None,

        # where shot appears in terms of "how many complete pairs to show first"
        prefix_pairs_range: Optional[Tuple[int, int]] = None,  # default (1, n_pairs)
        query_from_any_shown: bool = True,  # otherwise take the last one shown

        shot_mode: str = "after_pairs",  # "after_pairs" | "after_any_colon"
        max_extra_pairs_before_shot: int = 0,  # only for "after_any_colon"

        # vocabularies (if None — take from universe according to modes below)
        keys_vocab: Optional[List[str]] = None,
        values_vocab: Optional[List[str]] = None,

        # token length ranges (inclusive); most often (2,2)
        key_token_len_range: Tuple[int, int] = (2, 2),
        value_token_len_range: Tuple[int, int] = (2, 2),

        # alphabets for NON-DETERMINISTIC generation
        key_charset: str = string.ascii_letters + string.digits,
        value_charset: str = string.ascii_letters + string.digits,

        # vocabulary control and its stability
        deterministic_vocab: bool = True,
        full_universe_vocab: bool = True,
        randomize_pairs: bool = True,
        include_pass_token: bool = False,
    ):
        super().__init__()

        # RNG for episode dynamics and (optionally) random pair selection
        self.rng = random.Random(rng_seed)

        # --- parameter checks ---
        assert n_pairs >= 1, "n_pairs must be >= 1"
        if prefix_pairs_range is None:
            prefix_pairs_range = (1, n_pairs)
        min_p, max_p = prefix_pairs_range
        if not (1 <= min_p <= max_p <= n_pairs):
            raise ValueError("prefix_pairs_range must satisfy 1 <= min <= max <= n_pairs")
        if shot_mode not in ("after_pairs", "after_any_colon"):
            raise ValueError("shot_mode must be 'after_pairs' or 'after_any_colon'")

        self.n_pairs = n_pairs
        self.prefix_pairs_range = (min_p, max_p)
        self.query_from_any_shown = query_from_any_shown
        self.shot_mode = shot_mode
        self.max_extra_pairs_before_shot = max(0, int(max_extra_pairs_before_shot))

        # ---- SPECIAL tokens
        self.SPECIAL = ["!", ":", "shot", "pass"]
        if include_pass_token:
            self.SPECIAL.append("pass")
        reserved = set(self.SPECIAL)

        # ---- deterministic token universe by length range
        def det_tokens_for_range(length_range: Tuple[int, int]) -> List[str]:
            """
            Generates all tokens in lexicographic order using fixed alphabet:
              digits + lowercase + uppercase = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
            For length L: iterate through product(charset, repeat=L).
            """
            lo, hi = length_range
            if lo < 1 or hi < lo:
                raise ValueError("length_range must satisfy 1 <= lo <= hi")
            charset = "0123456789" + string.ascii_lowercase + string.ascii_uppercase
            out: List[str] = []
            for L in range(lo, hi + 1):
                for tup in product(charset, repeat=L):
                    out.append("".join(tup))
            # no special strings here, but keep the filter
            return [t for t in out if t not in reserved]

        # ---- random token generation (used only if deterministic_vocab=False and vocabularies not provided)
        def random_tokens(need: int, charset: str, length_range: Tuple[int, int], avoid: set[str]) -> List[str]:
            lo, hi = length_range
            if lo < 1 or hi < lo:
                raise ValueError("length_range must satisfy 1 <= lo <= hi")
            tokens: List[str] = []
            seen = set(avoid)
            attempts = 0
            while len(tokens) < need:
                attempts += 1
                L = self.rng.randint(lo, hi)
                cand = "".join(self.rng.choice(charset) for _ in range(L))
                if cand and cand not in seen:
                    tokens.append(cand)
                    seen.add(cand)
                if attempts > 100_000:
                    raise RuntimeError("Failed to generate enough unique random tokens; enlarge charset/lengths.")
            return tokens

        # ---- build source vocabularies for selecting pairs in episodes (keys_vocab/values_vocab) ----
        if keys_vocab is not None:
            seen = set()
            keys_vocab = [t for t in keys_vocab if (t not in reserved) and (t not in seen and not seen.add(t))]
            if len(keys_vocab) < n_pairs:
                raise ValueError("Provided keys_vocab has fewer unique tokens than n_pairs.")
        if values_vocab is not None:
            seen = set()
            values_vocab = [t for t in values_vocab if (t not in reserved) and (t not in seen and not seen.add(t))]

        # if vocabularies not provided — take from universe according to modes
        if keys_vocab is None or values_vocab is None:
            if deterministic_vocab:
                key_universe = det_tokens_for_range(key_token_len_range)
                val_universe = det_tokens_for_range(value_token_len_range)

                if randomize_pairs:
                    # randomly select pairs (but from fixed universe)
                    if len(key_universe) < n_pairs:
                        raise ValueError("Not enough deterministic tokens for keys.")
                    keys_vocab = self.rng.sample(key_universe, n_pairs)

                    key_set = set(keys_vocab)
                    val_candidates = [t for t in val_universe if t not in key_set]
                    if len(val_candidates) < n_pairs:
                        raise ValueError("Not enough deterministic tokens for values after excluding keys.")
                    values_vocab = self.rng.sample(val_candidates, n_pairs)
                else:
                    # take first n_pairs (fixed; pairs not randomized)
                    if len(key_universe) < n_pairs:
                        raise ValueError("Not enough deterministic tokens for keys.")
                    keys_vocab = key_universe[:n_pairs]

                    key_set = set(keys_vocab)
                    val_candidates = [t for t in val_universe if t not in key_set]
                    if len(val_candidates) < n_pairs:
                        raise ValueError("Not enough deterministic tokens for values after excluding keys.")
                    values_vocab = val_candidates[:n_pairs]
            else:
                # completely random vocabularies (not fixed universe/order)
                if keys_vocab is None:
                    keys_vocab = random_tokens(
                        need=n_pairs, charset=key_charset, length_range=key_token_len_range, avoid=reserved
                    )
                avoid_for_values = reserved | set(keys_vocab)
                if values_vocab is None:
                    values_vocab = random_tokens(
                        need=n_pairs, charset=value_charset, length_range=value_token_len_range, avoid=avoid_for_values
                    )

        # final checks
        if set(keys_vocab) & set(values_vocab):
            raise ValueError("keys_vocab and values_vocab must be disjoint.")
        if len(keys_vocab) < n_pairs or len(values_vocab) < n_pairs:
            raise ValueError("Not enough tokens in keys_vocab/values_vocab for n_pairs.")

        self.keys_vocab = list(keys_vocab)
        self.values_vocab = list(values_vocab)

        # ---- build env.vocab (observation/action space) ----
        if deterministic_vocab and full_universe_vocab:
            U_keys = det_tokens_for_range(key_token_len_range)
            U_vals = det_tokens_for_range(value_token_len_range)
            # combine in stable order: first U_keys, then add from U_vals everything not in U_keys
            universe = U_keys + [t for t in U_vals if t not in set(U_keys)]
            self.vocab = self.SPECIAL + universe
        else:
            # compact vocabulary: only special + selected keys/values
            self.vocab = self.SPECIAL + self.keys_vocab + self.values_vocab

        self.token_to_id = {tok: i for i, tok in enumerate(self.vocab)}
        self.id_to_token = {i: tok for tok, i in self.token_to_id.items()}

        # gym spaces
        self.observation_space = spaces.Discrete(len(self.vocab))
        self.action_space = spaces.Discrete(len(self.vocab))

        # episode state
        self._tokens: List[int] = []
        self._ptr: int = 0
        self._query_key: Optional[str] = None
        self._mapping: Dict[str, str] = {}

    # ---------- helpers ----------
    def _tok(self, s: str) -> int:
        return self.token_to_id[s]

    def _append_full_pair_tokens(self, stream: List[str], key: str):
        """Add tokens for a complete pair: ! key : value !"""
        stream += ["!", key, ":", self._mapping[key], "!"]

    def _build_after_pairs(self) -> List[str]:
        # sample n_pairs unique keys and values from source vocabularies
        keys = self.rng.sample(self.keys_vocab, self.n_pairs)
        values = self.rng.sample(self.values_vocab, self.n_pairs)
        self._mapping = {k: v for k, v in zip(keys, values)}

        min_p, max_p = self.prefix_pairs_range
        shown_pairs = self.rng.randint(min_p, max_p)
        shown_order = self.rng.sample(keys, shown_pairs)

        stream: List[str] = []
        for k in shown_order:
            self._append_full_pair_tokens(stream, k)

        self._query_key = self.rng.choice(shown_order) if self.query_from_any_shown else shown_order[-1]
        stream += ["!", self._query_key, ":", "shot"]
        return stream

    def _build_after_any_colon(self) -> List[str]:
        keys = self.rng.sample(self.keys_vocab, self.n_pairs)
        values = self.rng.sample(self.values_vocab, self.n_pairs)
        self._mapping = {k: v for k, v in zip(keys, values)}

        min_p, max_p = self.prefix_pairs_range
        min_p = max(1, min_p)  # need at least one k:v to have something to recall
        shown_pairs = self.rng.randint(min_p, max_p)

        shown_order = self.rng.sample(keys, shown_pairs)

        stream: List[str] = []
        for k in shown_order:
            self._append_full_pair_tokens(stream, k)

        # query key from shown ones
        self._query_key = self.rng.choice(shown_order) if self.query_from_any_shown else shown_order[-1]

        # optional additional pairs before repeated appearance of key
        remaining_keys = [k for k in keys if k not in shown_order]
        extra_cap = min(self.max_extra_pairs_before_shot, len(remaining_keys))
        extra_pairs = self.rng.randint(0, extra_cap)
        self.rng.shuffle(remaining_keys)
        for k in remaining_keys[:extra_pairs]:
            self._append_full_pair_tokens(stream, k)

        # repeated show of query_key, but instead of value → 'shot'
        stream += ["!", self._query_key, ":", "shot"]
        return stream

    def _build_episode(self):
        if self.shot_mode == "after_pairs":
            stream = self._build_after_pairs()
        else:
            stream = self._build_after_any_colon()
        self._tokens = [self._tok(s) for s in stream]
        self._ptr = 0

    # ---------- Gym API ----------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.rng.seed(seed)
        self._build_episode()
        obs = self._tokens[self._ptr]
        info = {
            "mapping": self._mapping.copy(),
            "query_key": self._query_key,
            "vocab": self.vocab,
        }
        return obs, info

    def step(self, action: int):
        assert 0 <= self._ptr < len(self._tokens), "Episode finished. Call reset()."

        cur_tok_id = self._tokens[self._ptr]
        cur_tok = self.id_to_token[cur_tok_id]

        reward = 0.0
        terminated = False
        truncated = False

        if cur_tok == "shot":
            correct_value = self._mapping[self._query_key]
            reward = 1.0 if action == self._tok(correct_value) else 0.0
            terminated = True

        self._ptr += 1
        if self._ptr >= len(self._tokens):
            terminated = True

        obs = self._tok("pass") if (terminated or truncated) else self._tokens[self._ptr]
        info = {
            "query_key": self._query_key,
            "correct_value": self._mapping[self._query_key],
            "was_shot_step": (cur_tok == "shot"),
        }
        return obs, reward, terminated, truncated, info

    # ---------- utils ----------
    def decode_stream(self) -> List[str]:
        return [self.id_to_token[t] for t in self._tokens]

    def render(self):
        print(" ".join(self.decode_stream()))
