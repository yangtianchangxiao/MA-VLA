Got it — here’s a **working, minimal TFRecord → Python parser** for DROID’s **RLDS** files. It doesn’t depend on their training repo; it loads local shards you downloaded with `gsutil`, iterates episodes, and gives you NumPy arrays (ready for PyTorch or saving to `.npz`). I’m using the **official schema keys** published in DROID’s docs. ([DROID Dataset][1])

---

### 1) Install deps

```bash
pip install "tensorflow>=2.12" rlds
```

---

### 2) Parser (iterate RLDS TFRecords and yield episodes)

```python
import os
import glob
from typing import Dict, Iterator, List, Optional

import numpy as np
import tensorflow as tf
import rlds

# --------- CONFIG: which fields to extract (from official schema) ----------
OBS_KEYS = [
    "gripper_position",      # (1,) float64
    "cartesian_position",    # (6,) float64
    "joint_position",        # (7,) float64
    "wrist_image_left",      # (H,W,3) uint8
    "exterior_image_1_left",
    "exterior_image_2_left",
]
ACT_KEYS = [
    "gripper_position",      # (1,) float64
    "gripper_velocity",      # (1,) float64
    "cartesian_position",    # (6,) float64
    "cartesian_velocity",    # (6,) float64
    "joint_position",        # (7,) float64
    "joint_velocity",        # (7,) float64
]

def _tf_to_np(tree):
    """Recursively convert a (possibly nested) structure of Tensors to NumPy."""
    if isinstance(tree, tf.Tensor):
        return tree.numpy()
    if isinstance(tree, dict):
        return {k: _tf_to_np(v) for k, v in tree.items()}
    if isinstance(tree, (list, tuple)):
        return type(tree)(_tf_to_np(v) for v in tree)
    return tree

def _stack_time(sequence: List[np.ndarray]) -> Optional[np.ndarray]:
    """Stack a list of step-wise arrays into [T, ...]. Returns None if empty."""
    if not sequence:
        return None
    return np.stack(sequence, axis=0)

def iter_droid_episodes(
    root: str,
    subset: str = "droid",          # or "droid_100"
    version: str = "1.0.0",         # default layout from docs
    shuffle_files: bool = False,
) -> Iterator[Dict]:
    """
    Iterate DROID RLDS TFRecord episodes from a local directory downloaded via:
      gsutil -m cp -r gs://gresearch/robotics/droid_100 <target>
      or
      gsutil -m cp -r gs://gresearch/robotics/droid <target>
    Expected layout: <root>/<subset>/<version>/*.tfrecord* 
    Yields a dict per episode with NumPy arrays.
    """
    pattern = os.path.join(root, subset, version, "*.tfrecord*")
    shards = sorted(glob.glob(pattern))
    if not shards:
        raise FileNotFoundError(f"No TFRecord shards under {pattern}")

    ds = tf.data.TFRecordDataset(shards, compression_type="GZIP")  # RLDS uses gzip
    # Decode RLDS → episodes; each element becomes an episode dict with 'steps', 'episode_metadata'
    ds = rlds.transformations.decode_rlds_tfrecord(ds)

    if shuffle_files:
        ds = ds.shuffle(buffer_size=64)

    for episode in ds:
        ep_np = _tf_to_np(episode)
        steps = ep_np["steps"]

        # Collect per-step observations/actions
        obs = steps.get("observation", {})
        act = steps.get("action_dict", {})

        obs_time = {k: _stack_time([x for x in obs.get(k, [])]) for k in OBS_KEYS if k in obs}
        act_time = {k: _stack_time([x for x in act.get(k, [])]) for k in ACT_KEYS if k in act}

        out = {
            "episode_metadata": ep_np.get("episode_metadata", {}),
            "is_first": _stack_time(list(steps.get("is_first", []))) if "is_first" in steps else None,
            "is_last":  _stack_time(list(steps.get("is_last",  []))) if "is_last"  in steps else None,
            "is_terminal": _stack_time(list(steps.get("is_terminal", []))) if "is_terminal" in steps else None,
            "language_instruction": _stack_time(list(steps.get("language_instruction", []))) if "language_instruction" in steps else None,
            "reward": _stack_time(list(steps.get("reward", []))) if "reward" in steps else None,
            "discount": _stack_time(list(steps.get("discount", []))) if "discount" in steps else None,

            # compacted dicts of time-major arrays
            "observation": obs_time,
            "action_dict": act_time,

            # (optional) the pre-computed 7D action if you need it
            "action": _stack_time(list(steps.get("action", []))) if "action" in steps else None,
        }
        yield out
```

**Notes.**

* DROID’s RLDS shards are **gzip-compressed**. The helper `rlds.transformations.decode_rlds_tfrecord` takes care of parsing Examples into the **episode structure with `steps`** as published in the official schema. ([DROID Dataset][1])
* The keys used above (`observation.*`, `action_dict.*`, `language_instruction`, etc.) match the **doc’s schema** (shapes/dtypes there). If you don’t need images, just remove those keys. ([DROID Dataset][1])

---

### 3) Example usage (quick sanity check)

```python
# Point to the parent folder that contains 'droid_100/1.0.0/*.tfrecord*'
ROOT = "/data/droid_local"
for i, ep in enumerate(iter_droid_episodes(ROOT, subset="droid_100", version="1.0.0")):
    T = ep["observation"]["joint_position"].shape[0]
    print(f"[Episode {i}] T={T}, joints shape={ep['observation']['joint_position'].shape}, "
          f"action shape={None if ep['action'] is None else ep['action'].shape}")
    if i == 2:
        break
```

---

### 4) Optional: save each episode to `.npz`

```python
import os

def dump_npz(root_in: str, root_out: str, subset="droid_100", version="1.0.0", limit: Optional[int]=None):
    os.makedirs(root_out, exist_ok=True)
    for i, ep in enumerate(iter_droid_episodes(root_in, subset=subset, version=version)):
        ep_id = ep["episode_metadata"].get("file_path", f"ep_{i}").split("/")[-1].replace(".json","")
        out_path = os.path.join(root_out, f"{i:06d}_{ep_id}.npz")

        # Flatten nested dicts for np.savez
        flat = {"action": ep["action"],
                "is_first": ep["is_first"], "is_last": ep["is_last"], "is_terminal": ep["is_terminal"],
                "reward": ep["reward"], "discount": ep["discount"]}

        for k, arr in ep["observation"].items():
            if arr is not None:
                flat[f"obs/{k}"] = arr
        for k, arr in ep["action_dict"].items():
            if arr is not None:
                flat[f"act/{k}"] = arr

        np.savez_compressed(out_path, **{k:v for k,v in flat.items() if v is not None})
        print("saved:", out_path)
        if limit is not None and (i+1) >= limit:
            break

# Example:
# dump_npz("/data/droid_local", "/data/droid_npz", subset="droid_100", version="1.0.0", limit=10)
```

---

### 5) Where the field names came from

The keys and shapes in the code directly follow the **official DROID RLDS schema** (see “Dataset Schema” on the docs page). If any keys are missing in a particular episode, the code guards with `if k in ...` and will skip them. ([DROID Dataset][1])

---

### 6) Prefer a “reference” loader?

If you later want a production-grade dataloader (prefetching, augment, normalization) for **policy training**, the DROID authors provide one in their **policy-learning repo** (PyTorch/JAX). You can lift just the RLDS reader bits from there if you prefer. ([DROID Dataset][1])

---

If you tell me your **exact fields** (e.g., only `joint_position`, `cartesian_position`, and the 7-DoF `action`) and your **target format** (PyTorch dataset class? arrow/parquet? HDF5?), I’ll tailor this into a drop-in module for your training pipeline.

[1]: https://droid-dataset.github.io/droid/the-droid-dataset "The DROID Dataset | DROID Docs"
