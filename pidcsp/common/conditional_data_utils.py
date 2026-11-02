import re
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple


PROPERTY_KEYS = [
    "formation_energy_per_atom",
    "band_gap",
]

DEFAULT_LOG_TRANSFORM_PROPS = {"band_gap"}


def parse_elements_to_chemsys(elements_raw: str) -> str:


    if not elements_raw or elements_raw.strip() in ('', 'nan', 'None'):
        return "Unknown"


    list_pattern = re.findall(r"['\"]([A-Z][a-z]?)['\"]", elements_raw)
    if list_pattern:
        elems = sorted(set(list_pattern))
        return "-".join(elems)


    space_split = [e.strip() for e in elements_raw.split()
                   if re.match(r'^[A-Z][a-z]?$', e.strip())]
    if space_split:
        elems = sorted(set(space_split))
        return "-".join(elems)


    formula_pattern = re.findall(r'([A-Z][a-z]?)\d*', elements_raw)
    if formula_pattern:
        elems = sorted(set(formula_pattern))
        return "-".join(elems)

    return "Unknown"


def compute_property_statistics(
    cached_data: list,
    prop_keys: List[str] = None,
    log_transform_props: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:


    if prop_keys is None:
        prop_keys = PROPERTY_KEYS
    log_transform_props = set(
        DEFAULT_LOG_TRANSFORM_PROPS if log_transform_props is None
        else log_transform_props
    )

    stats = {}

    for key in prop_keys:
        values = []
        for d in cached_data:
            val = d.get(key, None)
            if val is not None and not _is_nan(val):
                try:
                    raw_value = float(val)
                    values.append(
                        float(np.log1p(max(raw_value, 0.0)))
                        if key in log_transform_props else raw_value
                    )
                except (TypeError, ValueError):
                    continue

        if len(values) == 0:
            print(f"[PropStats] ⚠️  {key}: 无有效数据，使用默认统计量")
            stats[key] = {
                "mean":  0.0,
                "std":   1.0,
                "min":   0.0,
                "max":   1.0,
                "count": 0,
                "log_transformed": key in log_transform_props,
            }
            continue

        arr = np.array(values, dtype=np.float64)
        std = float(arr.std())


        if std < 1e-8:
            std = 1.0
            print(f"[PropStats] ⚠️  {key}: std接近0，设为1.0")

        stats[key] = {
            "mean":  float(arr.mean()),
            "std":   std,
            "min":   float(arr.min()),
            "max":   float(arr.max()),
            "count": len(values),
            "log_transformed": key in log_transform_props,
            "space": "transformed",
        }

        print(f"[PropStats] {key}:")
        print(f"  count={len(values)}, mean={stats[key]['mean']:.4f}, "
              f"std={stats[key]['std']:.4f}, "
              f"min={stats[key]['min']:.4f}, max={stats[key]['max']:.4f}")

    return stats


def _is_nan(val) -> bool:

    try:
        import math
        value = float(val)
        return math.isnan(value) or math.isinf(value)
    except (TypeError, ValueError):
        return False


def normalize_property(
    raw_val,
    mean: float,
    std: float,
    clip_range: float = 5.0,
    log_transform: bool = False,
) -> Tuple[float, int]:


    if raw_val is None or _is_nan(raw_val):
        return 0.0, 1

    try:
        value = float(raw_val)
        if log_transform:
            value = float(np.log1p(max(value, 0.0)))
        normed = (value - mean) / (std + 1e-8)
        normed = float(np.clip(normed, -clip_range, clip_range))
        return normed, 0
    except (TypeError, ValueError):
        return 0.0, 1


def denormalize_property(
    normed_val: float,
    mean: float,
    std: float,
) -> float:


    return normed_val * std + mean


def save_prop_stats(stats: dict, path: str):

    import os
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    torch.save(stats, path)
    print(f"[PropStats] 已保存: {path}")


def load_prop_stats(path: str) -> dict:

    if not path or not __import__('os').path.exists(path):
        print(f"[PropStats] ⚠️  文件不存在: {path}，使用空统计量")
        return {}
    stats = torch.load(path, map_location='cpu')
    print(f"[PropStats] 已加载: {path}")
    for key, s in stats.items():
        print(f"  {key}: mean={s['mean']:.4f}, std={s['std']:.4f}, "
              f"count={s.get('count', 'N/A')}")
    return stats
