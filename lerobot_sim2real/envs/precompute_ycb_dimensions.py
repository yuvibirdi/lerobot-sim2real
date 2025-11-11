import os
import json
import numpy as np
from typing import Optional, List
from pathlib import Path

from mani_skill.utils import download_asset, assets


def filter_ycb_objects_by_dimensions(
    max_x: Optional[float] = None,
    max_y: Optional[float] = None,
    max_z: Optional[float] = None,
    download_if_missing: bool = True
) -> List[str]:
    asset_dir = Path(assets.DATA_SOURCES["ycb"].output_dir)
    ycb_info_path = asset_dir / "assets/mani_skill2_ycb/info_pick_v0.json" 
    if not os.path.exists(ycb_info_path) and download_if_missing:
       print("YCB info file not found. Downloading YCB assets...")
       download_asset.download(data_source=assets.DATA_SOURCES["ycb"], non_interactive=True)
       download_path = assets.DATA_SOURCES["ycb"].output_dir
       print("Assets downloaded to:", download_path)
       print("Download complete")
    if not os.path.exists(ycb_info_path):
        raise FileNotFoundError(f"YCB info file not found at {ycb_info_path}") 

    with open(ycb_info_path, "r") as f:
        ycb_info = json.load(f)

    filtered_ids = []
    for model_id, info in ycb_info.items():
        if 'bbox' not in info:
            continue
        bbox_min = np.array(info['bbox']['min'])
        bbox_max = np.array(info['bbox']['max'])
        dimensions = bbox_max - bbox_min
        passes = True
        if max_x is not None and dimensions[0] > max_x:
            passes = False
        if max_y is not None and dimensions[1] > max_y:
            passes = False
        if max_z is not None and dimensions[2] > max_z:
            passes = False
        if passes:
            filtered_ids.append(model_id)
    return filtered_ids

if __name__ == "__main__":
    print("example usage:\n")
    filtered_ycb_ids = filter_ycb_objects_by_dimensions()
    print("Filtered YCB Object IDs:", filtered_ycb_ids)