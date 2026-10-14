"""Restore the manifest's fixed targets and full history sequences from server infos.

Records are copied intact: pose [4, 4] maps vehicle coordinates to world coordinates;
objects retain the source vehicle-coordinate box fields. No labels or poses change.
"""
import argparse
import json
import os
import pickle
from pathlib import Path

from build_label_manifest import HashTextLines


MODEL_DIR = Path(__file__).resolve().parents[1]
INFO_PREFIX = "three_task_20260930_infos"


def LoadPool(data_root, filenames):
    pool = {}
    for filename in filenames:
        with (data_root / filename).open("rb") as stream:
            infos = pickle.load(stream)
        for info in infos:
            token = str(info["token"])
            if token in pool:
                if pickle.dumps(pool[token]) != pickle.dumps(info):
                    raise ValueError(f"conflicting source records: {token}")
            else:
                pool[token] = info
    return pool


def SelectTargets(pool, split_manifest):
    targets = split_manifest["targets"]
    tokens = [str(target["token"]) for target in targets]
    if len(set(tokens)) != len(tokens):
        raise ValueError("duplicate target tokens in manifest")
    if len(tokens) != split_manifest["target_count"]:
        raise ValueError("manifest target_count mismatch")
    if HashTextLines(tokens) != split_manifest["token_sha256"]:
        raise ValueError("manifest token_sha256 mismatch")
    selected_infos = []
    for target in targets:
        info = pool[str(target["token"])]
        if str(info["sequence_name"]) != str(target["sequence_name"]):
            raise ValueError(f"sequence mismatch: {target['token']}")
        if int(info["frame_timestamp_micros"]) // 1000 != int(target["timestamp_ms"]):
            raise ValueError(f"timestamp mismatch: {target['token']}")
        selected_infos.append(info)
    return selected_infos


def SelectFullInfos(pool, target_infos):
    sequence_names = {str(info["sequence_name"]) for info in target_infos}
    full_infos = [info for info in pool.values() if str(info["sequence_name"]) in sequence_names]
    for target in target_infos:
        full_info = pool[str(target["token"])]
        for field in ("sequence_name", "frame_timestamp_micros"):
            if full_info[field] != target[field]:
                raise ValueError(f"full history {field} mismatch: {target['token']}")
    return full_infos


def WriteInfos(filename, infos):
    payload = pickle.dumps(infos, protocol=pickle.HIGHEST_PROTOCOL)
    if filename.exists():
        if filename.read_bytes() != payload:
            raise FileExistsError(f"refusing to replace different existing infos: {filename}")
        return "reused"
    with filename.open("xb") as stream:
        stream.write(payload)
    return "created"


def Main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=os.environ.get("WAYMO_DATA_ROOT", ""))
    parser.add_argument(
        "--manifest", type=Path,
        default=MODEL_DIR / "data_manifests/waymo_three_task_20260930.json",
    )
    args = parser.parse_args()
    if not args.data_root:
        parser.error("set WAYMO_DATA_ROOT or --data-root")
    data_root = Path(args.data_root).expanduser().resolve()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    target_pool = LoadPool(data_root, (
        "LYF_waymo_infos_training_semantic.pkl", "LYF_waymo_infos_validation_semantic.pkl",
    ))
    full_pool = LoadPool(data_root, (
        "waymo_infos_training_point_only.pkl", "waymo_infos_validation_point_only.pkl",
    ))
    selected = {split: SelectTargets(target_pool, manifest["splits"][split]) for split in ("train", "val")}
    sequence_sets = {split: {str(info["sequence_name"]) for info in infos} for split, infos in selected.items()}
    if sequence_sets["train"] & sequence_sets["val"]:
        raise ValueError("train/val sequence overlap")
    # Full history includes every available point-cloud frame in the selected sequences.
    full_selected = {split: SelectFullInfos(full_pool, infos) for split, infos in selected.items()}
    for split, suffix in (("train", "training"), ("val", "validation")):
        for kind, infos in (("semantic", selected[split]), ("point_only", full_selected[split])):
            filename = data_root / f"{INFO_PREFIX}_{suffix}_{kind}.pkl"
            status = WriteInfos(filename, infos)
            print(json.dumps({
                "split": split, "kind": kind, "count": len(infos),
                "sequence_count": len(sequence_sets[split]), "status": status,
                "token_sha256": HashTextLines([info["token"] for info in infos]),
                "path": str(filename),
            }))


if __name__ == "__main__":
    Main()
