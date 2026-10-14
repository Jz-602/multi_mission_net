"""Build a content manifest for configured three-task target labels.

The tool only streams existing files. It does not load complete label arrays or
modify/re-export the dataset. A full local manifest can therefore be reused by
single-frame, temporal, and ablation runs that select the same targets.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
import pickle
from pathlib import Path

import numpy as np
import yaml


HASH_CHUNK_SIZE = 8 * 1024 * 1024
TERRAIN_FILENAMES = (
    "terrain_z_inpainted.bin",
    "terrain_inpainted_mask.bin",
    "terrain_train_weight.bin",
    "terrain_inpaint_distance.bin",
)


def ParseArgs():
    parser = argparse.ArgumentParser(
        description="Stream SHA256 hashes for all labels selected by a three-task config."
    )
    parser.add_argument("--config", required=True, help="Single-frame or temporal YAML config.")
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON filename, normally <run-root>/label_manifest.json.",
    )
    parser.add_argument(
        "--data-root",
        default=os.environ.get("WAYMO_DATA_ROOT", ""),
        help="Dataset root. Defaults to WAYMO_DATA_ROOT, then common_variables.root_path.",
    )
    parser.add_argument(
        "--split",
        choices=("both", "train", "val"),
        default="both",
        help="Configured target split(s) to include.",
    )
    parser.add_argument(
        "--max-targets-per-split",
        type=int,
        default=None,
        help="Optional smoke-check bound. Omit for a formal full manifest.",
    )
    return parser.parse_args()


def MergeConfig(config, update):
    for key, value in update.items():
        if isinstance(value, dict):
            if key not in config or not isinstance(config[key], dict):
                config[key] = {}
            MergeConfig(config[key], value)
        else:
            config[key] = os.path.expandvars(value) if isinstance(value, str) else value
    return config


def LoadConfig(config_filename):
    config_path = Path(config_filename).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream)
    if not isinstance(loaded, dict):
        raise TypeError(f"config must contain a mapping: {config_path}")
    base_filename = loaded.pop("_BASE_CONFIG_", None)
    config = {}
    if base_filename is not None:
        config = LoadConfig(config_path.parent / str(base_filename))
    return MergeConfig(config, loaded)


def HashFile(filename):
    file_hash = hashlib.sha256()
    with open(filename, "rb") as stream:
        while True:
            chunk = stream.read(HASH_CHUNK_SIZE)
            if not chunk:
                break
            file_hash.update(chunk)
    return file_hash.hexdigest()


def HashTextLines(lines):
    return hashlib.sha256("\n".join(str(line) for line in lines).encode("utf-8")).hexdigest()


def LoadSelectedInfos(data_root, dataset_config, run_config, split):
    info_path = Path(os.path.expandvars(str(dataset_config["info_path"]))).expanduser()
    if not info_path.is_absolute():
        info_path = data_root / info_path
    info_path = info_path.resolve()
    with info_path.open("rb") as stream:
        infos = pickle.load(stream)
    if not isinstance(infos, list) or len(infos) == 0:
        raise TypeError(f"info file must contain a non-empty list: {info_path}")

    sample_fraction = float(dataset_config.get("sample_fraction", 1.0))
    if not 0.0 < sample_fraction <= 1.0:
        raise ValueError(f"{split}: sample_fraction must be in (0, 1], got {sample_fraction}")
    sample_seed = int(dataset_config.get("sample_seed", 0))
    b_dynamic_epoch_sampling = bool(dataset_config.get("b_dynamic_epoch_sampling", False))
    if b_dynamic_epoch_sampling:
        # 每轮抽样只改变训练顺序与成员；清单绑定原始 infos 的完整候选标签。
        selected_infos = infos
    else:
        sample_count = max(1, int(math.ceil(len(infos) * sample_fraction)))
        random_generator = np.random.default_rng(sample_seed)
        sampled_indices = np.sort(
            random_generator.choice(len(infos), size=sample_count, replace=False)
        )
        selected_infos = [infos[int(sample_index)] for sample_index in sampled_indices]

    limit = run_config.get(f"{split}_samples")
    offset = int(run_config.get(f"{split}_offset", 0))
    if limit is not None:
        selected_infos = selected_infos[offset:offset + int(limit)]
    if len(selected_infos) == 0:
        raise ValueError(f"{split}: configured target selection is empty")
    return selected_infos, info_path, sample_fraction, sample_seed, offset, limit


def BuildTargetFiles(data_root, dataset_config, info):
    sequence_name = str(info["sequence_name"])
    timestamp_ms = int(info["frame_timestamp_micros"]) // 1000
    terrain_label_path = Path(
        str(dataset_config.get("terrain_label_path", "output/label/ground_truth/frames"))
    )
    occ_label_path = Path(
        str(dataset_config.get("occ_label_path", "output/label/background_occ"))
    )
    if terrain_label_path.is_absolute() or occ_label_path.is_absolute():
        raise ValueError("terrain_label_path and occ_label_path must be relative")
    frame_path = data_root / sequence_name / terrain_label_path / f"{timestamp_ms:013d}"
    terrain_path = frame_path / "base" / "terrain_completion"
    files = [("frame_meta", frame_path / "frame_meta.json")]
    files.extend((filename.removesuffix(".bin"), terrain_path / filename) for filename in TERRAIN_FILENAMES)
    files.append(("background_occ", data_root / sequence_name / occ_label_path / f"{timestamp_ms}.npz"))
    return sequence_name, timestamp_ms, files


def BuildSplitManifest(data_root, dataset_config, run_config, split, max_targets):
    infos, info_path, sample_fraction, sample_seed, offset, limit = LoadSelectedInfos(
        data_root, dataset_config, run_config, split
    )
    if max_targets is not None:
        if max_targets <= 0:
            raise ValueError("--max-targets-per-split must be positive")
        infos = infos[:max_targets]

    file_records = []
    aggregate_hash = hashlib.sha256()
    tokens = []
    total_bytes = 0
    for target_index, info in enumerate(infos):
        token = str(info["token"])
        sequence_name, timestamp_ms, files = BuildTargetFiles(data_root, dataset_config, info)
        tokens.append(token)
        target_files = {}
        for field_name, filename in files:
            filename = filename.resolve()
            if not filename.is_file():
                raise FileNotFoundError(
                    f"{split} target {target_index} {token} is missing {field_name}: {filename}"
                )
            relative_path = filename.relative_to(data_root).as_posix()
            size_bytes = filename.stat().st_size
            sha256 = HashFile(filename)
            target_files[field_name] = {
                "path": relative_path,
                "size_bytes": size_bytes,
                "sha256": sha256,
            }
            aggregate_hash.update(relative_path.encode("utf-8"))
            aggregate_hash.update(b"\0")
            aggregate_hash.update(sha256.encode("ascii"))
            aggregate_hash.update(b"\n")
            total_bytes += size_bytes
        file_records.append({
            "token": token,
            "sequence_name": sequence_name,
            "timestamp_ms": timestamp_ms,
            "files": target_files,
        })

    return {
        "info_path": str(info_path),
        "sample_fraction": sample_fraction,
        "sample_seed": sample_seed,
        "manifest_scope": ("full_candidate_pool" if dataset_config.get("b_dynamic_epoch_sampling", False)
                           else "static_selected_targets"),
        "b_dynamic_epoch_sampling": bool(dataset_config.get("b_dynamic_epoch_sampling", False)),
        "epoch_sample_fraction": (sample_fraction if dataset_config.get("b_dynamic_epoch_sampling", False)
                                  else None),
        "run_offset": offset,
        "run_limit": limit,
        "target_count": len(file_records),
        "token_sha256": HashTextLines(tokens),
        "label_content_sha256": aggregate_hash.hexdigest(),
        "total_label_bytes": total_bytes,
        "targets": file_records,
    }


def WriteJsonAtomic(filename, value):
    path = Path(filename).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    temporary_path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary_path, path)


def Main():
    args = ParseArgs()
    config_path = Path(args.config).expanduser().resolve()
    config = LoadConfig(config_path)
    configured_root = config.get("common_variables", {}).get("root_path", "")
    data_root_text = args.data_root or configured_root
    if not data_root_text:
        raise ValueError("Set WAYMO_DATA_ROOT, --data-root, or common_variables.root_path")
    data_root = Path(os.path.expandvars(str(data_root_text))).expanduser().resolve()
    if not data_root.is_dir():
        raise NotADirectoryError(f"data root does not exist: {data_root}")

    requested_splits = ("train", "val") if args.split == "both" else (args.split,)
    data_config = config.get("data", {})
    run_config = config.get("run", {})
    split_manifests = {}
    for split in requested_splits:
        dataset_config = data_config.get(f"{split}_dataset")
        if not isinstance(dataset_config, dict):
            raise KeyError(f"config has no data.{split}_dataset mapping")
        split_manifests[split] = BuildSplitManifest(
            data_root,
            dataset_config,
            run_config,
            split,
            args.max_targets_per_split,
        )

    manifest = {
        "schema_version": 1,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "config": {
            "path": str(config_path),
            "sha256": HashFile(config_path),
        },
        "data_root": str(data_root),
        "source_version_uncertain": True,
        "source_version_note": (
            "Content hashes identify the selected existing label bytes; current frame metadata "
            "does not record the terrain exporter commit."
        ),
        "b_partial_manifest": args.max_targets_per_split is not None,
        "splits": split_manifests,
    }
    WriteJsonAtomic(args.output, manifest)
    print(json.dumps({
        "output": str(Path(args.output).expanduser().resolve()),
        "splits": {
            split: {
                "target_count": value["target_count"],
                "total_label_bytes": value["total_label_bytes"],
                "label_content_sha256": value["label_content_sha256"],
            }
            for split, value in split_manifests.items()
        },
        "source_version_uncertain": True,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    Main()
