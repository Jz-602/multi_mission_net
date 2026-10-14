import os
from argparse import Namespace

DEFAULT_FH_WAYMO_ROOT = os.environ.get("WAYMO_DATA_ROOT", "")
DEFAULT_FH_WAYMO_SEQUENCE_NAME = "16102220208346880_1420_000_1440_000"


def BuildTimestampFile(data_dir, timestamp, suffix):
    timestamp = int(timestamp)
    candidate_names = [
        f"{timestamp}{suffix}",
        f"{timestamp:013d}{suffix}",
    ]
    for candidate_name in candidate_names:
        candidate_filename = os.path.join(data_dir, candidate_name)
        if os.path.exists(candidate_filename):
            return candidate_filename
    return os.path.join(data_dir, candidate_names[0])


def FindExistingFile(candidate_filenames):
    for candidate_filename in candidate_filenames:
        if os.path.exists(candidate_filename):
            return candidate_filename
    return candidate_filenames[0] if candidate_filenames else ""


def ListFhSequenceDirs(data_path):
    if not os.path.isdir(data_path):
        return []
    sequence_dirs = []
    for item_name in sorted(os.listdir(data_path)):
        candidate_dir = os.path.join(data_path, item_name)
        output_dir = os.path.join(candidate_dir, "output")
        if os.path.isdir(output_dir) and (
            os.path.isdir(os.path.join(output_dir, "pcd"))
            or os.path.isdir(os.path.join(output_dir, "SemanticSegmentation"))
        ):
            sequence_dirs.append(candidate_dir)
    return sequence_dirs


def ResolveFhOutputDir(data_path, sequence_name):
    if sequence_name:
        candidate_root = os.path.join(data_path, sequence_name)
        if os.path.isdir(os.path.join(candidate_root, "output")):
            return candidate_root, os.path.join(candidate_root, "output")
        raise FileNotFoundError(f"FH sequence not found: {candidate_root}")

    if os.path.basename(os.path.normpath(data_path)) == "output":
        return os.path.dirname(os.path.normpath(data_path)), data_path

    output_dir = os.path.join(data_path, "output")
    if os.path.isdir(output_dir):
        return data_path, output_dir

    sequence_dirs = ListFhSequenceDirs(data_path)
    if len(sequence_dirs) == 1:
        sequence_root = sequence_dirs[0]
        return sequence_root, os.path.join(sequence_root, "output")
    if len(sequence_dirs) > 1:
        example_names = ", ".join(os.path.basename(item) for item in sequence_dirs[:5])
        raise ValueError(
            "data_path points to a FH multi-sequence root. "
            f"Please pass sequence_name=<one sequence>, for example: {example_names}"
        )
    raise FileNotFoundError(f"Cannot resolve FH output dir from data_path: {data_path}")


def ResolveTerrainOnlyDataPaths(args, b_require_semantic=True, b_require_pose=True):
    data_path = str(getattr(args, "data_path", "")).strip()
    sequence_name = str(getattr(args, "sequence_name", "")).strip()
    pcd_dir = str(getattr(args, "pcd_dir", "")).strip()
    semantic_label_dir = str(getattr(args, "semantic_label_dir", "")).strip()
    pose_filename = str(getattr(args, "pose_filename", "")).strip()

    sequence_root, output_dir = ResolveFhOutputDir(data_path, sequence_name)
    if not pcd_dir:
        pcd_dir = os.path.join(output_dir, "pcd")
    if not semantic_label_dir:
        semantic_label_dir = os.path.join(output_dir, "SemanticSegmentation")
    if not pose_filename:
        pose_filename = FindExistingFile(
            [
                os.path.join(output_dir, "pose_slam_euler_waymo.txt"),
                os.path.join(output_dir, "pose_slam_euler.txt"),
                os.path.join(output_dir, "pose_global.txt"),
            ]
        )

    if not os.path.isdir(pcd_dir):
        raise FileNotFoundError(f"pcd dir not found: {pcd_dir}")
    if bool(b_require_semantic) and not os.path.isdir(semantic_label_dir):
        raise FileNotFoundError(f"semantic label dir not found: {semantic_label_dir}")
    if bool(b_require_pose) and not os.path.exists(pose_filename):
        raise FileNotFoundError(f"pose file not found: {pose_filename}")

    return Namespace(
        layout="fh_waymo",
        data_path=data_path,
        sequence_name=sequence_name,
        sequence_root=sequence_root,
        pcd_dir=pcd_dir,
        semantic_label_dir=semantic_label_dir,
        pose_filename=pose_filename,
    )


def ApplyResolvedDataPaths(args, data_paths):
    args.data_path = data_paths.sequence_root
    args.sequence_name = ""
    args.pcd_dir = data_paths.pcd_dir
    args.semantic_label_dir = data_paths.semantic_label_dir
    args.pose_filename = data_paths.pose_filename
    args.data_layout = data_paths.layout


def BuildDefaultTerrainDir(args):
    data_paths = ResolveTerrainOnlyDataPaths(args, b_require_semantic=False, b_require_pose=False)
    return os.path.join(data_paths.sequence_root, "output", "label", "ground_truth")
