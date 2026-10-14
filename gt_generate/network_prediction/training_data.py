import json
import os
from argparse import Namespace

import numpy as np

from gt_generate.common.constants import INVALID_HEIGHT
from gt_generate.common.io_utils import EnsureDir, LoadPcdXyzi, ReadMat, WriteMat
from gt_generate.ground_truth.data_paths import BuildTimestampFile, ResolveTerrainOnlyDataPaths


def BuildTerrainOnlyInputArgs():
    return Namespace(
        input_point_stride=1,
        b_cache_observation_inputs=True,
        observation_input_subdir="input_observation_pillar_current_frame",
        hit_count_clip=10.0,
        low_quantile=0.2,
        high_quantile=0.8,
        low_band_delta_m=0.15,
        mid_band_delta_m=0.60,
        high_band_delta_m=1.20,
        default_input_field_names=[
            "obs_hit_count_norm",
            "obs_z_min",
            "obs_z_mean",
            "obs_z_std",
            "obs_z_low_q",
            "obs_z_high_q",
            "obs_z_max",
            "obs_z_span",
            "obs_low_band_count_norm",
            "obs_mid_band_count_norm",
            "obs_high_band_count_norm",
        ],
    )


def AddChannelDim(layer_map):
    return layer_map[np.newaxis, ...].astype(np.float32)


def SanitizeHeightMap(layer_map):
    output_map = layer_map.astype(np.float32).copy()
    output_map[output_map == INVALID_HEIGHT] = 0.0
    return output_map


class TerrainOnlyObservationInputBuilder:
    def __init__(self, terrain_dir, args):
        self.terrain_dir = terrain_dir
        self.args = args
        summary_filename = os.path.join(terrain_dir, "terrain_only_frame_gt_summary.json")
        if os.path.exists(summary_filename):
            with open(summary_filename, "r", encoding="utf-8") as fp:
                self.summary_data = json.load(fp)
            self.data_path = self.summary_data["data_path"]
        else:
            self.summary_data = {}
            self.data_path = str(getattr(args, "data_path", ""))
        if not self.data_path:
            raise ValueError("TerrainOnlyObservationInputBuilder requires terrain summary or args.data_path.")
        data_paths = ResolveTerrainOnlyDataPaths(
            Namespace(
                data_path=self.data_path,
                sequence_name=str(getattr(args, "sequence_name", "")),
                pcd_dir=str(getattr(args, "pcd_dir", "")),
                semantic_label_dir=str(getattr(args, "semantic_label_dir", "")),
                pose_filename=str(getattr(args, "pose_filename", "")),
            ),
            b_require_semantic=False,
            b_require_pose=False,
        )
        self.pcd_dir = data_paths.pcd_dir

    def BuildPcdFilename(self, timestamp):
        return BuildTimestampFile(self.pcd_dir, timestamp, ".pcd")

    def LocalToPatchIndex(self, local_x, local_y, frame_meta):
        local_x = np.asarray(local_x, dtype=np.float64)
        local_y = np.asarray(local_y, dtype=np.float64)
        row_idx = np.floor((float(frame_meta["forward_range_m"]) - local_x) / float(frame_meta["map_resolution_m"])).astype(np.int32)
        col_idx = np.floor((float(frame_meta["left_range_m"]) - local_y) / float(frame_meta["map_resolution_m"])).astype(np.int32)
        valid_mask = (
            (row_idx >= 0)
            & (row_idx < int(frame_meta["patch_rows"]))
            & (col_idx >= 0)
            & (col_idx < int(frame_meta["patch_cols"]))
        )
        safe_rows = np.clip(row_idx, 0, int(frame_meta["patch_rows"]) - 1)
        safe_cols = np.clip(col_idx, 0, int(frame_meta["patch_cols"]) - 1)
        return safe_rows, safe_cols, valid_mask

    def BuildObservationMaps(self, frame_meta):
        rows = int(frame_meta["patch_rows"])
        cols = int(frame_meta["patch_cols"])
        hit_count = np.zeros((rows, cols), dtype=np.float32)
        z_min = np.zeros((rows, cols), dtype=np.float32)
        z_mean = np.zeros((rows, cols), dtype=np.float32)
        z_std = np.zeros((rows, cols), dtype=np.float32)
        z_low_q = np.zeros((rows, cols), dtype=np.float32)
        z_high_q = np.zeros((rows, cols), dtype=np.float32)
        z_max = np.full((rows, cols), -np.inf, dtype=np.float32)
        z_span = np.zeros((rows, cols), dtype=np.float32)
        low_band_count = np.zeros((rows, cols), dtype=np.float32)
        mid_band_count = np.zeros((rows, cols), dtype=np.float32)
        high_band_count = np.zeros((rows, cols), dtype=np.float32)

        pcd_filename = self.BuildPcdFilename(frame_meta["timestamp"])
        if not os.path.exists(pcd_filename):
            return self.FinalizeObservationMaps(hit_count, z_min, z_mean, z_std, z_low_q, z_high_q, z_max, z_span, low_band_count, mid_band_count, high_band_count)
        raw_points_xyzi = LoadPcdXyzi(pcd_filename)
        sampled_points = raw_points_xyzi[:: max(1, int(self.args.input_point_stride)), :3]
        local_points = sampled_points
        row_idx, col_idx, valid_mask = self.LocalToPatchIndex(local_points[:, 0], local_points[:, 1], frame_meta)
        if not np.any(valid_mask):
            return self.FinalizeObservationMaps(hit_count, z_min, z_mean, z_std, z_low_q, z_high_q, z_max, z_span, low_band_count, mid_band_count, high_band_count)

        valid_rows = row_idx[valid_mask]
        valid_cols = col_idx[valid_mask]
        valid_z = local_points[valid_mask, 2].astype(np.float32)
        np.add.at(hit_count, (valid_rows, valid_cols), 1.0)
        np.maximum.at(z_max, (valid_rows, valid_cols), valid_z)
        self.FillVerticalObservationStats(
            valid_rows,
            valid_cols,
            valid_z,
            z_min,
            z_mean,
            z_std,
            z_low_q,
            z_high_q,
            z_max,
            z_span,
            low_band_count,
            mid_band_count,
            high_band_count,
        )
        return self.FinalizeObservationMaps(hit_count, z_min, z_mean, z_std, z_low_q, z_high_q, z_max, z_span, low_band_count, mid_band_count, high_band_count)

    def FillVerticalObservationStats(self, valid_rows, valid_cols, valid_z, z_min, z_mean, z_std, z_low_q, z_high_q, z_max, z_span, low_band_count, mid_band_count, high_band_count):
        flat_idx = valid_rows.astype(np.int64) * int(z_min.shape[1]) + valid_cols.astype(np.int64)
        sort_idx = np.argsort(flat_idx, kind="mergesort")
        flat_idx = flat_idx[sort_idx]
        valid_z = valid_z[sort_idx]
        split_idx = np.concatenate(([0], np.flatnonzero(np.diff(flat_idx)) + 1, [flat_idx.shape[0]]))
        for start_idx, end_idx in zip(split_idx[:-1], split_idx[1:]):
            cell_idx = int(flat_idx[start_idx])
            row_idx = cell_idx // int(z_min.shape[1])
            col_idx = cell_idx % int(z_min.shape[1])
            cell_z = valid_z[start_idx:end_idx]
            cell_z_min = float(np.min(cell_z))
            cell_z_low_q = float(np.quantile(cell_z, float(self.args.low_quantile)))
            cell_z_max = float(np.max(cell_z))
            z_min[row_idx, col_idx] = cell_z_min
            z_mean[row_idx, col_idx] = float(np.mean(cell_z))
            z_std[row_idx, col_idx] = float(np.std(cell_z))
            z_low_q[row_idx, col_idx] = cell_z_low_q
            z_high_q[row_idx, col_idx] = float(np.quantile(cell_z, float(self.args.high_quantile)))
            z_max[row_idx, col_idx] = cell_z_max
            z_span[row_idx, col_idx] = float(max(0.0, cell_z_max - cell_z_min))
            low_band_count[row_idx, col_idx] = float(np.sum(cell_z <= (cell_z_low_q + float(self.args.low_band_delta_m))))
            mid_lower = cell_z_low_q + float(self.args.low_band_delta_m)
            mid_upper = cell_z_low_q + float(self.args.mid_band_delta_m)
            mid_band_count[row_idx, col_idx] = float(np.sum((cell_z > mid_lower) & (cell_z < mid_upper)))
            high_band_count[row_idx, col_idx] = float(np.sum(cell_z >= (cell_z_low_q + float(self.args.high_band_delta_m))))

    def FinalizeObservationMaps(self, hit_count, z_min, z_mean, z_std, z_low_q, z_high_q, z_max, z_span, low_band_count, mid_band_count, high_band_count):
        hit_mask = hit_count > 0.0
        z_max[~hit_mask] = 0.0
        hit_count_norm = np.clip(np.log1p(hit_count) / np.log1p(float(self.args.hit_count_clip)), 0.0, 1.0)
        low_band_count_norm = np.clip(np.log1p(low_band_count) / np.log1p(float(self.args.hit_count_clip)), 0.0, 1.0)
        mid_band_count_norm = np.clip(np.log1p(mid_band_count) / np.log1p(float(self.args.hit_count_clip)), 0.0, 1.0)
        high_band_count_norm = np.clip(np.log1p(high_band_count) / np.log1p(float(self.args.hit_count_clip)), 0.0, 1.0)
        return {
            "obs_hit_count_norm": hit_count_norm.astype(np.float32),
            "obs_z_min": z_min.astype(np.float32),
            "obs_z_mean": z_mean.astype(np.float32),
            "obs_z_std": z_std.astype(np.float32),
            "obs_z_low_q": z_low_q.astype(np.float32),
            "obs_z_high_q": z_high_q.astype(np.float32),
            "obs_z_max": z_max.astype(np.float32),
            "obs_z_span": z_span.astype(np.float32),
            "obs_low_band_count_norm": low_band_count_norm.astype(np.float32),
            "obs_mid_band_count_norm": mid_band_count_norm.astype(np.float32),
            "obs_high_band_count_norm": high_band_count_norm.astype(np.float32),
        }

    def BuildObservationInputDict(self, frame_dir, requested_field_names):
        cache_dir = os.path.join(frame_dir, str(self.args.observation_input_subdir))
        if bool(self.args.b_cache_observation_inputs):
            cached_items = {}
            b_all_cached = True
            for field_name in requested_field_names:
                cache_filename = os.path.join(cache_dir, f"{field_name}.bin")
                if not os.path.exists(cache_filename):
                    b_all_cached = False
                    break
                cached_items[field_name] = ReadMat(cache_filename).astype(np.float32)
            if b_all_cached:
                return cached_items
        with open(os.path.join(frame_dir, "frame_meta.json"), "r", encoding="utf-8") as fp:
            frame_meta = json.load(fp)
        observation_maps = self.BuildObservationMaps(frame_meta)
        output_dict = {field_name: observation_maps[field_name] for field_name in requested_field_names}
        if bool(self.args.b_cache_observation_inputs):
            EnsureDir(cache_dir)
            for field_name, field_value in output_dict.items():
                WriteMat(os.path.join(cache_dir, f"{field_name}.bin"), field_value.astype(np.float32))
        return output_dict


class TerrainOnlyFrameDataset:
    def __init__(self, args):
        self.args = args
        with open(args.manifest_filename, "r", encoding="utf-8") as fp:
            self.manifest = json.load(fp)
        self.terrain_dir = self.manifest["terrain_dir"]
        self.frame_items = list(self.manifest["frames"])
        if int(args.max_samples) > 0:
            self.frame_items = self.frame_items[: int(args.max_samples)]
        input_args = BuildTerrainOnlyInputArgs()
        input_args.input_point_stride = int(args.input_point_stride)
        input_args.b_cache_observation_inputs = bool(args.b_cache_observation_inputs)
        self.input_field_names = list(input_args.default_input_field_names)
        self.input_builder = TerrainOnlyObservationInputBuilder(self.terrain_dir, input_args)

    def __len__(self):
        return len(self.frame_items)

    def __getitem__(self, sample_idx):
        frame_item = self.frame_items[int(sample_idx)]
        frame_dir = os.path.join(self.terrain_dir, frame_item["frame_dir"])
        input_dict = self.input_builder.BuildObservationInputDict(frame_dir, self.input_field_names)
        terrain_dir = os.path.join(frame_dir, "base", "terrain_completion")
        terrain_z = ReadMat(os.path.join(terrain_dir, "terrain_z_inpainted.bin")).astype(np.float32)
        terrain_mask = ReadMat(os.path.join(terrain_dir, "terrain_inpainted_mask.bin")).astype(np.float32)
        terrain_weight = ReadMat(os.path.join(terrain_dir, "terrain_train_weight.bin")).astype(np.float32)
        terrain_distance = ReadMat(os.path.join(terrain_dir, "terrain_inpaint_distance.bin")).astype(np.float32)
        return {
            "timestamp": int(frame_item["timestamp"]),
            "input": {field_name: AddChannelDim(input_dict[field_name]) for field_name in self.input_field_names},
            "target": {"terrain_z_inpainted": AddChannelDim(SanitizeHeightMap(terrain_z))},
            "mask": {"terrain_inpainted_mask": AddChannelDim(terrain_mask)},
            "weight": {
                "terrain_train_weight": AddChannelDim(terrain_weight),
                "terrain_inpaint_distance": AddChannelDim(terrain_distance),
            },
            "meta": {"frame_dir": frame_dir},
        }
