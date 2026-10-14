"""Target selection must affect output frames, never global-map input stages."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from gt_generate import generate_three_task_gt as driver
from gt_generate.ground_truth import export_frame_terrain_gt as terrain


class TargetExportTest(unittest.TestCase):
    def test_manifest_merges_splits_by_sequence(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            manifest.write_text(json.dumps({"splits": {
                "train": {"targets": [{"sequence_name": "a", "timestamp_ms": 1}]},
                "val": {"targets": [{"sequence_name": "a", "timestamp_ms": 2}]},
            }}))
            self.assertEqual(driver.LoadTargetTimestamps(manifest), {"a": {1, 2}})

    def test_driver_only_passes_targets_to_frame_exports(self):
        with tempfile.TemporaryDirectory() as directory:
            sequence_dir = Path(directory)
            for stage, script, extra_args in driver.STAGES:
                with patch.object(driver, "ValidateOutputs", side_effect=[FileNotFoundError(), None]) as validate, patch.object(driver.subprocess, "run") as run:
                    driver.RunStage(sequence_dir, {1, 2, 3}, stage, script, extra_args, target_timestamps={2})
                    command = run.call_args.args[0]
                    self.assertEqual("target_timestamps=2" in command, stage != "terrain_global")
                    self.assertEqual(validate.call_args.args[1], {1, 2, 3} if stage == "terrain_global" else {2})

    def test_terrain_keeps_full_pose_index_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            args = terrain.BuildArgs()
            args.terrain_dir = directory
            args.output_dir = str(Path(directory) / "frames")
            args.pcd_dir = directory
            args.b_clip_to_last_semantic_frame = False
            args.target_timestamps = "2"
            args.forward_range_m = args.backward_range_m = 0.2
            args.left_range_m = args.right_range_m = 0.2
            for timestamp in (1, 2, 3):
                (Path(directory) / f"{timestamp}.pcd").touch()
            poses = np.arange(18).reshape(3, 6)
            with patch.object(terrain, "LoadMapInfo", return_value={}), patch.object(terrain, "LoadGlobalTerrainMaps", return_value={}) as load_maps, patch.object(terrain, "LoadPoseDb", return_value=(np.array([1, 2, 3]), poses)), patch.object(terrain, "BuildFrameTerrainGt", return_value={}) as build_frame, patch.object(terrain, "SaveFrameTerrainGt"), patch.object(terrain, "SaveJson") as save_json:
                manifest = terrain.ExportFrameTerrainGt(args)
                self.assertEqual([frame["timestamp"] for frame in manifest["frames"]], [2])
                self.assertEqual(manifest["frames"][0]["frame_index"], 1)
                np.testing.assert_array_equal(build_frame.call_args.args[1], poses[1])
                load_maps.assert_called_once_with(directory)
                metadata = save_json.call_args_list[0].args[1]
                self.assertEqual(metadata["last_semantic_frame_index"], 2)
                self.assertNotIn("target_timestamps", metadata)
                args.target_timestamps = "4"
                with self.assertRaisesRegex(ValueError, "incomplete"):
                    terrain.ExportFrameTerrainGt(args)


if __name__ == "__main__":
    unittest.main()
