"""CPU regressions for fixed target selection and non-destructive export."""
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from build_label_manifest import HashTextLines
from prepare_server_infos import SelectFullInfos, SelectTargets, WriteInfos


class PrepareServerInfosTest(unittest.TestCase):
    def setUp(self):
        self.pool = {
            "a-1000": {"token": "a-1000", "sequence_name": "a", "frame_timestamp_micros": 1000},
            "a-2000": {"token": "a-2000", "sequence_name": "a", "frame_timestamp_micros": 2000},
            "b-3000": {"token": "b-3000", "sequence_name": "b", "frame_timestamp_micros": 3000},
        }
        tokens = ["b-3000", "a-2000"]
        self.manifest = {
            "target_count": 2, "token_sha256": HashTextLines(tokens),
            "targets": [
                {"token": "b-3000", "sequence_name": "b", "timestamp_ms": 3},
                {"token": "a-2000", "sequence_name": "a", "timestamp_ms": 2},
            ],
        }

    def test_manifest_order_and_all_history_preserved(self):
        targets = SelectTargets(self.pool, self.manifest)
        self.assertEqual([info["token"] for info in targets], ["b-3000", "a-2000"])
        self.assertIs(targets[0], self.pool["b-3000"])
        self.assertEqual(SelectFullInfos(self.pool, targets), list(self.pool.values()))

    def test_mismatch_or_missing_target_rejected(self):
        self.manifest["targets"][0]["timestamp_ms"] = 4
        with self.assertRaisesRegex(ValueError, "timestamp mismatch"):
            SelectTargets(self.pool, self.manifest)
        self.manifest["targets"][0]["timestamp_ms"] = 3
        del self.pool["b-3000"]
        with self.assertRaises(KeyError):
            SelectTargets(self.pool, self.manifest)

    def test_existing_difference_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / "infos.pkl"
            records = list(self.pool.values())
            self.assertEqual(WriteInfos(filename, records), "created")
            self.assertEqual(WriteInfos(filename, records), "reused")
            with self.assertRaises(FileExistsError):
                WriteInfos(filename, records[:-1])
            self.assertEqual(pickle.loads(filename.read_bytes()), records)


if __name__ == "__main__":
    unittest.main()
