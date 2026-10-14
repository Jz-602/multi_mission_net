"""CPU metadata derivation/binding tests; never open GT-bearing source infos."""
import json
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from easydict import EasyDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference_metadata import (PrepareInferenceMetadata, VerifyInferenceMetadata,
                                ConfigureInferenceMetadata, TokenHash, INPUT_FIELDS)
from run_options import HashFile


class InferenceMetadataTests(unittest.TestCase):
    def Fixture(self, root):
        rows = [dict(token=f'seq-{1000000+i*100000}', sequence_name='seq',
                     frame_timestamp_micros=1000000+i*100000, sample_idx=i,
                     pose=np.eye(4, dtype=np.float64), semantic_path='never_open/label.bin') for i in range(4)]
        source = root / 'validation_point_only.pkl'
        with source.open('wb') as stream:
            pickle.dump(rows, stream)
        tokens = [rows[3]['token'], rows[1]['token']]
        token_file = root / 'tokens.json'
        token_file.write_text(json.dumps(tokens))
        return source, token_file, tokens

    def Prepare(self, root):
        source, token_file, tokens = self.Fixture(root)
        out = root / 'input_only'
        PrepareInferenceMetadata(source, token_file, out, 2, TokenHash(tokens))
        return source, out, tokens

    def TestPreparationReadsPointOnlyAndPreservesInputsOrder(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, out, tokens = self.Prepare(Path(temporary))
            binding = VerifyInferenceMetadata(out / 'manifest.json', 2, TokenHash(tokens))
            self.assertEqual(binding['target_count'], 2)
            with Path(binding['info_path']).open('rb') as stream:
                rows = pickle.load(stream)
            self.assertEqual([row['token'] for row in rows], tokens)
            self.assertEqual([row['sample_idx'] for row in rows], [3, 1])
            self.assertTrue(all(set(row) == INPUT_FIELDS for row in rows))
            self.assertEqual(rows[0]['pose'].dtype, np.float64)
            np.testing.assert_array_equal(rows[0]['pose'], np.eye(4))
            # Verify/runtime uses generated data only; recorded source can be offline.
            source.unlink()
            self.assertTrue(VerifyInferenceMetadata(out / 'manifest.json', 2, TokenHash(tokens))['b_strict_input_only_metadata'])

    def TestGTSourceRejectedNotStripped(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, token_file, tokens = self.Fixture(root)
            with patch('inference_metadata.pickle.load', side_effect=AssertionError('must not open GT source')):
                with self.assertRaisesRegex(ValueError, 'point_only'):
                    PrepareInferenceMetadata(root / 'infos_validation.pkl', token_file, root / 'bad', 2, TokenHash(tokens))
            with source.open('rb') as stream:
                rows = pickle.load(stream)
            rows[0]['objects'] = []
            with source.open('wb') as stream:
                pickle.dump(rows, stream)
            with self.assertRaisesRegex(ValueError, 'non-input/GT'):
                PrepareInferenceMetadata(source, token_file, root / 'bad', 2, TokenHash(tokens))
            self.assertFalse((root / 'bad').exists())

    def TestMissingDuplicateAndChangedTargetListRejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, token_file, tokens = self.Fixture(root)
            for changed in ([tokens[0]] * 2, tokens[::-1], [tokens[0], 'seq-absent']):
                token_file.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    PrepareInferenceMetadata(source, token_file, root / 'bad', 2, TokenHash(tokens))
            token_file.write_text(json.dumps([tokens[0], 'seq-absent']))
            with self.assertRaisesRegex(ValueError, 'absent'):
                PrepareInferenceMetadata(source, token_file, root / 'bad', 2, TokenHash([tokens[0], 'seq-absent']))

    def TestContentTamperAndNonInputFieldsRejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, out, tokens = self.Prepare(Path(temporary))
            path = out / 'target_infos.pkl'
            with path.open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'binding mismatch'):
                VerifyInferenceMetadata(out / 'manifest.json', 2, TokenHash(tokens))
        with tempfile.TemporaryDirectory() as temporary:
            _, out, tokens = self.Prepare(Path(temporary))
            path = out / 'target_infos.pkl'
            with path.open('rb') as stream:
                rows = pickle.load(stream)
            rows[0]['objects'] = []
            with path.open('wb') as stream:
                pickle.dump(rows, stream)
            manifest = json.loads((out / 'manifest.json').read_text())
            manifest['files']['target_infos.pkl'].update(sha256=HashFile(path), size_bytes=path.stat().st_size)
            (out / 'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'non-input/GT'):
                VerifyInferenceMetadata(out / 'manifest.json', 2, TokenHash(tokens))

    def TestTargetPoseLinkAndMalformedPoseRejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, out, tokens = self.Prepare(Path(temporary))
            path = out / 'target_infos.pkl'
            with path.open('rb') as stream:
                rows = pickle.load(stream)
            rows[0]['pose'][0, 3] = 9.
            with path.open('wb') as stream:
                pickle.dump(rows, stream)
            manifest = json.loads((out / 'manifest.json').read_text())
            manifest['files']['target_infos.pkl'].update(sha256=HashFile(path), size_bytes=path.stat().st_size)
            (out / 'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'Target/history'):
                VerifyInferenceMetadata(out / 'manifest.json', 2, TokenHash(tokens))

    def TestBenchmarkConfigGuardsAndPrivateOverride(self):
        cfg = EasyDict(loading_pipelines=['FH_LoadPointCloud'], prepare_label=None, augmentation=None,
                       sampler=None, b_use_gt_sampling=False, info_path='legacy_GT.pkl', warmup_len=15)
        binding = dict(info_path='/generated/target_infos.pkl', full_info_path='/generated/history_infos.pkl')
        with patch('inference_metadata.VerifyInferenceMetadata', return_value=binding):
            self.assertEqual(ConfigureInferenceMetadata(cfg, 'manifest.json', 'val'), binding)
            self.assertEqual(cfg.info_path, binding['info_path'])
            self.assertFalse(cfg.b_use_completed_detection_labels)
            self.assertEqual(cfg.warmup_len, 15)
            self.assertEqual(cfg.sample_fraction, 1.)
            cfg.loading_pipelines += ['load_box3d']
            with self.assertRaisesRegex(ValueError, 'only FH_LoadPointCloud'):
                ConfigureInferenceMetadata(cfg, 'manifest.json', 'val')
            with self.assertRaisesRegex(ValueError, 'validation split'):
                ConfigureInferenceMetadata(cfg, 'manifest.json', 'train')


if __name__ == '__main__':
    loader = unittest.TestLoader()
    loader.testMethodPrefix = 'Test'
    result = unittest.TextTestRunner(verbosity=2).run(loader.loadTestsFromTestCase(InferenceMetadataTests))
    sys.exit(0 if result.wasSuccessful() else 1)
