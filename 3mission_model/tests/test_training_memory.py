"""CPU-only resource sampling tests; CUDA counters are mocked, never initialized."""
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import training_memory as memory


class TrainingMemoryTests(unittest.TestCase):
    def MakeProcess(self, root, pid, children='', pss=10, rss=20, start=123):
        directory = root / str(pid)
        (directory / 'task' / str(pid)).mkdir(parents=True)
        # Values after comm start at field 3; starttime is field 22.
        (directory / 'stat').write_text(f'{pid} (name with ) space) ' + ' '.join(['S'] + ['0'] * 18 + [str(start)]))
        (directory / 'task' / str(pid) / 'children').write_text(children)
        (directory / 'status').write_text(f'VmRSS: {rss} kB\n')
        if pss is not None:
            (directory / 'smaps_rollup').write_text(f'Rss: {rss} kB\nPss: {pss} kB\n')
        return directory

    def TestRecursiveChildrenAndThreadChildrenCountOnce(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent = self.MakeProcess(root, 10, '20', pss=10)
            self.MakeProcess(root, 20, '30', pss=11)
            self.MakeProcess(root, 30, pss=12)
            (parent / 'task' / '11').mkdir()
            (parent / 'task' / '11' / 'children').write_text('30')
            sample = memory.SampleProcessMemory(10, root)
            self.assertTrue(sample['b_complete_pss'])
            self.assertEqual(sample['pss_sum_bytes'], 33 * 1024)
            self.assertEqual(sample['rss_sum_bytes'], 60 * 1024)
            self.assertEqual([p['pid'] for p in sample['processes']], [10, 20, 30])
            self.assertEqual(sample['processes'][0]['starttime_ticks'], 123)

    def TestMissingPSSHasNamedRSSFallbackNotFakePSS(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.MakeProcess(root, 10, '20')
            self.MakeProcess(root, 20, pss=None, rss=30)
            sample = memory.SampleProcessMemory(10, root)
            self.assertFalse(sample['b_complete_pss'])
            self.assertIsNone(sample['pss_sum_bytes'])
            self.assertTrue(sample['b_complete_rss'])
            self.assertEqual(sample['rss_sum_bytes'], 50 * 1024)
            self.assertEqual(sample['processes'][1]['source'], 'rss_only_fallback')
            self.assertEqual(sample['processes'][1]['pss_error'], 'FileNotFoundError')

    def TestPermissionDeniedIsExplicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.MakeProcess(root, 10)
            read_text = Path.read_text
            def ReadText(path, *args, **kwargs):
                if path.name == 'smaps_rollup':
                    raise PermissionError('fixture')
                return read_text(path, *args, **kwargs)
            with patch.object(Path, 'read_text', ReadText):
                sample = memory.SampleProcessMemory(10, root)
            self.assertIsNone(sample['pss_sum_bytes'])
            self.assertEqual(sample['processes'][0]['pss_error'], 'PermissionError')
            self.assertEqual(sample['rss_sum_bytes'], 20 * 1024)

    def TestPIDReuseAndUnreadableTreeCannotLookComplete(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.MakeProcess(root, 10)
            with patch.object(memory, 'ProcessIdentity', side_effect=[123, 124]):
                sample = memory.SampleProcessMemory(10, root)
            self.assertFalse(sample['b_complete_pss'])
            self.assertEqual(sample['errors'][0]['operation'], 'identity')
            with patch.object(Path, 'iterdir', side_effect=PermissionError('fixture')):
                sample = memory.SampleProcessMemory(10, root)
            self.assertFalse(sample['b_complete_pss'])
            self.assertEqual(sample['errors'][0]['operation'], 'children')

    def TestRealLinuxSelfPSS(self):
        sample = memory.SampleProcessMemory(os.getpid())
        self.assertTrue(sample['b_complete_pss'], sample['errors'])
        self.assertGreater(sample['pss_sum_bytes'], 0)
        self.assertGreaterEqual(sample['rss_sum_bytes'], sample['pss_sum_bytes'])
        self.assertGreaterEqual(sample['finished_unix_ns'], sample['started_unix_ns'])

    def TestMissingChildIdentityIsNotAnAuthoritativePartialSum(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.MakeProcess(root, 10, '999')
            sample = memory.SampleProcessMemory(10, root)
            self.assertFalse(sample['b_complete_pss'])
            self.assertIsNone(sample['pss_sum_bytes'])
            self.assertEqual(sample['errors'][0]['pid'], 999)

    def TestSamplerIOFailureIsExplicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            sampler = memory.TrainingCPUMemorySampler(Path(temporary) / 'missing' / 'samples.jsonl')
            sampler.Start()
            result = sampler.Stop()
            self.assertEqual(result['sample_count'], 0)
            self.assertFalse(result['b_all_samples_complete_pss'])
            self.assertIsNone(result['sampled_peak_pss_sum_bytes'])
            self.assertIn('FileNotFoundError', result['sampling_error'])

    def TestSamplerSavesTimedRowsAndDistinctPeaks(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'samples.jsonl'
            sampler = memory.TrainingCPUMemorySampler(path, interval_seconds=0.001)
            ready = threading.Event()
            counter = [0]
            def Sample(pid):
                counter[0] += 1
                if counter[0] >= 3:
                    ready.set()
                return dict(started_unix_ns=counter[0] * 1000, finished_unix_ns=counter[0] * 1000 + 1,
                            started_monotonic_seconds=counter[0], duration_seconds=0,
                            pss_sum_bytes=20 if counter[0] == 2 else 10,
                            rss_sum_bytes=100 if counter[0] == 1 else 50)
            with patch.object(memory, 'SampleProcessMemory', Sample):
                sampler.Start()
                self.assertTrue(ready.wait(2))
                result = sampler.Stop()
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(result['sample_count'], len(rows))
            self.assertGreaterEqual(len(rows), 3)
            self.assertEqual(result['sampled_peak_pss_sum_bytes'], 20)
            self.assertEqual(result['sampled_peak_rss_sum_bytes'], 100)
            self.assertTrue(result['b_all_samples_complete_pss'])
            self.assertFalse(sampler.thread.is_alive())
            self.assertIn('NOT a simultaneous host peak', result['aggregation'])

    def TestTrainingWrapperSuccessFailureAndIdempotence(self):
        for b_failure in (False, True):
            with self.subTest(b_failure=b_failure), tempfile.TemporaryDirectory() as temporary:
                trainer = MagicMock()
                trainer.b_training_memory_wrapped = False
                trainer.epoch = 4
                trainer.rank = 3
                trainer.checkpoint_dir = str(Path(temporary) / 'checkpoints')
                if b_failure:
                    trainer.train_epoch.side_effect = RuntimeError('real training failure')
                original = trainer.train_epoch
                with patch.object(memory.torch, 'cuda') as cuda, patch.object(memory, 'TrainingCPUMemorySampler') as sampler_class:
                    cuda.max_memory_allocated.return_value = 111
                    cuda.max_memory_reserved.return_value = 222
                    sampler_class.return_value.Stop.return_value = {'sample_count': 1}
                    memory.TrackTrainingMemory(trainer)
                    wrapped = trainer.train_epoch
                    memory.TrackTrainingMemory(trainer)
                    self.assertIs(wrapped, trainer.train_epoch)
                    if b_failure:
                        with self.assertRaisesRegex(RuntimeError, 'real training failure'):
                            trainer.train_epoch()
                    else:
                        trainer.train_epoch()
                    original.assert_called_once_with()
                    sampler_class.return_value.Start.assert_called_once_with()
                    sampler_class.return_value.Stop.assert_called_once_with()
                    self.assertEqual(cuda.synchronize.call_count, 1 if b_failure else 2)
                result = json.loads((Path(temporary) / 'training_resources/epoch_5_rank3.json').read_text())
                self.assertEqual(result['epoch'], 5)
                self.assertEqual(result['rank'], 3)
                self.assertEqual(result['b_epoch_completed'], not b_failure)
                self.assertEqual(result['training_error_type'], 'RuntimeError' if b_failure else None)
                self.assertEqual(result['peak_allocated_bytes'], 111)
                self.assertEqual(result['peak_reserved_bytes'], 222)

    def TestTrainDurationExcludesSamplerStop(self):
        with tempfile.TemporaryDirectory() as temporary:
            trainer = MagicMock()
            trainer.b_training_memory_wrapped = False
            trainer.epoch = 0
            trainer.rank = 0
            trainer.checkpoint_dir = str(Path(temporary) / 'checkpoints')
            clock = [10.]
            def Train():
                clock[0] += 3.
            def Stop():
                clock[0] += 9.
                return {}
            trainer.train_epoch = Train
            with patch.object(memory.torch, 'cuda') as cuda, patch.object(memory, 'TrainingCPUMemorySampler') as sampler_class, patch.object(memory.time, 'monotonic', side_effect=lambda: clock[0]):
                cuda.max_memory_allocated.return_value = 111
                cuda.max_memory_reserved.return_value = 222
                sampler_class.return_value.Stop.side_effect = Stop
                memory.TrackTrainingMemory(trainer)
                trainer.train_epoch()
            result = json.loads((Path(temporary) / 'training_resources/epoch_1_rank0.json').read_text())
            self.assertEqual(result['elapsed_seconds'], 3.)


if __name__ == '__main__':
    loader = unittest.TestLoader()
    loader.testMethodPrefix = 'Test'
    result = unittest.TextTestRunner(verbosity=2).run(loader.loadTestsFromTestCase(TrainingMemoryTests))
    sys.exit(0 if result.wasSuccessful() else 1)
