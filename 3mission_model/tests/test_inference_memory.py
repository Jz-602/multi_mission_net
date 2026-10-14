"""CPU-only resource sampling: process tree, missing data and window attribution."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference_memory import InferenceMemorySampler, SummarizeWindow


def Row(start, finish, pss, rss, b_complete=True):
    return dict(started_unix_ns=start, finished_unix_ns=finish,
                started_monotonic_seconds=start / 1e9, duration_seconds=(finish-start)/1e9,
                b_complete_pss=b_complete and pss is not None,
                b_complete_rss=b_complete and rss is not None,
                pss_sum_bytes=pss, rss_sum_bytes=rss,
                parent_pid=1, processes=[], errors=[])


class TestInferenceMemory(unittest.TestCase):
    def test_windows_and_missing(self):
        rows = [Row(0, 2, 800, 900), Row(3, 4, 20, 30),
                Row(5, 6, None, 70), Row(7, 8, 1000, 2000, False),
                Row(9, 11, 9999, 9999)]
        result = SummarizeWindow(rows, 1, 10)
        self.assertEqual(result['sample_indices'], [1, 2, 3])
        self.assertEqual(result['boundary_overlap_sample_indices'], [0, 4])
        self.assertEqual(result['sampled_peak_pss_sum_bytes'], 20)
        self.assertEqual(result['sampled_peak_rss_sum_bytes'], 70)
        self.assertEqual(result['incomplete_pss_samples'], 2)
        self.assertFalse(result['b_all_samples_complete_pss'])
        for interval in result['actual_start_intervals_seconds']:
            self.assertAlmostEqual(interval, 2e-9)
        empty = SummarizeWindow(rows, 20, 21)
        self.assertIsNone(empty['sampled_peak_pss_sum_bytes'])
        self.assertFalse(empty['b_all_samples_complete_rss'])

    def test_incomplete_sample_keeps_exit_evidence(self):
        now = time.time_ns()
        row = Row(now, now + 1, None, 70)
        row['errors'] = [{'pid': 42, 'operation': 'identity', 'error': 'exited_or_reused_during_sample'}]
        row['b_complete_rss'] = False
        row['rss_sum_bytes'] = None
        with tempfile.TemporaryDirectory() as directory:
            with patch('inference_memory.SampleProcessMemory', return_value=row):
                with InferenceMemorySampler(Path(directory) / 'memory.jsonl') as sampler:
                    time.sleep(.03)
            self.assertEqual(sampler.samples[0]['errors'], row['errors'])
            self.assertIsNone(sampler.samples[0]['pss_sum_bytes'])
            self.assertIsNone(sampler.samples[0]['rss_sum_bytes'])

    def test_no_sum_of_independent_peaks(self):
        rows = [Row(1, 2, 100, 150), Row(3, 4, 90, 250)]
        result = SummarizeWindow(rows, 0, 5)
        self.assertEqual(result['sampled_peak_pss_sum_bytes'], 100)
        self.assertEqual(result['sampled_peak_rss_sum_bytes'], 250)

    def test_failure_and_unfinished_window_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'memory.jsonl'
            with patch('inference_memory.SampleProcessMemory', side_effect=PermissionError('denied')):
                with InferenceMemorySampler(path) as sampler:
                    sampler.Begin('unfinished')
                    sampler.thread.join(timeout=2)
            result = sampler.Report()
            self.assertIn('PermissionError', result['sampling_error'])
            self.assertFalse(result['windows']['unfinished']['b_window_complete'])
            self.assertEqual(result['windows']['whole_resource_pass']['sample_count'], 0)
            self.assertFalse(result['b_latency_eligible'])
            self.assertTrue(path.with_suffix('.windows.json').exists())

    def test_real_child_raw_rows_and_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            child = subprocess.Popen([sys.executable, '-u', '-c',
                'import time; data=bytearray(16*1024*1024); print("ready",flush=True); time.sleep(20)'],
                stdout=subprocess.PIPE, text=True)
            try:
                self.assertEqual(child.stdout.readline().strip(), 'ready')
                path = Path(directory) / 'memory.jsonl'
                with InferenceMemorySampler(path) as sampler:
                    sampler.Begin('case')
                    time.sleep(1.2)
                    sampler.End('case')
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                self.assertEqual(rows, sampler.samples)
                self.assertGreaterEqual(len(rows), 2)
                self.assertIsNone(sampler.sampling_error)
                result = sampler.Report()
                whole = result['windows']['whole_resource_pass']
                self.assertEqual(whole['sampled_peak_pss_sum_bytes'],
                                 max(row['pss_sum_bytes'] for row in rows))
                self.assertGreater(whole['maximum_actual_interval_seconds'], .9)
                for row in rows:
                    children = [item for item in row['processes'] if item['pid'] == child.pid]
                    self.assertEqual(len(children), 1)
                    self.assertGreater(children[0]['pss_bytes'], 16*1024*1024)
                    self.assertTrue(row['b_complete_pss'])
                    self.assertEqual(sum(item['pss_bytes'] for item in row['processes']), row['pss_sum_bytes'])
                    self.assertEqual(sum(item['rss_bytes'] for item in row['processes']), row['rss_sum_bytes'])
                    self.assertEqual(next(item for item in row['processes'] if item['pid'] == row['parent_pid'])['role'], 'inference_main')
            finally:
                child.terminate()
                child.wait(timeout=5)
                child.stdout.close()


if __name__ == '__main__':
    unittest.main()
