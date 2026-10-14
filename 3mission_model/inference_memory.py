"""独立推理资源 pass：复用训练的 /proc 读取，保留采样窗口及缺失证据。"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading
import time

from training_memory import SampleProcessMemory


def UTC(timestamp_ns):
    return datetime.fromtimestamp(timestamp_ns / 1e9, timezone.utc).isoformat()


def SummarizeWindow(samples, started_unix_ns, finished_unix_ns):
    """仅完全落在窗口内的读取参与峰值；跨边界读取另列，不能硬归属。"""
    contained = [i for i, row in enumerate(samples)
                 if started_unix_ns <= row['started_unix_ns']
                 and row['finished_unix_ns'] <= finished_unix_ns]
    overlapping = [i for i, row in enumerate(samples)
                   if row['started_unix_ns'] < finished_unix_ns
                   and row['finished_unix_ns'] > started_unix_ns and i not in contained]
    rows = [samples[i] for i in contained]
    intervals = [b['started_monotonic_seconds'] - a['started_monotonic_seconds']
                 for a, b in zip(rows, rows[1:])]
    result = {'started_unix_ns': started_unix_ns, 'finished_unix_ns': finished_unix_ns,
              'started_utc': UTC(started_unix_ns), 'finished_utc': UTC(finished_unix_ns),
              'sample_indices': contained, 'boundary_overlap_sample_indices': overlapping,
              'sample_count': len(rows), 'actual_start_intervals_seconds': intervals,
              'maximum_actual_interval_seconds': max(intervals, default=None)}
    for metric in ('pss', 'rss'):
        values = [row[metric + '_sum_bytes'] for row in rows
                  if row['b_complete_' + metric] and row[metric + '_sum_bytes'] is not None]
        result.update({f'complete_{metric}_samples': len(values),
                       f'incomplete_{metric}_samples': len(rows) - len(values),
                       f'b_all_samples_complete_{metric}': bool(rows) and len(values) == len(rows),
                       f'sampled_peak_{metric}_sum_bytes': max(values, default=None)})
    return result


class InferenceMemorySampler:
    """资源专用；名义 1 秒，实际间隔由原始时间戳核对，不宣称正式时延。"""
    def __init__(self, path):
        self.path = Path(path)
        self.samples = []
        self.windows = {}
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.Run, daemon=True, name='inference_pss_sampler')
        self.sampling_error = None

    def Begin(self, name):
        if name in self.windows:
            raise ValueError(f'duplicate memory window: {name}')
        self.windows[name] = {'started_unix_ns': time.time_ns()}

    def End(self, name):
        self.windows[name]['finished_unix_ns'] = time.time_ns()

    def Run(self):
        try:
            with self.path.open('x') as stream:
                while not self.stop_event.is_set():
                    sample = SampleProcessMemory(os.getpid())
                    # 复用读取函数的数值，明确当前角色并非训练 rank。
                    for process in sample['processes']:
                        if process['pid'] == sample['parent_pid']:
                            process['role'] = 'inference_main'
                    self.samples.append(sample)
                    stream.write(json.dumps(sample, allow_nan=False) + '\n')
                    stream.flush()
                    self.stop_event.wait(max(0., 1.0 - sample['duration_seconds']))
        except Exception as error:
            self.sampling_error = type(error).__name__ + ': ' + str(error)

    def __enter__(self):
        self.Begin('whole_resource_pass')
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop_event.set()
        self.thread.join()
        self.End('whole_resource_pass')
        self.path.with_suffix('.windows.json').write_text(json.dumps(self.Report(), indent=2) + '\n')

    def Report(self):
        windows = {}
        for name, window in self.windows.items():
            if 'finished_unix_ns' not in window:
                windows[name] = dict(window, b_window_complete=False)
            else:
                windows[name] = dict(SummarizeWindow(self.samples, **window), b_window_complete=True)
        return {'b_latency_eligible': False, 'nominal_interval_seconds': 1.0,
                'b_sampling_succeeded': bool(self.samples) and self.sampling_error is None,
                'sampling_error': self.sampling_error, 'raw_samples_path': str(self.path),
                'samples': self.samples, 'windows': windows,
                'scope': 'Main process plus recursive live descendants, including loader workers; excludes unrelated processes. Window peaks include only samples wholly contained in that window. Short windows may have no sample. Process exits and permission errors remain explicit incomplete observations.',
                'aggregation': 'Each row sums one sequential process-tree sweep, then the window takes its maximum; never sum per-process peaks. Sweep start/end and duration bound time skew. PSS apportions shared pages; RSS counts shared pages repeatedly. Missing PSS is never replaced by RSS. One-second sampling can miss short peaks.',
                'timing_policy': 'All timings in this resource pass are diagnostic only. Run without --collect-pss for formal latency, with matching inputs/config/checkpoint.'}
