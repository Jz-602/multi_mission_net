"""真实train_epoch每卡CUDA峰值及主进程/递归loader子进程的低频PSS。"""
import json
import os
from pathlib import Path
import threading
import time

import torch

from run_options import WriteJsonAtomic

CPU_SAMPLE_INTERVAL_SECONDS = 1.0


def ProcessIdentity(proc_root, pid):
    """/proc stat的starttime绑定PID，避免采样期间PID被复用。"""
    try:
        value = (proc_root / str(pid) / 'stat').read_text()
        return int(value[value.rfind(')') + 2:].split()[19])
    except (OSError, ValueError, IndexError):
        return None


def ProcessTree(proc_root, parent_pid):
    pending = [parent_pid]
    identities = {}
    errors = []
    while pending:
        pid = pending.pop()
        if pid in identities:
            continue
        identity = ProcessIdentity(proc_root, pid)
        if identity is None:
            errors.append({'pid': pid, 'operation': 'identity',
                           'error': 'exited_or_identity_unreadable'})
            continue  # 无法区分退出与权限不足，不能把未读到的进程记为0 PSS。
        identities[pid] = identity
        try:
            # loader可能由不同线程创建；遍历每个task的children而不是只查主线程。
            for task in (proc_root / str(pid) / 'task').iterdir():
                try:
                    pending.extend(int(v) for v in (task / 'children').read_text().split())
                except FileNotFoundError:
                    continue  # 线程或子进程在遍历期间退出。
        except (OSError, ValueError) as error:
            errors.append({'pid': pid, 'operation': 'children', 'error': type(error).__name__})
    return identities, errors


def MemoryFields(text):
    values = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] in ('Pss:', 'Rss:', 'VmRSS:'):
            values[parts[0][:-1]] = int(parts[1]) * 1024
    return values


def SampleProcessMemory(parent_pid, proc_root=Path('/proc')):
    """每行保留实际身份和PSS/RSS；读不到PSS时绝不以RSS冒充。"""
    started_unix_ns = time.time_ns()
    started_monotonic = time.monotonic()
    identities, errors = ProcessTree(proc_root, parent_pid)
    processes = []
    for pid, identity in sorted(identities.items()):
        fields = {}
        pss_error = None
        try:
            fields = MemoryFields((proc_root / str(pid) / 'smaps_rollup').read_text())
            if 'Pss' not in fields:
                pss_error = 'missing_Pss_field'
        except (OSError, ValueError) as error:
            pss_error = type(error).__name__
        if 'Rss' not in fields:
            try:
                status = MemoryFields((proc_root / str(pid) / 'status').read_text())
                if 'VmRSS' in status:
                    fields['Rss'] = status['VmRSS']
            except (OSError, ValueError) as error:
                errors.append({'pid': pid, 'operation': 'rss_fallback', 'error': type(error).__name__})
        if ProcessIdentity(proc_root, pid) != identity:
            errors.append({'pid': pid, 'operation': 'identity', 'error': 'exited_or_reused_during_sample'})
            continue
        processes.append({'pid': pid, 'starttime_ticks': identity,
                          'role': 'training_rank' if pid == parent_pid else 'descendant',
                          'pss_bytes': fields.get('Pss'), 'rss_bytes': fields.get('Rss'),
                          'source': 'smaps_rollup' if 'Pss' in fields else 'rss_only_fallback',
                          'pss_error': pss_error})
    b_complete_pss = (bool(processes) and not errors
                      and all(p['pss_bytes'] is not None for p in processes))
    b_complete_rss = (bool(processes) and not errors
                      and all(p['rss_bytes'] is not None for p in processes))
    return {'started_unix_ns': started_unix_ns, 'finished_unix_ns': time.time_ns(),
            'started_monotonic_seconds': started_monotonic,
            'duration_seconds': time.monotonic() - started_monotonic,
            'parent_pid': parent_pid, 'processes': processes, 'errors': errors,
            'b_complete_pss': b_complete_pss, 'b_complete_rss': b_complete_rss,
            'pss_sum_bytes': sum(p['pss_bytes'] for p in processes) if b_complete_pss else None,
            'rss_sum_bytes': sum(p['rss_bytes'] for p in processes) if b_complete_rss else None}


class TrainingCPUMemorySampler:
    def __init__(self, path, interval_seconds=CPU_SAMPLE_INTERVAL_SECONDS):
        self.path = Path(path)
        self.interval_seconds = interval_seconds
        self.parent_pid = os.getpid()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.SampleLoop, daemon=True,
                                       name='training_pss_sampler')
        self.sample_count = 0
        self.complete_pss_samples = 0
        self.complete_rss_samples = 0
        self.peak_pss_bytes = None
        self.peak_rss_bytes = None
        self.sampling_error = None
        self.last_started = None
        self.max_actual_interval_seconds = 0.0

    def Start(self):
        self.thread.start()

    def SampleLoop(self):
        try:
            with self.path.open('x') as stream:
                while True:
                    sample = SampleProcessMemory(self.parent_pid)
                    stream.write(json.dumps(sample, allow_nan=False) + '\n')
                    stream.flush()
                    self.sample_count += 1
                    started = sample['started_monotonic_seconds']
                    if self.last_started is not None:
                        self.max_actual_interval_seconds = max(self.max_actual_interval_seconds,
                                                               started - self.last_started)
                    self.last_started = started
                    for metric in ('pss', 'rss'):
                        value = sample[metric + '_sum_bytes']
                        if value is not None:
                            setattr(self, 'complete_' + metric + '_samples',
                                    getattr(self, 'complete_' + metric + '_samples') + 1)
                            previous = getattr(self, 'peak_' + metric + '_bytes')
                            setattr(self, 'peak_' + metric + '_bytes', value if previous is None else max(previous, value))
                    if self.stop_event.wait(max(0., self.interval_seconds - sample['duration_seconds'])):
                        break
        except Exception as error:
            self.sampling_error = type(error).__name__ + ': ' + str(error)

    def Stop(self):
        self.stop_event.set()
        self.thread.join()
        return {'sample_interval_seconds': self.interval_seconds,
                'sample_count': self.sample_count,
                'complete_pss_samples': self.complete_pss_samples,
                'complete_rss_samples': self.complete_rss_samples,
                'b_all_samples_complete_pss': (self.sample_count > 0 and self.sampling_error is None
                                               and self.complete_pss_samples == self.sample_count),
                'sampled_peak_pss_sum_bytes': self.peak_pss_bytes,
                'sampled_peak_rss_sum_bytes': self.peak_rss_bytes,
                'maximum_actual_interval_seconds': self.max_actual_interval_seconds,
                'raw_samples': str(self.path), 'sampling_error': self.sampling_error,
                'scope': 'This rank PID plus recursive descendants (including data-loader workers) during train_epoch only; excludes torchrun launcher and validation/save/figures. PSS from smaps_rollup, RSS separately named fallback from rollup or status. Process exits/permission gaps are marked incomplete, not zero.',
                'aggregation': 'Per-rank sampled peaks are NOT a simultaneous host peak. Use per-sample start/end UTC nanoseconds and process identities to align all ranks within the 1-second sampling resolution; do not sum peaks from different times. Sampling may miss brief peaks; RSS double-counts shared pages, PSS apportions them.'}


def TrackTrainingMemory(trainer):
    if getattr(trainer, 'b_training_memory_wrapped', False):
        return
    trainer.b_training_memory_wrapped = True
    original_train_epoch = trainer.train_epoch

    def TrainEpochWithMemory():
        epoch = trainer.epoch + 1
        directory = Path(trainer.checkpoint_dir).parent / 'training_resources'
        directory.mkdir(exist_ok=True)
        sampler = TrainingCPUMemorySampler(directory / f'epoch_{epoch}_rank{trainer.rank}_cpu_{time.time_ns()}.jsonl')
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        sampler.Start()
        b_completed = False
        error_type = None
        try:
            original_train_epoch()
            torch.cuda.synchronize()
            b_completed = True
        except BaseException as error:
            error_type = type(error).__name__
            raise
        finally:
            elapsed_seconds = time.monotonic() - started
            cpu_memory = sampler.Stop()
            # 失败时也保存已知峰值和CPU样本，不能把部分轮记为完整轮。
            WriteJsonAtomic(directory / f'epoch_{epoch}_rank{trainer.rank}.json',
                            dict(epoch=epoch, rank=trainer.rank, elapsed_seconds=elapsed_seconds,
                                 b_epoch_completed=b_completed, training_error_type=error_type,
                                 peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                 peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                                 scope='train_epoch only, excludes validation, checkpoint saving and figures',
                                 tool='torch.cuda memory allocator peak counters', cpu_memory=cpu_memory))

    trainer.train_epoch = TrainEpochWithMemory
