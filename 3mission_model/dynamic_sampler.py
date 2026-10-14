"""Epoch target indices only; temporal history/point coordinates remain unchanged."""
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Sampler


def HashTokens(tokens):
    return hashlib.sha256('\n'.join(tokens).encode()).hexdigest()


class EpochSubsetSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None):
        self.num_replicas = num_replicas if num_replicas is not None else (dist.get_world_size() if dist.is_initialized() else 1)
        self.rank = rank if rank is not None else (dist.get_rank() if dist.is_initialized() else 0)
        self.fraction = float(dataset.sample_fraction)
        self.seed = int(dataset.sample_seed)
        self.tokens = [info['token'] for info in dataset.infos]
        if not self.tokens or len(set(self.tokens)) != len(self.tokens):
            raise ValueError('Dynamic candidate pool must contain nonempty unique tokens')
        if not 0 < self.fraction <= 1 or self.seed < 0 or not 0 <= self.rank < self.num_replicas:
            raise ValueError('Invalid epoch sampling fraction/seed/rank/world size')
        self.selected_count = math.ceil(len(self.tokens) * self.fraction)
        self.num_samples = math.ceil(self.selected_count / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        self.epoch = 0
        self.coverage_epoch = -1
        self.covered_indices = set()
        self.candidate_sha256 = HashTokens(self.tokens)

    def __len__(self):
        return self.num_samples

    def SelectedIndices(self, epoch=None):
        generator = torch.Generator().manual_seed(self.seed + (self.epoch if epoch is None else epoch))
        return torch.randperm(len(self.tokens), generator=generator)[:self.selected_count].tolist()

    def __iter__(self):
        indices = self.SelectedIndices()
        indices = (indices * math.ceil(self.total_size / len(indices)))[:self.total_size]
        return iter(indices[self.rank:self.total_size:self.num_replicas])

    def SetEpoch(self, epoch):
        if epoch < 0:
            raise ValueError('Epoch must be nonnegative')
        self.epoch = int(epoch)
        if self.epoch < self.coverage_epoch:
            self.coverage_epoch = -1
            self.covered_indices.clear()
        for previous_epoch in range(self.coverage_epoch + 1, self.epoch + 1):
            self.covered_indices.update(self.SelectedIndices(previous_epoch))
        self.coverage_epoch = self.epoch

    def Definition(self):
        return {'mode':'dynamic_epoch_subset', 'candidate_pool_count':len(self.tokens), 'candidate_pool_sha256':self.candidate_sha256,
                'fraction':self.fraction, 'seed':self.seed, 'selection_rule':'torch.randperm(seed + zero_based_epoch), prefix ceil(pool*fraction), without replacement',
                'world_size':self.num_replicas, 'selected_count':self.selected_count, 'sampled_count':self.total_size, 'samples_per_rank':self.num_samples}

    def EpochRecord(self):
        indices = self.SelectedIndices()
        tokens = [self.tokens[index] for index in indices]
        union_tokens = [self.tokens[index] for index in sorted(self.covered_indices)]
        padded_indices = (indices * math.ceil(self.total_size / len(indices)))[len(indices):self.total_size]
        return dict(self.Definition(), epoch=self.epoch + 1, selection_seed=self.seed + self.epoch,
                    selected_tokens=tokens, selected_sha256=HashTokens(tokens), padding_count=len(padded_indices),
                    padding_tokens=[self.tokens[index] for index in padded_indices], union_count=len(union_tokens), union_sha256=HashTokens(union_tokens),
                    union_fraction=len(union_tokens)/len(self.tokens), coverage_basis='Selected targets through this epoch; current selection is recorded before training, not proof of completed updates')


def SamplingDefinition(dataloader):
    sampler = getattr(dataloader, 'sampler', None)
    if not isinstance(sampler, EpochSubsetSampler):
        return None
    return dict(sampler.Definition(), worker_seed_base=int(dataloader.epoch_seed), batch_size=dataloader.batch_size, steps_per_epoch=len(dataloader), b_epoch_boundary_resume_only=True)


def PrepareEpochSampling(dataloader, epoch, rank, checkpoint_dir):
    sampler = dataloader.sampler
    if isinstance(sampler, EpochSubsetSampler):
        sampler.SetEpoch(epoch)
        epoch_seed = int(dataloader.epoch_seed) + int(epoch) + int(rank)
        dataloader.generator.manual_seed(epoch_seed)
        if dataloader.num_workers == 0:
            np.random.seed(epoch_seed % 2**32)
            random.seed(epoch_seed)
        record = dict(sampler.EpochRecord(), worker_seed_base=int(dataloader.epoch_seed), worker_seed_rule='worker generator base + zero_based_epoch + rank; workers use torch.initial_seed')
        if dist.is_initialized():
            hashes = [None] * dist.get_world_size()
            summary_sha256 = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
            dist.all_gather_object(hashes, summary_sha256)
            if len(set(hashes)) != 1:
                raise RuntimeError('Dynamic epoch selection differs across DDP ranks')
        if rank == 0:
            path = Path(checkpoint_dir).parent / 'epoch_sampling' / f'epoch_{epoch+1:03d}.json'
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                if json.loads(path.read_text()) != record:
                    raise RuntimeError('Existing epoch selection record differs; refusing overwrite')
            else:
                temporary = path.with_suffix('.json.tmp')
                temporary.write_text(json.dumps(record))
                temporary.replace(path)
    elif hasattr(sampler, 'set_epoch'):
        sampler.set_epoch(epoch)


def ValidateSamplingResume(dataloader, checkpoint, scheduler):
    definition = SamplingDefinition(dataloader)
    if definition is None:
        if checkpoint['meta'].get('epoch_sampling') is not None:
            raise RuntimeError('Dynamic checkpoint cannot resume with a static loader')
        return
    if checkpoint['meta'].get('epoch_sampling') != definition:
        raise RuntimeError('Resume epoch sampling definition/pool/seed/world size mismatch')
    if checkpoint['meta']['iter'] != checkpoint['meta']['epoch'] * len(dataloader):
        raise RuntimeError('Resume requires a complete epoch boundary with matching update count')
    if scheduler is not None and (checkpoint['scheduler']['total_steps'] != scheduler.total_steps or checkpoint['scheduler']['last_epoch'] != checkpoint['meta']['iter']):
        raise RuntimeError('Resume scheduler budget/progress mismatch')
