"""CPU mock验证真实trainer训练步、有限预算、仅根Adam重置和截断checkpoint。"""
import copy
import json
import logging
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from occ_head import GeneratedOCCHead
from root_diagnostic_runtime import InstrumentOptimizeStep, ResetRootHead, RunRootDiagnostic, TrackRootOptimization
from trainer.trainer.trainer_multi import Trainer
from trainer.utils.checkpoint import save_checkpoint


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(256, 256)
        self.occ_head = GeneratedOCCHead.__new__(GeneratedOCCHead)
        torch.nn.Module.__init__(self.occ_head)
        self.occ_head.root_score = torch.nn.Linear(256, 1)
        self.occ_head.root_input_normalization = 'none'
        self.occ_head.b_collect_root_diagnostics = True
        self.occ_head.delete_threshold = .5
        self.occ_head.root_loss_weight = 1.
        self.config = {}
        self.task_uncertainty_loss = SimpleNamespace(task_names=['det', 'terrain', 'occ'])

    def forward(self, batch, memory=None, b_memory_only=False):
        if b_memory_only:
            return None
        features = self.backbone(batch['x'])
        logits = self.occ_head.root_score(self.occ_head.RootInputFeatures(features)).flatten()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, batch['y'])
        logs = [{'task': [name], 'loss': loss.detach(), 'loss_weight': 1.}
                for name in ['det', 'terrain', 'occ']]
        logs[-1].update(self.occ_head.RootGroupStatistics(logits, batch['y']))
        logs[-1].update(root_loss=loss.detach(), root_count=len(logits),
                        occupied_root_count=int((batch['y'] == 0).sum()),
                        occupied_root_kept=int(((batch['y'] == 0) & (logits.sigmoid() <= .5)).sum()))
        return loss, logs, None


class Loader:
    def __len__(self):
        return 4

    def __iter__(self):
        for index in range(4):
            yield {'frames': [{'x': torch.arange(8 * 256).reshape(8, 256).float() / 256,
                               'y': torch.tensor([0., 1.] * 4), 'token': [str(index)]}],
                   'warmup_len': 0, 'train_len': 1}


def Build(directory):
    model = ToyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.00075)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=.00075, total_steps=192)
    trainer = Trainer(model, Loader(), None, optimizer, scheduler, clip_grad_val=35., max_epochs=48,
                      checkpoint_dir=str(directory / 'checkpoints'), logger=logging.getLogger('test'))
    # 不调用真实train_iter的自动CUDA选设备；CPU mock明确固定CPU。
    trainer_type = type('CPUTrainer', (Trainer,), {'device': property(lambda self: torch.device('cpu'))})
    trainer.__class__ = trainer_type
    for _ in range(80):
        trainer.optimize_step(model(next(iter(Loader()))['frames'][0])[0])
    trainer.epoch, trainer.global_step = 20, 80
    return trainer


def Main():
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        trainer = Build(root / 'base')
        saved_optimizer = copy.deepcopy(trainer.optimizer.state_dict())
        state = copy.deepcopy(trainer.model.state_dict())
        original_clip = torch.nn.utils.clip_grad_norm_
        loss = trainer.model(next(iter(Loader()))['frames'][0])[0]
        record = {}
        wrapped = InstrumentOptimizeStep(trainer, trainer.model.occ_head, record)
        wrapped(loss)
        expected = copy.deepcopy(trainer.model.state_dict())
        expected_opt = copy.deepcopy(trainer.optimizer.state_dict())
        assert torch.nn.utils.clip_grad_norm_ is original_clip
        trainer.model.load_state_dict(state)
        trainer.optimizer.load_state_dict(saved_optimizer)
        trainer.lr_scheduler.last_epoch -= 1
        trainer.optimize_step(trainer.model(next(iter(Loader()))['frames'][0])[0])
        assert all(torch.equal(v, expected[k]) for k, v in trainer.model.state_dict().items())
        for key, item in trainer.optimizer.state_dict()['state'].items():
            assert all(torch.equal(v, expected_opt['state'][key][k]) for k, v in item.items())
        assert 0 < record['global_clip_factor'] <= 1
        for name in ['weight', 'bias']:
            assert abs(record['root_gradient_post_clip'][name] - record['root_gradient_pre_clip'][name] * record['global_clip_factor']) < 1e-6
        other = trainer.model.backbone.weight
        other_state = copy.deepcopy(trainer.optimizer.state[other])
        reset = ResetRootHead(trainer.model.occ_head, trainer.optimizer)
        assert reset['optimizer_states_removed'] == 2
        assert all(torch.equal(v, trainer.optimizer.state[other][k]) for k, v in other_state.items())
        for arm, mode in [('original_resume', 'none'), ('reset_none', 'none'), ('reset_layer_norm', 'layer_norm')]:
            directory = root / arm
            trainer = Build(directory)
            source = root / f'{arm}_source.pth'
            save_checkpoint(trainer.model, source, trainer.optimizer, trainer.lr_scheduler,
                            {'epoch': 20, 'iter': 80, 'task_names': ['det', 'terrain', 'occ']})
            trainer.model.occ_head.root_input_normalization = mode
            cfg = SimpleNamespace(root_diagnostic={'arm': arm, 'max_steps': 2})
            cfg.get = lambda key, default=None: default
            args = SimpleNamespace(resume=str(source), validate_only=False, init_checkpoint=None, stop_after_epoch=None)
            with patch('root_diagnostic_runtime.PrepareEpochSampling') as sampling, patch('root_diagnostic_runtime.SamplingDefinition', return_value=None):
                RunRootDiagnostic(trainer, cfg, args)
                sampling.assert_called_once()
            result = json.loads((directory / 'root_diagnostic_result.json').read_text())
            checkpoint = torch.load(result['checkpoint'], map_location='cpu', weights_only=False)
            assert result['b_complete'] and result['completed_steps'] == 2
            assert checkpoint['meta']['epoch'] == 20 and checkpoint['meta']['iter'] == 82
            assert checkpoint['meta']['diagnostic_steps'] == 2 and checkpoint['meta']['b_partial_epoch']
            assert checkpoint['scheduler']['last_epoch'] == 82 and checkpoint['scheduler']['total_steps'] == 192
            assert len((directory / 'root_diagnostic_rank0.jsonl').read_text().splitlines()) == 2
        tracked = Build(root / 'tracked')
        control = Build(root / 'control')
        control.model.load_state_dict(copy.deepcopy(tracked.model.state_dict()))
        control.optimizer.load_state_dict(copy.deepcopy(tracked.optimizer.state_dict()))
        control.lr_scheduler.load_state_dict(copy.deepcopy(tracked.lr_scheduler.state_dict()))
        TrackRootOptimization(tracked)
        for _ in range(20):
            batch = next(iter(Loader()))
            tracked.train_iter(batch)
            control.train_iter(batch)
        assert all(torch.equal(v, control.model.state_dict()[k]) for k, v in tracked.model.state_dict().items())
        rows = [json.loads(line) for line in (root / 'tracked/root_optimization_rank0.jsonl').read_text().splitlines()]
        samples = [row for row in rows if row['status'] == 'sample']
        assert [row['global_step'] for row in samples] == [90, 100]
        assert all(row['effective_root_bce_weight'] == 1. for row in samples)
        assert 'value_after' in samples[-1]['root_parameter_updates']['bias']
        disabled = Build(root / 'disabled')
        disabled.model.occ_head.root_score = None
        TrackRootOptimization(disabled)
        disabled_rows = [json.loads(line) for line in (root / 'disabled/root_optimization_rank0.jsonl').read_text().splitlines()]
        assert len(disabled_rows) == 1 and disabled_rows[0]['status'] == 'disabled'
        no_tracking = Build(root / 'default')
        no_tracking.model.occ_head.b_collect_root_diagnostics = False
        original_train_iter = no_tracking.train_iter
        TrackRootOptimization(no_tracking)
        assert no_tracking.train_iter == original_train_iter
        assert not (root / 'default/root_optimization_rank0.jsonl').exists()
    print('PASS wrapped step bitwise model/Adam parity; clip factors; root-only state reset; three finite arms; partial checkpoint; formal every10 logging parity; disabled head; default no-op')


if __name__ == '__main__':
    Main()
