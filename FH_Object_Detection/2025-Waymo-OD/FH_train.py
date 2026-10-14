import datetime
import inspect
import logging
import os
from pathlib import Path
import random
import subprocess
import sys
from collections import defaultdict
from importlib import import_module

import numpy as np
import yaml
from easydict import EasyDict
from tensorboardX import SummaryWriter

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import Augmentation
from LYF_all_Model import LYF_all_Model
from WaymoDatasetV2 import WaymoDataset
from trainer.trainer import Trainer

#分布式训练   python FH_train.py CUDA=1 2 3 5
def HandleCudaLaunch():
    if dist.is_initialized() or ('LOCAL_RANK' in os.environ) or ('__internal_launch__' in sys.argv[1:]):
        return

    cuda_id_list = []
    b_collect_following_cuda_ids = False
    for argument in sys.argv[1:]:
        if argument == '__internal_launch__':
            continue

        if argument.startswith('CUDA='):
            b_collect_following_cuda_ids = True
            first_cuda_text = argument.split('=', 1)[1].strip()
            if first_cuda_text != '':
                for cuda_id in first_cuda_text.split(','):
                    cuda_id = cuda_id.strip()
                    if cuda_id != '':
                        cuda_id_list.append(cuda_id)
            continue

        if b_collect_following_cuda_ids:
            normalized_argument = argument.replace(',', '').strip()
            if normalized_argument.isdigit():
                cuda_id = argument.strip().strip(',')
                if cuda_id != '':
                    cuda_id_list.append(cuda_id)
                continue
            b_collect_following_cuda_ids = False

    if len(cuda_id_list) == 0:
        if 'CUDA_VISIBLE_DEVICES' not in os.environ:
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            print('[Launch] CUDA not specified, default to GPU 0.')
        return

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(cuda_id_list)

    if len(cuda_id_list) == 1:
        print(f'[Launch] single GPU mode, use GPU {cuda_id_list[0]}.')
        return

    print(f"[Launch] multi-GPU mode, use GPUs {','.join(cuda_id_list)}.")
    launch_command = [
        sys.executable,
        '-m',
        'torch.distributed.run',
        f'--nproc_per_node={len(cuda_id_list)}',
        str(Path(__file__).resolve()),
        '__internal_launch__',
    ]
    raise SystemExit(subprocess.call(launch_command, env=os.environ.copy()))




def merge_new_config(config, new_config):
    if '_BASE_CONFIG_' in new_config:
        with open(new_config['_BASE_CONFIG_'], 'r') as f:
            try:
                yaml_config = yaml.load(f, Loader=yaml.FullLoader)
            except:
                yaml_config = yaml.load(f)
        config.update(EasyDict(yaml_config))

    for key, val in new_config.items():
        if not isinstance(val, dict):
            config[key] = os.path.expandvars(val) if isinstance(val, str) else val
            continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)

    return config


def cfg_from_yaml_file(cfg_file, config):
    with open(cfg_file, 'r') as f:
        try:
            new_config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.load(f)

        merge_new_config(config=config, new_config=new_config)

    return config

def BuildDataLoader(
    dataset,
    batch_size,
    num_workers,
    shuffle,
    pin_memory=False,
    distributed=False,
    worker_init_fn=None,
    generator=None,
):
    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle)

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None and shuffle),
        num_workers=num_workers,
        collate_fn=dataset.CollateBatch,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )

    return data_loader


def SeedWorker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def SetRandomSeed(seed, rank):
    rank_seed = int(seed) + int(rank)
    random.seed(rank_seed)
    np.random.seed(rank_seed)
    torch.manual_seed(rank_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rank_seed)


def BuildSeededDataLoader(dataset, seed, rank, **kwargs):
    generator = torch.Generator()
    generator.manual_seed(int(seed) + int(rank))
    return BuildDataLoader(
        dataset,
        worker_init_fn=SeedWorker,
        generator=generator,
        **kwargs,
    )


def setup_distributed():
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    b_distributed = world_size > 1
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    rank = int(os.environ.get('RANK', '0'))

    if b_distributed:
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(local_rank)

    return b_distributed, rank, local_rank, world_size


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()



def FH_Instantiate(cfg, **kargs):
    arguments = {}
    func_name = None
    for name, value in cfg.items():
        if name == '_target_':
            func_name = value
        else:
            arguments.update({name: value})

    arguments.update(**kargs)

    return eval(func_name)(**arguments)


if __name__ == '__main__':
    HandleCudaLaunch()
    b_distributed, rank, local_rank, world_size = setup_distributed()

    ## 1) 读配置：YAML → cfg（EasyDict）
    cfg = EasyDict()
    cfg_file = Path(__file__).resolve().parent / 'FH_waymo_det_pp18_aspp_iou_car_sp.yaml'
    cfg_from_yaml_file(cfg_file, cfg)
    random_seed = int(cfg.get('random_seed', 0))
    SetRandomSeed(random_seed, rank)

    ## 2) 构造数据集：WaymoDataset(...)    3) DataLoader 的关键：你自定义的collate(batch_list)
    train_dataset = WaymoDataset(cfg.data.train_dataset, cfg.common_variables)
    train_dataloader = BuildSeededDataLoader(
        train_dataset,
        seed=random_seed,
        rank=rank,
        distributed=b_distributed,
        **cfg.dataloader.train,
    )

    if 'val_dataset' in cfg.data:
        val_dataset = WaymoDataset(cfg.data.val_dataset, cfg.common_variables)
        val_dataloader = BuildSeededDataLoader(
            val_dataset,
            seed=random_seed + 100000,
            rank=rank,
            distributed=b_distributed,
            **cfg.dataloader.val,
        )
    else:
        val_dataloader = None

    ## 4) 构造模型
    model = LYF_all_Model(cfg.model, cfg.common_variables)
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    if b_distributed and cfg.get('sync_batchnorm', False):
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if b_distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

    if rank == 0:
        print(model)
        print(sum([param.nelement() for param in model.parameters()]))

    ## 5) 优化器/调度器：FH_Instantiate 动态实例化
    optimizer = FH_Instantiate(cfg.optimizer, params=model.parameters())

    cfg.scheduler.epochs = cfg.trainer.max_epochs
    lr_scheduler = FH_Instantiate(cfg.scheduler, optimizer=optimizer, steps_per_epoch=len(
        train_dataloader))

    project_dir = Path(__file__).resolve().parent
    output_dir = project_dir / 'log'
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger()

    if rank == 0:
        log_file = output_dir / ('log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(levelname)s %(message)s',
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
            force=True,
        )
        logger.info('Logging to %s', log_file)
        logger.info('Dataset: train=%d (semantic 50%% subset) | validation=%d (semantic subset)', len(train_dataset), len(val_dataset))
        logger.info('Seed=%d | world_size=%d | batch_per_gpu=%d | NMS=%s', random_seed, world_size, cfg.dataloader.train.batch_size, cfg.model.post_processing.nms.nms_iou_threshold)
        tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard'))
    else:
        tb_log = None

    trainer = Trainer(
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        lr_scheduler,
        logger=logger,
        tb_log=tb_log,
        checkpoint_dir=str(project_dir / 'checkpoints'),
        **cfg.trainer,
    )
    ## 6) Trainer 驱动训练：核心训练循环在 Trainer.fit()

    try:
        # 断点续训时取消下一行注释，trainer.fit() 保持启用。
        # trainer.resume(str(project_dir / 'checkpoints' / 'epoch_40.pth'))
        # 只验证时取消下面两行注释，并注释 trainer.fit()。
        # trainer.load_checkpoint(str(project_dir / 'checkpoints' / 'epoch_48.pth'))
        # trainer.val_epoch()
        trainer.fit()
    finally:
        if tb_log is not None:
            tb_log.close()
        cleanup_distributed()
