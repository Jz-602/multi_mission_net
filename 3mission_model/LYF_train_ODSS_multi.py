from dynamic_sampler import EpochSubsetSampler
from run_options import ParseRunOptions, CheckRunConfig, LimitDataset, SaveRunManifest
import datetime
import logging
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import torch
import torch.distributed as dist
import yaml
from easydict import EasyDict
from tensorboardX import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from WaymoDataset_multi import WaymoDatasetMulti
from multi_mission_model_memory import MultiMissionModel
from trainer.trainer.trainer_multi import FreezeX3Encoder, Trainer


def HandleCudaLaunch():
    if dist.is_initialized() or ("LOCAL_RANK" in os.environ) or ("__internal_launch__" in sys.argv[1:]):
        return

    cuda_id_list = []
    b_collect_following_cuda_ids = False
    for argument in sys.argv[1:]:
        if argument == "__internal_launch__":
            continue

        if argument.startswith("CUDA="):
            b_collect_following_cuda_ids = True
            first_cuda_text = argument.split("=", 1)[1].strip()
            if first_cuda_text != "":
                for cuda_id in first_cuda_text.split(","):
                    cuda_id = cuda_id.strip()
                    if cuda_id != "":
                        cuda_id_list.append(cuda_id)
            continue

        if b_collect_following_cuda_ids:
            normalized_argument = argument.replace(",", "").strip()
            if normalized_argument.isdigit():
                cuda_id = argument.strip().strip(",")
                if cuda_id != "":
                    cuda_id_list.append(cuda_id)
                continue
            b_collect_following_cuda_ids = False

    if len(cuda_id_list) == 0:
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            print("[Launch] CUDA not specified, default to GPU 0.")
        return

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(cuda_id_list)

    if len(cuda_id_list) == 1:
        print(f"[Launch] single GPU mode, use GPU {cuda_id_list[0]}.")
        return

    print(f"[Launch] multi-GPU mode, use GPUs {','.join(cuda_id_list)}.")
    launch_command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={len(cuda_id_list)}",
        str(Path(__file__).resolve()),
        "__internal_launch__",
    ]
    launch_command.extend(sys.argv[1:])
    raise SystemExit(subprocess.call(launch_command, env=os.environ.copy()))


def MergeNewConfig(config, new_config):
    for key, value in new_config.items():
        if not isinstance(value, dict):
            if isinstance(value, str):
                value = os.path.expandvars(value)
            config[key] = value
            continue
        if key not in config:
            config[key] = EasyDict()
        MergeNewConfig(config[key], value)

    return config


def CfgFromYamlFile(cfg_file, config):
    cfg_path = Path(cfg_file).resolve()
    with cfg_path.open() as file:
        new_config = yaml.safe_load(file)
    base = new_config.pop('_BASE_CONFIG_', None)
    if base is not None:
        CfgFromYamlFile(cfg_path.parent / base, config)
    return MergeNewConfig(config=config, new_config=new_config)


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
    if getattr(dataset, "b_dynamic_epoch_sampling", False):
        if not shuffle:
            raise ValueError("Dynamic epoch sampling is only supported for shuffled training")
        sampler = EpochSubsetSampler(dataset)
    elif distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle)

    return DataLoader(
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
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def BuildSeededDataLoader(dataset, seed, rank, **kwargs):
    generator = torch.Generator()
    generator.manual_seed(int(seed) + int(rank))
    loader = BuildDataLoader(
        dataset,
        worker_init_fn=SeedWorker,
        generator=generator,
        **kwargs,
    )
    loader.epoch_seed = int(seed)
    return loader


def SetupDistributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    b_distributed = world_size > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))

    if b_distributed:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)

    return b_distributed, rank, local_rank, world_size


def CleanupDistributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def Instantiate(cfg, **kwargs):
    arguments = {}
    func_name = None
    for key, value in cfg.items():
        if key == "_target_":
            func_name = value
        else:
            arguments[key] = value

    arguments.update(**kwargs)
    return eval(func_name)(**arguments)


def Main():
    HandleCudaLaunch()
    b_distributed, rank, local_rank, world_size = SetupDistributed()

    cfg = EasyDict()
    args = ParseRunOptions(Path(__file__).resolve().parent / 'LYF_ODSS_config_multi.yaml')
    cfg_file = Path(args.config)
    cfg = CfgFromYamlFile(cfg_file, cfg)
    CheckRunConfig(cfg)
    cfg.common_variables.ignore_index = cfg.model.ignore_index

    random_seed = int(cfg.get("random_seed", 0))
    SetRandomSeed(random_seed, rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    ## 多帧训练集：一个 sample 以带语义标签的 target frame 为监督锚点，前序 full Waymo 帧只 warmup memory。
    train_dataset = WaymoDatasetMulti(cfg.data.train_dataset, cfg.common_variables)
    LimitDataset(train_dataset, cfg, 'train')
    if rank == 0:
        print(f"[Dataset] train samples: {len(train_dataset)}")
    train_dataloader = BuildSeededDataLoader(
        train_dataset,
        seed=random_seed,
        rank=rank,
        distributed=b_distributed,
        **cfg.dataloader.train,
    )

    b_has_val_dataset = "val_dataset" in cfg.data and cfg.data.val_dataset is not None
    if b_has_val_dataset:
        ## 验证集也使用多帧 dataset，当前只统计最后一个 target frame。
        val_dataset = WaymoDatasetMulti(cfg.data.val_dataset, cfg.common_variables)
        LimitDataset(val_dataset, cfg, 'val')
        if rank == 0:
            print(f"[Dataset] val samples: {len(val_dataset)}")
        if len(val_dataset) > 0:
            val_dataloader = BuildSeededDataLoader(
                val_dataset,
                seed=random_seed + 100000,
                rank=rank,
                distributed=b_distributed,
                **cfg.dataloader.val,
            )
        else:
            if rank == 0:
                print(f"[Validation] {cfg.data.val_dataset.info_path} contains 0 samples, disable validation.")
            val_dataloader = None
    else:
        val_dataloader = None

    cfg.model.class_names = train_dataset.class_names
    cfg.model.ignore_index = train_dataset.ignore_index
    cfg.model.tasks = [list(task) for task in cfg.common_variables.tasks]
    cfg.model.out_size_factor = list(cfg.common_variables.out_size_factor)
    cfg.model.voxel_size = list(cfg.common_variables.voxel_size)
    cfg.model.point_cloud_range = list(cfg.common_variables.pc_range)

    model_class = MultiMissionModel
    if 'x4_detection_source' in cfg.model:
        from multi_mission_model_memory_x4 import MultiMissionModel as X4Model
        model_class = X4Model
    model = model_class(cfg.model).to(device)

    if b_distributed and cfg.get("sync_batchnorm", False):
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if bool(cfg.trainer.get("freeze_x3_encoder", False)):
        FreezeX3Encoder(model)

    if b_distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

    if rank == 0:
        print(model)
        print(sum(param.nelement() for param in model.parameters()))

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if len(trainable_params) == 0:
        raise RuntimeError("No trainable parameters left after applying freeze settings.")
    optimizer = Instantiate(cfg.optimizer, params=trainable_params)
    cfg.scheduler.epochs = cfg.trainer.max_epochs
    lr_scheduler = Instantiate(
        cfg.scheduler,
        optimizer=optimizer,
        steps_per_epoch=len(train_dataloader),
    )

    project_dir = Path(__file__).resolve().parent
    output_dir = Path(args.output or cfg.get('output_dir', project_dir / 'log' / 'three_task_temporal'))
    logger = logging.getLogger()
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        SaveRunManifest(output_dir, cfg, args, train_dataset, val_dataset)
        log_file = output_dir / (
            "log_train_multi_%s.txt" % datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(),
            ],
            force=True,
        )
        logger.info("Logging to %s", log_file)
        logger.info("Dataset: train=%d (three-task targets) | validation=%d (three-task targets)", len(train_dataset), len(val_dataset))
        logger.info("Seed=%d | world_size=%d | batch_per_gpu=%d | NMS=%s", random_seed, world_size, cfg.dataloader.train.batch_size, cfg.model.post_processing.nms.nms_iou_threshold)
        tb_log = SummaryWriter(log_dir=str(output_dir / "tensorboard_minkunet_seg_multi"))
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
        checkpoint_dir=str(output_dir / 'checkpoints'),
        **cfg.trainer,
    )

    try:
        if cfg.get('collect_training_memory', False):
            from training_memory import TrackTrainingMemory
            TrackTrainingMemory(trainer)
        if args.resume:
            trainer.resume(args.resume)
        elif args.init_checkpoint:
            trainer.resume_from_single_model(args.init_checkpoint)
        elif True and not cfg.get('run', {}).get('allow_random_temporal', False):
            raise ValueError('Temporal training requires --init-checkpoint from single-frame training or --resume.')
        if (cfg.model.occ.get('b_collect_root_diagnostics', False)
                and not cfg.get('root_diagnostic') and not args.validate_only):
            from root_diagnostic_runtime import TrackRootOptimization
            TrackRootOptimization(trainer)
        if cfg.get('root_diagnostic'):
            from root_diagnostic_runtime import RunRootDiagnostic
            RunRootDiagnostic(trainer, cfg, args)
        elif cfg.get('thesis_v1'):
            from thesis_v1_runtime import RunThesisEpoch
            RunThesisEpoch(trainer, cfg, args)
        elif cfg.get('detection_screen'):
            from detection_screen_runtime import RunScreenStep
            RunScreenStep(trainer, cfg, args)
        elif args.validate_only:
            if not (args.resume or args.init_checkpoint):
                raise ValueError('--validate-only requires a checkpoint')
            trainer.val_epoch()
        else:
            if args.stop_after_epoch is not None:
                if args.stop_after_epoch <= trainer.epoch:
                    raise ValueError('--stop-after-epoch must exceed the restored epoch')
                trainer.max_epochs = min(trainer.max_epochs, args.stop_after_epoch)
            trainer.fit()
    finally:
        if tb_log is not None:
            tb_log.close()
        CleanupDistributed()


if __name__ == "__main__":
    Main()
