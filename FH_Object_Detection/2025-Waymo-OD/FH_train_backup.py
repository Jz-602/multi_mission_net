import Augmentation
import inspect
import yaml
from importlib import import_module
from easydict import EasyDict
from torch.utils.data import DataLoader
from collections import defaultdict
import torch
import numpy as np
import logging
import random
from tensorboardX import SummaryWriter
from WaymoDatasetV2 import WaymoDataset
from trainer.trainer import Trainer
import datetime
from pathlib import Path
from LYF_all_Model import LYF_all_Model



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
            config[key] = val
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

def build_dataloader(dataset, batch_size, num_workers, shuffle, pin_memory=False):

    sampler = None

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None and shuffle),
        num_workers=num_workers,
        collate_fn=dataset.CollateBatch,
        pin_memory=pin_memory,
    )

    return data_loader



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

    random.seed(0)
    np.random.seed(0)

    ## 1) 读配置：YAML → cfg（EasyDict）
    cfg = EasyDict()
    cfg_file = 'FH_waymo_det_pp18_aspp_iou_car_sp.yaml'
    cfg_from_yaml_file(cfg_file, cfg)

    ## 2) 构造数据集：WaymoDataset(...)    3) DataLoader 的关键：你自定义的collate(batch_list)
    train_dataset = WaymoDataset(cfg.data.train_dataset, cfg.common_variables)
    train_dataloader = build_dataloader(train_dataset, **cfg.dataloader.train)

    if 'val_dataset' in cfg.data:
        val_dataset = WaymoDataset(cfg.data.val_dataset, cfg.common_variables)
        val_dataloader = build_dataloader(val_dataset, **cfg.dataloader.val)
    else:
        val_dataloader = None

    ## 4) 构造模型
    # model = SingleStageDetector(cfg.model, cfg.common_variables)
    model = LYF_all_Model(cfg.model, cfg.common_variables)
    print(model)
#    model = torch.compile(model)

    model = model.cuda()
    print(sum([param.nelement() for param in model.parameters()]))

    ## 5) 优化器/调度器：FH_Instantiate 动态实例化
    optimizer = FH_Instantiate(cfg.optimizer, params=model.parameters())

    cfg.scheduler.epochs = cfg.trainer.max_epochs
    lr_scheduler = FH_Instantiate(cfg.scheduler, optimizer=optimizer, steps_per_epoch=len(
        train_dataloader))

    logger = logging.getLogger()

    output_dir = Path('./log/')
    log_file = output_dir / ('log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard'))


    logging.basicConfig(filename=log_file, level=logging.INFO)
    trainer = Trainer(model, train_dataloader, val_dataloader, optimizer, lr_scheduler, logger=logger, tb_log=tb_log, **cfg.trainer)
    ## 6) Trainer 驱动训练：核心训练循环在 Trainer.fit()

    if 1:
        # trainer.resume('./checkpoints/epoch_36.pth')

        trainer.fit()
    else:
        trainer.load_checkpoint('./chkpts/epoch_20.pth')
        trainer.val_epoch()

