"""CPU checks for global epoch subsets, temporal anchors and epoch-boundary resume."""
import ast
import copy
import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from dynamic_sampler import EpochSubsetSampler, PrepareEpochSampling, SamplingDefinition, ValidateSamplingResume


class TokenDataset(Dataset):
    def __init__(self, count=40):
        self.infos = [{'token':str(index)} for index in range(count)]
        self.sample_fraction = .5
        self.sample_seed = 0
        self.b_dynamic_epoch_sampling = True

    def __len__(self):
        return len(self.infos)

    def __getitem__(self, index):
        return index, float(np.random.random()), float(random.random())

    @staticmethod
    def CollateBatch(rows):
        return rows


def NativeFunctions(filename, names, namespace):
    tree = ast.parse((PROJECT / filename).read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=functions, type_ignores=[]), filename, 'exec'), namespace)
    return namespace


def CheckCPURank(rank, directory):
    torch.set_num_threads(1)
    torch.distributed.init_process_group('gloo',init_method='file://'+str(Path(directory)/'rendezvous'),rank=rank,world_size=8)
    try:
        dataset=TokenDataset(40)
        sampler=EpochSubsetSampler(dataset)
        loader=DataLoader(dataset,sampler=sampler,generator=torch.Generator());loader.epoch_seed=0
        PrepareEpochSampling(loader,2,rank,Path(directory)/'checkpoints')
        (Path(directory)/f'rank_{rank}.json').write_text(json.dumps(list(sampler)))
    finally:
        torch.distributed.destroy_process_group()


class DynamicSamplerTests(unittest.TestCase):
    def TestEightRankSelectionAndPadding(self):
        dataset = TokenDataset(23688)
        samplers = [EpochSubsetSampler(dataset,8,rank) for rank in range(8)]
        for sampler in samplers: sampler.SetEpoch(3)
        self.assertEqual({tuple(sampler.SelectedIndices()) for sampler in samplers}, {tuple(samplers[0].SelectedIndices())})
        rows = [list(sampler) for sampler in samplers]
        flat = [index for row in rows for index in row]
        self.assertEqual([len(row) for row in rows],[1481]*8)
        self.assertEqual(len(flat),11848)
        self.assertEqual(len(set(flat)),11844)
        self.assertEqual(samplers[0].EpochRecord()['padding_count'],4)
        previous = samplers[0].SelectedIndices()
        samplers[0].SetEpoch(4)
        self.assertNotEqual(previous,samplers[0].SelectedIndices())
        self.assertGreater(samplers[0].EpochRecord()['union_count'],11844)
        restored = EpochSubsetSampler(dataset,8,0); restored.SetEpoch(4)
        self.assertEqual(restored.EpochRecord(),samplers[0].EpochRecord())
        self.assertEqual(len(dataset.infos),23688)

    def TestPoolRetentionAndTemporalAnchors(self):
        tree=ast.parse((PROJECT/'WaymoDatasetV2.py').read_text())
        init=next(node for node in next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='BaseDataset').body if isinstance(node,ast.FunctionDef) and node.name=='__init__')
        start=next(index for index,node in enumerate(init.body) if isinstance(node,ast.Assign) and any(isinstance(target,ast.Attribute) and target.attr=='sample_fraction' for target in node.targets))
        statements=ast.Module(body=init.body[start:],type_ignores=[])
        infos=[{'token':str(index),'sequence_name':'seq','original_sample_idx':index} for index in range(40)]
        for b_dynamic,count in ((True,40),(False,20)):
            dataset=SimpleNamespace(infos=infos.copy())
            exec(compile(statements,'native dataset sampling','exec'),{'self':dataset,'cfg':{'sample_fraction':.5,'sample_seed':0,'b_dynamic_epoch_sampling':b_dynamic},'np':np})
            self.assertEqual(len(dataset.infos),count)
        temporal=ast.parse((PROJECT/'WaymoDataset_multi.py').read_text())
        cls=next(node for node in temporal.body if isinstance(node,ast.ClassDef) and node.name=='WaymoDatasetMulti')
        methods=[node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name in ('BuildAnchorList','GetTargetFullIndex')]
        helper=ast.ClassDef(name='Clip',bases=[],keywords=[],body=methods,decorator_list=[])
        namespace={};exec(compile(ast.fix_missing_locations(ast.Module(body=[helper],type_ignores=[])),'native temporal anchors','exec'),namespace)
        clip=namespace['Clip']();clip.infos=infos;clip.full_sequence_info_dict={'seq':[{}]*100};clip.full_info_path='full_history'
        anchors=clip.BuildAnchorList()
        self.assertEqual(len(anchors),40)
        self.assertEqual([anchor['target_full_idx'] for anchor in anchors],list(range(40)))
        self.assertEqual(len(clip.full_sequence_info_dict['seq']),100)

    def TestBothNativeLoadersAndWorkerResume(self):
        for filename in ('LYF_train_ODSS.py','LYF_train_ODSS_multi.py'):
            namespace=NativeFunctions(filename,{'BuildDataLoader','BuildSeededDataLoader','SeedWorker'},dict(torch=torch,np=np,random=random,DataLoader=DataLoader,DistributedSampler=DistributedSampler,EpochSubsetSampler=EpochSubsetSampler))
            with tempfile.TemporaryDirectory() as tmp:
                dataset=TokenDataset()
                loader=namespace['BuildSeededDataLoader'](dataset,seed=12,rank=0,batch_size=1,num_workers=2,shuffle=True)
                self.assertIsInstance(loader.sampler,EpochSubsetSampler)
                self.assertEqual(len(loader),20)
                PrepareEpochSampling(loader,0,0,Path(tmp)/'checkpoints')
                list(loader)
                PrepareEpochSampling(loader,1,0,Path(tmp)/'checkpoints')
                continuous=list(loader)
                restored=namespace['BuildSeededDataLoader'](dataset,seed=12,rank=0,batch_size=1,num_workers=2,shuffle=True)
                PrepareEpochSampling(restored,1,0,Path(tmp)/'checkpoints')
                self.assertEqual(continuous,list(restored))
                record=json.loads((Path(tmp)/'epoch_sampling/epoch_002.json').read_text())
                self.assertEqual(record['selected_count'],20)
                self.assertGreater(record['union_count'],20)
                record['selected_sha256']='wrong';(Path(tmp)/'epoch_sampling/epoch_002.json').write_text(json.dumps(record))
                with self.assertRaisesRegex(RuntimeError,'refusing overwrite'): PrepareEpochSampling(restored,1,0,Path(tmp)/'checkpoints')
                dataset.b_dynamic_epoch_sampling=False
                static=namespace['BuildSeededDataLoader'](dataset,seed=12,rank=0,batch_size=1,num_workers=0,shuffle=True)
                self.assertNotIsInstance(static.sampler,EpochSubsetSampler)

    def TestBothTrainerEpochHooksAndPoolManifest(self):
        from tqdm import tqdm
        from run_options import SaveRunManifest
        for filename in ('trainer/trainer/trainer.py','trainer/trainer/trainer_multi.py'):
            tree=ast.parse((PROJECT/filename).read_text())
            trainer_class=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='Trainer')
            method=next(node for node in trainer_class.body if isinstance(node,ast.FunctionDef) and node.name=='train_epoch')
            namespace={'PrepareEpochSampling':PrepareEpochSampling,'tqdm':lambda *args,**kwargs:tqdm(*args,disable=True,**kwargs)}
            exec(compile(ast.Module(body=[method],type_ignores=[]),filename,'exec'),namespace)
            with tempfile.TemporaryDirectory() as tmp:
                dataset=TokenDataset()
                loader=DataLoader(dataset,sampler=EpochSubsetSampler(dataset),generator=torch.Generator());loader.epoch_seed=0
                trainer=SimpleNamespace(model=SimpleNamespace(train=lambda:None),train_dataloader=loader,epoch=5,max_epochs=48,rank=0,world_size=1,checkpoint_dir=str(Path(tmp)/'checkpoints'),b_freeze_x3_encoder=False,train_iter=lambda batch:(0,[]),_build_train_tqdm_str=lambda value,logs:'')
                namespace['train_epoch'](trainer)
                self.assertEqual(trainer.epoch,6)
                self.assertEqual(loader.sampler.epoch,5)
                self.assertTrue((Path(tmp)/'epoch_sampling/epoch_006.json').is_file())
        with tempfile.TemporaryDirectory() as tmp,patch.dict('os.environ',{'WAYMO_DATA_ROOT':tmp}),patch('run_options.subprocess.check_output',return_value=''):
            dataset=TokenDataset()
            SaveRunManifest(tmp,{},SimpleNamespace(config='unused',init_checkpoint=None,resume=None),dataset,None)
            manifest=json.loads((Path(tmp)/'manifest.json').read_text())
            self.assertEqual(manifest['splits']['train']['count'],40)
            self.assertEqual(len(json.loads((Path(tmp)/'train_tokens.json').read_text())),40)
            self.assertEqual(manifest['epoch_sampling']['epoch_sample_fraction'],.5)
            self.assertTrue(manifest['epoch_sampling']['b_train_tokens_are_candidate_pool'])

    def TestCPUCollectiveEightRanks(self):
        with tempfile.TemporaryDirectory() as tmp:
            torch.multiprocessing.start_processes(CheckCPURank,args=(tmp,),nprocs=8,start_method='fork',join=True)
            rows=[json.loads((Path(tmp)/f'rank_{rank}.json').read_text()) for rank in range(8)]
            self.assertEqual([len(row) for row in rows],[3]*8)
            self.assertEqual(len(set(index for row in rows for index in row)),20)
            record=json.loads((Path(tmp)/'epoch_sampling/epoch_003.json').read_text())
            self.assertEqual(record['world_size'],8)
            self.assertEqual(record['padding_count'],4)

    def TestDDPMismatchRejected(self):
        loader=DataLoader(TokenDataset(),sampler=EpochSubsetSampler(TokenDataset(),2,0),generator=torch.Generator());loader.epoch_seed=0
        def GatherDifferent(destination, source): destination[:]=[source,'mismatch']
        with tempfile.TemporaryDirectory() as tmp,patch('dynamic_sampler.dist.is_initialized',return_value=True),patch('dynamic_sampler.dist.get_world_size',return_value=2),patch('dynamic_sampler.dist.all_gather_object',side_effect=GatherDifferent):
            with self.assertRaisesRegex(RuntimeError,'differs across DDP ranks'): PrepareEpochSampling(loader,0,0,Path(tmp)/'checkpoints')

    def TestOneCycleResumeAtFiveAndTwentyFive(self):
        dataset=TokenDataset(32)
        def Build():
            loader=DataLoader(dataset,sampler=EpochSubsetSampler(dataset,8,0),generator=torch.Generator());loader.epoch_seed=0
            parameter=torch.nn.Parameter(torch.tensor([1.]));optimizer=torch.optim.Adam([parameter],lr=.001)
            scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=.001,epochs=48,steps_per_epoch=len(loader))
            return loader,parameter,optimizer,scheduler
        for epoch in (5,25):
            loader,parameter,optimizer,scheduler=Build()
            for _ in range(epoch*len(loader)):
                optimizer.zero_grad();parameter.square().sum().backward();optimizer.step();scheduler.step()
            checkpoint=copy.deepcopy({'meta':{'epoch':epoch,'iter':epoch*len(loader),'epoch_sampling':SamplingDefinition(loader)},'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict()})
            restored,new_parameter,new_optimizer,new_scheduler=Build()
            ValidateSamplingResume(restored,checkpoint,new_scheduler)
            new_parameter.data.copy_(parameter.data);new_optimizer.load_state_dict(checkpoint['optimizer']);new_scheduler.load_state_dict(checkpoint['scheduler'])
            for sampler in (loader.sampler,restored.sampler): sampler.SetEpoch(epoch)
            self.assertEqual(loader.sampler.EpochRecord(),restored.sampler.EpochRecord())
            for _ in range(len(loader)):
                for weight,opt,cycle in ((parameter,optimizer,scheduler),(new_parameter,new_optimizer,new_scheduler)):
                    opt.zero_grad();weight.square().sum().backward();opt.step();cycle.step()
                self.assertEqual(optimizer.param_groups[0]['lr'],new_optimizer.param_groups[0]['lr'])
                self.assertTrue(torch.equal(parameter,new_parameter))
            self.assertEqual(new_scheduler.total_steps,96)
            bad=copy.deepcopy(checkpoint);bad['meta']['epoch_sampling']['seed']=1
            with self.assertRaisesRegex(RuntimeError,'definition'): ValidateSamplingResume(restored,bad,new_scheduler)
            bad=copy.deepcopy(checkpoint);bad['meta']['iter']+=1
            with self.assertRaisesRegex(RuntimeError,'epoch boundary'): ValidateSamplingResume(restored,bad,new_scheduler)
            bad=copy.deepcopy(checkpoint);bad['scheduler']['total_steps']+=1
            with self.assertRaisesRegex(RuntimeError,'scheduler'): ValidateSamplingResume(restored,bad,new_scheduler)
            restored.epoch_seed=1
            with self.assertRaisesRegex(RuntimeError,'definition'): ValidateSamplingResume(restored,checkpoint,new_scheduler)
            restored.epoch_seed=0
            wrong_world=DataLoader(dataset,sampler=EpochSubsetSampler(dataset,4,0),generator=torch.Generator());wrong_world.epoch_seed=0
            with self.assertRaisesRegex(RuntimeError,'definition'): ValidateSamplingResume(wrong_world,checkpoint,new_scheduler)
            old=copy.deepcopy(checkpoint);old['meta'].pop('epoch_sampling')
            with self.assertRaisesRegex(RuntimeError,'definition'): ValidateSamplingResume(restored,old,new_scheduler)


if __name__=='__main__':
    loader=unittest.TestLoader();loader.testMethodPrefix='Test'
    unittest.main(testLoader=loader)
