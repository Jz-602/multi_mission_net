"""独立CPU恢复/门禁检查；mock不代表真实CUDA训练或评价。"""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'tools'))
from test_detection_screen_gate import Record
from run_options import HashFile, WriteJsonAtomic
import run_selected_x4_long48 as runner


class SelectedLong48Tests(unittest.TestCase):
    def Checkpoint(self, epoch=5):
        return dict(meta=dict(epoch=epoch, iter=epoch*1481),
                    scheduler=dict(total_steps=71088, last_epoch=epoch*1481),
                    optimizer=dict(state={i:dict(step=epoch*1481) for i in range(339)}),
                    state_dict={'weight':torch.tensor([1.])})

    def TestCheckpointBoundariesAndState(self):
        for epoch in (5, 6, 48):
            with patch.object(runner.torch, 'load', return_value=self.Checkpoint(epoch)):
                self.assertEqual(runner.CheckCheckpoint(Path('unused')), epoch)
        for epoch in (4, 49):
            with patch.object(runner.torch, 'load', return_value=self.Checkpoint(epoch)):
                with self.assertRaises(ValueError):
                    runner.CheckCheckpoint(Path('unused'))

    def TestRejectResetOptimizerOrShortScheduler(self):
        for section, key, value in (('scheduler','total_steps',7405),
                                    ('scheduler','last_epoch',0),
                                    ('meta','iter',0)):
            checkpoint=self.Checkpoint(); checkpoint[section][key]=value
            with patch.object(runner.torch,'load',return_value=checkpoint):
                with self.assertRaises(ValueError):
                    runner.CheckCheckpoint(Path('unused'))
        checkpoint=self.Checkpoint();checkpoint['optimizer']['state'][0]['step']=0
        with patch.object(runner.torch,'load',return_value=checkpoint):
            with self.assertRaises(ValueError):
                runner.CheckCheckpoint(Path('unused'))

    def TestRejectNonfiniteWeights(self):
        checkpoint=self.Checkpoint();checkpoint['state_dict']['weight'][0]=float('nan')
        with patch.object(runner.torch,'load',return_value=checkpoint):
            with self.assertRaises(ValueError):
                runner.CheckCheckpoint(Path('unused'))

    def TestRealConfigChangesOnlyAllowedExecutionFields(self):
        with patch.object(runner,'ConfigIdentity',return_value={'mock':'identity'}):
            self.assertEqual(runner.CheckConfig(), {'mock':'identity'})
        old=runner.LoadConfig(PROJECT/'configs/server_x4_screen_f.yaml')
        new=runner.LoadConfig(runner.CONFIG);new['trainer']['max_epochs']=53
        with patch.object(runner,'LoadConfig',side_effect=[old,new]):
            with self.assertRaises(ValueError):runner.CheckConfig()

    def SetupRun(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        root=Path(temporary.name);output=root/'output';source=root/'source'
        for directory in (output/'checkpoints', output/'validation',output/'acceptance',source):
            directory.mkdir(parents=True)
        baseline=root/'baseline.json';WriteJsonAtomic(baseline,Record((.6,.6,.6),0))
        splits={'train':{'count':23688,'sha256':'a'*64},'val':{'count':5973,'sha256':'b'*64}}
        for directory in (output,source):WriteJsonAtomic(directory/'manifest.json',{'splits':splits})
        (output/'checkpoints/epoch_5.pth').write_bytes(b'mock5')
        WriteJsonAtomic(output/'validation_metrics.json',Record((.6,.6,.6),5))
        identity={'config_sha256':'c'*64}
        WriteJsonAtomic(output/'long_protocol.json',dict(config_identity=identity,baseline_sha256=HashFile(baseline),split_signature=splits,source_manifest_sha256=HashFile(source/'manifest.json')))
        for name,value in [('OUTPUT',output),('SOURCE',source),('BASELINE',baseline),('PHASES',(6,10))]:
            p=patch.object(runner,name,value);p.start();self.addCleanup(p.stop)
        for name,value in [('CheckActive',None),('CheckConfig',identity),('ConfigIdentity',identity),('CheckRawUpdates',None)]:
            p=patch.object(runner,name,return_value=value);p.start();self.addCleanup(p.stop)
        return output

    def FakeImages(self, command, output, action):
        if '--split' not in command:raise AssertionError('Unexpected GPU training command')
        image=Path(command[command.index('--output')+1]);image.parent.mkdir(parents=True,exist_ok=True)
        image.write_bytes(b'x'*1200)
        if '--loss-log' in command:(image.parent/'loss_curves.png').write_bytes(b'x'*1200)

    def TestDetectionDropStopsBeforeNextPhase(self):
        output=self.SetupRun();(output/'checkpoints/epoch_6.pth').write_bytes(b'mock6')
        WriteJsonAtomic(output/'validation/epoch_6.json',Record((.55,.55,.55),6))
        with patch.object(runner,'CheckCheckpoint',return_value=6),patch.object(runner,'Execute',side_effect=self.FakeImages):
            runner.RunPhases()
        self.assertEqual(runner.Read(output/'stage_result.json')['status'],'early_stop')
        self.assertFalse((output/'checkpoints/epoch_10.pth').exists())

    def TestWrongEpochValidationRejected(self):
        output=self.SetupRun();(output/'checkpoints/epoch_6.pth').write_bytes(b'mock6')
        WriteJsonAtomic(output/'validation/epoch_6.json',Record((.6,.6,.6),5))
        with patch.object(runner,'CheckCheckpoint',return_value=6),patch.object(runner,'Execute',side_effect=self.FakeImages):
            with self.assertRaises(ValueError):runner.RunPhases()
        self.assertFalse((output/'acceptance/epoch_6.json').exists())

    def TestMissingValidationOnlyEvaluatesSavedEpoch(self):
        output=self.SetupRun();(output/'checkpoints/epoch_6.pth').write_bytes(b'mock6')
        calls=[]
        def Execute(command,directory,action):
            calls.append(command)
            if '--split' not in command:
                self.assertIn('--resume',command);self.assertIn('--validate-only',command)
                self.assertNotIn('--stop-after-epoch',command)
                WriteJsonAtomic(output/'validation_metrics.json',Record((.55,.55,.55),6))
            else:self.FakeImages(command,directory,action)
        with patch.object(runner,'CheckCheckpoint',return_value=6),patch.object(runner,'Execute',side_effect=Execute):runner.RunPhases()
        self.assertEqual(sum('--validate-only' in c for c in calls),1)
        self.assertEqual(runner.Read(output/'stage_result.json')['status'],'early_stop')

    def TestProtocolRejectsConfigBaselineAndTargetsDrift(self):
        output=self.SetupRun()
        runner.VerifyProtocol()
        with patch.object(runner,'ConfigIdentity',return_value={'changed':True}):
            with self.assertRaises(ValueError):runner.VerifyProtocol()
        original=runner.BASELINE.read_bytes()
        runner.BASELINE.write_text('{}')
        with self.assertRaises(ValueError):runner.VerifyProtocol()
        runner.BASELINE.write_bytes(original)
        manifest=runner.Read(output/'manifest.json');manifest['splits']['val']['sha256']='z'*64
        WriteJsonAtomic(output/'manifest.json',manifest)
        with self.assertRaises(ValueError):runner.VerifyProtocol()

    def TestMainRefusesFailedOrTerminalRun(self):
        output=self.SetupRun()
        with patch.object(sys,'argv',['run_selected_x4_long48.py']):
            for name in ('failed_action.json','stage_result.json'):
                WriteJsonAtomic(output/name,{})
                with patch.object(runner,'Prepare') as prepare:
                    with self.assertRaises(ValueError):runner.Main()
                    prepare.assert_not_called()
                (output/name).unlink()


if __name__=='__main__':
    loader=unittest.TestLoader();loader.testMethodPrefix='Test'
    result=unittest.TextTestRunner(verbosity=2).run(loader.loadTestsFromTestCase(SelectedLong48Tests))
    sys.exit(0 if result.wasSuccessful() else 1)
