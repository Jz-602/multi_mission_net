import torch
import torch.nn as nn

import inspect

import PillarEncoder
import SparseResnet
import ASPPNeck
import CenterHead

class SingleStageDetector(nn.Module):

    def __init__(self, cfg, common_variables):
        super(SingleStageDetector, self).__init__()

        self.reader = None
        self.backbone = None
        self.neck = None
        self.head = None


        if 1: # the following will generate a EasyDict, otherwise it will generate a dict
            self.post_processing = cfg.post_processing
            self.post_processing.update(common_variables)
        else:
            self.post_processing = {**cfg.post_processing, **common_variables} # generate a dict


        if cfg.reader is not None:
            arguments = {}
            func_name = None
            for name, value in cfg.reader.items():
                if name == '_target_':
                    func_name = value
                else:
                    arguments.update({name: value})

            required_arguments = inspect.getfullargspec(eval(func_name))
            for argument_name in required_arguments.args:
                if (not argument_name in arguments) and (argument_name in common_variables):
                    arguments[argument_name] = common_variables[argument_name]

            self.reader = eval(func_name)(**arguments)

        if cfg.backbone is not None:
            arguments = {}
            func_name = None
            for name, value in cfg.backbone.items():
                if name == '_target_':
                    func_name = value
                else:
                    arguments.update({name: value})

            required_arguments = inspect.getfullargspec(eval(func_name))
            for argument_name in required_arguments.args:
                if (not argument_name in arguments) and (argument_name in common_variables):
                    arguments[argument_name] = common_variables[argument_name]

            self.backbone = eval(func_name)(**arguments)

        if cfg.neck is not None:
            arguments = {}
            func_name = None
            for name, value in cfg.neck.items():
                if name == '_target_':
                    func_name = value
                else:
                    arguments.update({name: value})

            required_arguments = inspect.getfullargspec(eval(func_name))
            for argument_name in required_arguments.args:
                if (not argument_name in arguments) and (argument_name in common_variables):
                    arguments[argument_name] = common_variables[argument_name]

            self.neck = eval(func_name)(**arguments)

        if cfg.head is not None:
            arguments = {}
            func_name = None
            for name, value in cfg.head.items():
                if name == '_target_':
                    func_name = value
                else:
                    arguments.update({name: value})

            required_arguments = inspect.getfullargspec(eval(func_name))
            for argument_name in required_arguments.args:
                if (not argument_name in arguments) and (argument_name in common_variables):
                    arguments[argument_name] = common_variables[argument_name]

            self.head = eval(func_name)(**arguments)


    def _forward(self, example):
        points = example['points']

        x = self.reader(points)
        if self.backbone is not None:
            x = self.backbone(*x)
        if self.neck is not None:
            x = self.neck(x)

        return self.head(x)

    def forward(self, example):
        if  self.training:
            return self.training_step(example)
        else:
            return self.validation_step(example)

    def training_step(self, example):
        preds = self._forward(example)
        loss, log_vars = self.head.loss(example, preds)

        return loss, log_vars

    @torch.no_grad()
    def validation_step(self, example):
        preds = self._forward(example)
        outputs = self.head.predict(example, preds, self.post_processing)
        detections = {}
        for output in outputs:
            token = output["token"]
            for k, v in output.items():
                if k != "token":
                    output[k] = v.to(torch.device("cpu"))

            detections.update({token: output})
        return detections
