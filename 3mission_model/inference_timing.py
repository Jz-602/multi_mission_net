"""独立模块诊断：临时包装实际调用，不复制网络数据流，不用于正式 latency。"""
from functools import wraps
import sys
import time


class ModuleTiming:
    def __init__(self, model, cuda):
        self.model = model
        self.cuda = cuda
        self.restorations = []
        self.calls = []
        self.stack = []
        self.context = {}
        self.instrumented = []

    def Wrap(self, owner, attribute, name):
        original = getattr(owner, attribute)
        b_local = attribute in vars(owner)
        local_value = vars(owner).get(attribute)

        @wraps(original)
        def Timed(*args, **kwargs):
            call_id = len(self.calls)
            row = {'call_id': call_id, 'parent_call_id': self.stack[-1] if self.stack else None,
                   'name': name, 'context': dict(self.context), 'b_completed': False}
            start, end = self.cuda.Event(enable_timing=True), self.cuda.Event(enable_timing=True)
            self.calls.append((row, start, end))
            self.stack.append(call_id)
            start.record()
            row['started_perf_seconds'] = time.perf_counter()
            try:
                value = original(*args, **kwargs)
                row['b_completed'] = True
                return value
            finally:
                row['finished_perf_seconds'] = time.perf_counter()
                end.record()
                self.stack.pop()
        setattr(owner, attribute, Timed)
        self.restorations.append((owner, attribute, b_local, local_value))
        self.instrumented.append(name)

    def __enter__(self):
        try:
            # EncodeCurrent 的全局变量属于定义模块；X4 继承时仍包装实际定义处。
            encode_module = sys.modules[self.model.EncodeCurrent.__func__.__module__]
            self.Wrap(encode_module, 'BuildBatchVoxels', 'preprocess.BuildBatchVoxels')
            self.Wrap(self.model, 'EncodeCurrent', 'encode_to_x3_including_preprocess')
            for name in ('conv_input', 'encoder_stage1', 'encoder_stage2', 'encoder_stage3', 'encoder_stage4'):
                self.Wrap(getattr(self.model, name), 'forward', 'backbone.' + name)
            b_temporal = hasattr(self.model, 'x4_memory_transformer')
            if b_temporal:
                for name in ('AlignMemoryToCurrentX4', 'MemoryUpdate', 'SelectDetectionFeatures'):
                    self.Wrap(self.model, name, 'fusion.' + name)
                self.Wrap(self.model.x4_memory_transformer, 'forward', 'fusion.x4_attention')
                self.Wrap(self.model.terrain_bev_projector, 'forward', 'bev.terrain_projection')
            self.Wrap(self.model.detection_bev_projector, 'forward',
                      'bev.detection_projection' if b_temporal else 'bev.shared_projection')
            for name in ('detection_neck', 'detection_head', 'terrain_head', 'occ_head'):
                self.Wrap(getattr(self.model, name), 'forward', 'heads.' + name)
            self.Wrap(self.model, 'ForwardNetwork', 'network_inclusive')
            self.Wrap(self.model, 'ExportPredictions', 'export_inclusive')
            predict_module = sys.modules[self.model.detection_head.predict.__func__.__module__]
            self.Wrap(self.model.detection_head, 'predict', 'export.CenterHead.predict_inclusive')
            self.Wrap(predict_module, 'rotate_nms_pcdet', 'export.rotated_nms_inclusive')
            self.Wrap(self.model.occ_head, 'ExportPredictions', 'export.occ')
            return self
        except BaseException:
            self.Restore()
            raise

    def Restore(self):
        for owner, attribute, b_local, local_value in reversed(self.restorations):
            if b_local:
                setattr(owner, attribute, local_value)
            else:
                delattr(owner, attribute)
        self.restorations.clear()

    def __exit__(self, *args):
        self.Restore()

    def BeginIteration(self):
        if self.stack:
            raise RuntimeError('cannot reset module timing inside a call')
        self.calls.clear()

    def FinishIteration(self):
        """调用者已完成整段 CUDA synchronize；此处只读取 event，不增加模块同步。"""
        rows = []
        for row, start, end in self.calls:
            result = dict(row)
            result['host_inclusive_ms'] = 1000 * (row['finished_perf_seconds'] - row['started_perf_seconds'])
            result['cuda_stream_span_inclusive_ms'] = start.elapsed_time(end)
            rows.append(result)
        return rows

    def Definition(self):
        return {'instrumented': self.instrumented,
                'b_latency_eligible': False,
                'scope': 'Separate diagnostic pass of actual production calls; temporary wrappers restored on normal or exceptional exit. Forward wrappers preserve nn.Module __call__ and its hooks.',
                'host_inclusive_ms': 'Host wall duration of the wrapped call, including nested calls and any production blocking/transfers; NOT CPU-only compute time.',
                'cuda_stream_span_inclusive_ms': 'CUDA events on current stream around host calls, after outer synchronization; includes launch gaps/CPU waiting and nested events, NOT kernel-only busy time or a multi-stream attribution.',
                'aggregation': 'All spans inclusive. parent_call_id identifies nesting; do not sum inclusive spans, subtract them to invent exclusive GPU time, or add CPU and GPU spans. Unwrapped Python and sparse-tensor construction remain unattributed.',
                'preprocessing': 'BuildBatchVoxels includes production CPU quantization, transfers and device feature aggregation. Data loading/preparation measured separately. Outer pipeline includes H2D before voxelization.',
                'bev': 'Single-frame shared projection recorded once, outside both heads; temporal detection and terrain projections separately recorded.',
                'nms': 'rotate_nms_pcdet includes sorting, limits, box conversion, native NMS and index mapping; nested inside CenterHead.predict decoding/filtering/class merge, itself nested in full export (OCC/terrain export and CPU copies). No kernel-only NMS claim.'}
