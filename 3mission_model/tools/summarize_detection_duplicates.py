"""固定样本后NMS重叠对数；只是重复框诊断量，不把相邻真车一概当误检。"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from audit_detection_nms import BuildPolygons


def Summarize(directory):
    records = []
    for filename in sorted(directory.glob('*.pt')):
        output = torch.load(filename, map_location='cpu', weights_only=False)
        for token, value in output.items():
            det = value['det']
            boxes = det['box3d_lidar'].numpy()
            labels, scores = det['label_preds'].numpy(), det['scores'].numpy()
            polygons = BuildPolygons(boxes)
            pairs = []
            for left in range(len(boxes)):
                for right in range(left+1, len(boxes)):
                    if labels[left] != labels[right]:
                        continue
                    intersection = polygons[left].intersection(polygons[right]).area
                    union = polygons[left].area + polygons[right].area - intersection
                    iou = intersection / union if union else 0.0
                    if iou >= .5:
                        pairs.append(dict(indices=[left, right], label=int(labels[left]), iou=iou,
                                          scores=[float(scores[left]), float(scores[right])],
                                          center_distance_m=float(np.linalg.norm(boxes[left,:2]-boxes[right,:2]))))
            records.append(dict(sample=filename.stem, token=token, box_count=len(boxes),
                                visible_box_count=int((scores >= .15).sum()), overlapping_pairs=pairs))
    (directory/'duplicate_diagnostics.json').write_text(json.dumps(dict(
        note='同类别旋转BEV IoU>=.5的后NMS框对；保留分数，不等于已证明误检。正式车辆阈值.7。',
        records=records), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    Summarize(parser.parse_args().directory)
