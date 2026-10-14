"""CPU acceptance of one complete, fixed-set OCC statistics collection."""
import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from occ_stage_statistics import MergeStatistics
from run_options import HashFile, WriteJsonAtomic


def Read(path):
    return json.loads(path.read_text())


def Audit(directory, reference_run=None, paired_directory=None):
    summary = Read(directory/'summary.json')
    manifest = Read(directory/'manifest.json')
    tokens = Read(directory/'val_tokens.json')
    expected = Read(PROJECT/'log/server_runs_x4_detection_screen_20261008/f/val_tokens.json')
    assert tokens == expected and len(set(tokens)) == 5973
    assert summary['b_complete'] and summary['frame_count'] == 5973
    assert HashFile(manifest['checkpoint']) == manifest['checkpoint_sha256']
    fresh = MergeStatistics([directory/f'frames_rank{i}.jsonl' for i in range(8)], tokens)
    assert fresh['counts'] == summary['counts'] and fresh['ratios'] == summary['ratios']
    assert fresh['duplicate_records_removed'] == 0
    records = {}
    durations = []
    for rank in range(8):
        done = Read(directory/f'rank{rank}_complete.json')
        rows = [json.loads(line) for line in (directory/f'frames_rank{rank}.jsonl').read_text().splitlines()]
        assert [row['target_id'] for row in rows] == tokens[rank::8]
        assert done['frame_count'] == done['expected_count'] == len(rows)
        durations.append(done['elapsed_seconds_including_cpu_statistics'])
        for row in rows:
            c = row['counts']
            assert sum(c['gt_fates']) == c['gt']
            assert c['leaf_raw'] == 64*c['expanded_root'] == 8*c['y3_raw']
            assert c['leaf_raw'] - c['leaf'] == c['leaf_ignored']
            assert c['gt_fates'][3] == c['candidate_gt_nonempty_correct'] + c['candidate_gt_nonempty_wrong_nonempty']
            assert c['occupancy_tp'] + c['occupancy_fn'] == c['gt']
            assert c['occupancy_tp'] + c['occupancy_fp'] == c['pred_positive']
            if not manifest['b_apply_root_filter']:
                assert c['deleted_root'] == c['root_deletion_lost_gt'] == 0
            records[row['target_id']] = c
    result = dict(b_pass=True, frame_count=5973, b_per_frame_and_global_counts_verified=True,
                  checkpoint_sha256=manifest['checkpoint_sha256'],
                  b_apply_root_filter=manifest['b_apply_root_filter'],
                  gt_fates_per_100=summary['gt_fates_per_100'],
                  max_rank_statistics_seconds=max(durations), b_inference_timing=False)
    if reference_run:
        epoch = int(manifest['checkpoint_meta']['epoch'])
        previous = Read(reference_run/f'validation/epoch_{epoch}.json')['metrics']['occ']
        deltas = dict(occupancy_iou=summary['ratios']['occupancy_iou']['value']-previous['occupancy_iou'],
                      semantic_miou=summary['ratios']['semantic_miou']-previous['semantic_miou'])
        assert all(abs(value) < 1e-6 for value in deltas.values())
        result.update(reference_metric_deltas=deltas,
                      reference_count_deltas={key:summary['counts'][key]-previous[key]
                          for key in ('occupancy_tp', 'occupancy_fp', 'occupancy_fn')},
                      b_bitwise_prediction_identity=False)
    if paired_directory:
        paired_manifest = Read(paired_directory/'manifest.json')
        assert paired_manifest['checkpoint_sha256'] == manifest['checkpoint_sha256']
        fields = ('gt', 'ignored_leaf', 'input_leaf', 'input_gt', 'history_leaf', 'history_gt',
                  'history_current_leaf', 'history_current_gt', 'root', 'occupied_root',
                  'l0_raw', 'l0', 'l0_gt', 'lplus_raw', 'lplus', 'lplus_gt')
        for path in paired_directory.glob('frames_rank*.jsonl'):
            for line in path.read_text().splitlines():
                old = json.loads(line)
                assert all(old['counts'][key] == records[old['target_id']][key] for key in fields)
        result['same_weight_pre_filter_geometry_and_real_input_counts_equal'] = True
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--reference-run', type=Path)
    parser.add_argument('--paired-directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = Audit(args.directory, args.reference_run, args.paired_directory)
    WriteJsonAtomic(args.output, report)
    print(json.dumps(report))
