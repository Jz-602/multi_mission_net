"""CPU gate for complete-validation detection screening; never declares final success.

Records use the production validation_metrics.json schema: epoch and metrics
containing det/terrain/occ. AP/APH values are fractions; thresholds below are
absolute percentage points. This gate intentionally requires completed-label GT.
"""
import argparse
import json
import math
from pathlib import Path
import sys


CLASSES = ('vehicle', 'pedestrian', 'cyclist')
PROTOCOL = 'completed_detection_l2_v1'
CLASS_COLLAPSE_RATIO = 0.25
CLASS_COLLAPSE_DROP_PP = 10.0
DROP_TOLERANCE_PP = 1e-9
DETECTION_KEYS = (
    *(f'{name}_{metric}_{level}' for name in CLASSES
      for metric in ('ap', 'aph') for level in ('l1', 'l2')),
    'map_l1', 'maph_l1', 'map_l2', 'maph_l2',
)


def IsFiniteNumber(value):
    return type(value) in (int, float) and math.isfinite(value)


def CheckRecord(record, name, expected_targets):
    errors = []
    if not isinstance(record, dict):
        return [f'{name}: record must be a mapping']
    if type(record.get('epoch')) is not int or record['epoch'] < 0:
        errors.append(f'{name}: missing/invalid nonnegative integer epoch')
    metrics = record.get('metrics')
    if not isinstance(metrics, dict):
        return errors + [f'{name}: missing metrics mapping']

    def CheckFiniteTree(value, path):
        if isinstance(value, dict):
            for key, item in value.items():
                CheckFiniteTree(item, f'{path}.{key}')
        elif isinstance(value, list):
            for index, item in enumerate(value):
                CheckFiniteTree(item, f'{path}[{index}]')
        elif type(value) is float and not math.isfinite(value):
            errors.append(f'{path}: nonfinite value')

    CheckFiniteTree(metrics, f'{name}.metrics')
    det = metrics.get('det')
    if not isinstance(det, dict):
        return errors + [f'{name}: missing det mapping']
    for key in DETECTION_KEYS:
        value = det.get(key)
        # Installed TF metrics can exceed 1 by a few float32 ulps on perfect GT.
        if not IsFiniteNumber(value) or not -1e-6 <= value <= 1.0 + 1e-6:
            errors.append(f'{name}.det.{key}: missing/invalid finite AP fraction')
    if not errors:
        for level in ('l1', 'l2'):
            for metric, mean_key in (('ap', 'map'), ('aph', 'maph')):
                mean_value = sum(det[f'{key}_{metric}_{level}'] for key in CLASSES) / 3
                if abs(mean_value - det[f'{mean_key}_{level}']) > 1e-6:
                    errors.append(f'{name}.det.{mean_key}_{level}: inconsistent class mean')
    audit = det.get('ground_truth_audit')
    if not isinstance(audit, dict):
        errors.append(f'{name}: missing completed detection ground_truth_audit')
        audit = {}
    if audit.get('protocol') != PROTOCOL:
        errors.append(f'{name}: missing/wrong completed-label protocol')
    for task, entry in (('det', audit), ('terrain', metrics.get('terrain')),
                        ('occ', metrics.get('occ'))):
        if (not isinstance(entry, dict) or type(entry.get('frame_count')) is not int
                or entry['frame_count'] != expected_targets):
            errors.append(f'{name}.{task}: expected {expected_targets} full-validation targets')
    for task, keys in (('terrain', ('mae_m', 'rmse_m')),
                       ('occ', ('occupancy_iou', 'semantic_miou'))):
        entry = metrics.get(task)
        for key in keys:
            if not isinstance(entry, dict) or not IsFiniteNumber(entry.get(key)):
                errors.append(f'{name}.{task}.{key}: missing/nonfinite task metric')
    eligible = audit.get('eligible_by_class')
    if not isinstance(eligible, dict) or any(
            type(eligible.get(key)) is not int or eligible[key] < 0 for key in CLASSES):
        errors.append(f'{name}: missing/invalid GT eligible class counts')
    for key in ('original_eligible_count', 'supplemental_eligible_count',
                'supplemental_zero_point_count'):
        if type(audit.get(key)) is not int or audit[key] < 0:
            errors.append(f'{name}: missing/invalid GT {key}')
    return errors


def EvaluateGate(baseline_record, current_record, max_detection_drop_pp=5.0,
                 expected_targets=5973):
    """Return pass/early_stop/review_required without changing any process state."""
    result = {
        'schema_version': 1, 'status': 'review_required', 'action': 'hold_for_review',
        'b_can_continue': False, 'b_final_experiment_complete': False, 'reasons': [],
        'evidence_scope': ('Validation summaries only; runner must separately bind checkpoint, '
                           'config and target tokens, enforce max epochs and manual pause/cancel.'),
        'rules': {'map_metric': 'map_l2', 'drop_units': 'percentage_points',
                  'class_metric': 'ap_l2', 'class_ratio_strictly_below': CLASS_COLLAPSE_RATIO,
                  'class_drop_at_least_pp': CLASS_COLLAPSE_DROP_PP,
                  'drop_tolerance_pp': DROP_TOLERANCE_PP},
    }
    if (not IsFiniteNumber(max_detection_drop_pp) or not 0 < max_detection_drop_pp <= 100
            or type(expected_targets) is not int or expected_targets <= 0):
        result['reasons'] = ['missing/invalid finite positive drop threshold or target count']
        return result
    result['rules'].update(max_detection_drop_pp=max_detection_drop_pp,
                           expected_targets=expected_targets)
    result['reasons'] = (CheckRecord(baseline_record, 'baseline', expected_targets)
                         + CheckRecord(current_record, 'current', expected_targets))
    if result['reasons']:
        return result
    baseline = baseline_record['metrics']['det']
    current = current_record['metrics']['det']
    for key in ('protocol', 'frame_count', 'eligible_by_class', 'original_eligible_count',
                'supplemental_eligible_count', 'supplemental_zero_point_count'):
        if baseline['ground_truth_audit'][key] != current['ground_truth_audit'][key]:
            result['reasons'].append(f'baseline/current GT mismatch: {key}')
    if result['reasons']:
        return result
    drop_pp = (baseline['map_l2'] - current['map_l2']) * 100.0
    result.update(baseline_epoch=baseline_record['epoch'], current_epoch=current_record['epoch'],
                  baseline_map_l2=baseline['map_l2'], current_map_l2=current['map_l2'],
                  map_drop_pp=drop_pp, class_comparison={})
    if drop_pp + DROP_TOLERANCE_PP >= max_detection_drop_pp:
        result['reasons'].append(f'mAP_L2 drop {drop_pp:.12g} pp >= {max_detection_drop_pp:g} pp')
    for name in CLASSES:
        before, after = baseline[f'{name}_ap_l2'], current[f'{name}_ap_l2']
        class_drop_pp = (before - after) * 100.0
        b_collapse = (after < before * CLASS_COLLAPSE_RATIO
                      and class_drop_pp + DROP_TOLERANCE_PP >= CLASS_COLLAPSE_DROP_PP)
        result['class_comparison'][name] = {
            'baseline_ap_l2': before, 'current_ap_l2': after,
            'drop_pp': class_drop_pp, 'b_collapse': b_collapse,
        }
        if b_collapse:
            result['reasons'].append(
                f'{name} AP_L2 collapse: below baseline*{CLASS_COLLAPSE_RATIO:g} '
                f'and drop {class_drop_pp:.12g} pp >= {CLASS_COLLAPSE_DROP_PP:g} pp')
    if result['reasons']:
        result.update(status='early_stop', action='stop_training')
    else:
        result.update(status='pass', action='pass_to_next_epoch', b_can_continue=True)
    return result


def Main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True, type=Path)
    parser.add_argument('--current', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path, help='Acceptance JSON output path')
    parser.add_argument('--max-detection-drop-pp', type=float, default=5.0)
    parser.add_argument('--expected-targets', type=int, default=5973)
    args = parser.parse_args()
    try:
        result = EvaluateGate(json.loads(args.baseline.read_text()),
                              json.loads(args.current.read_text()),
                              args.max_detection_drop_pp, args.expected_targets)
    except (OSError, ValueError) as error:
        result = {'schema_version': 1, 'status': 'review_required', 'action': 'hold_for_review',
                  'b_can_continue': False, 'b_final_experiment_complete': False,
                  'reasons': [f'Unable to read validation records: {error}']}
    result['inputs'] = {'baseline': str(args.baseline.resolve()), 'current': str(args.current.resolve())}
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(args.output.name + '.tmp')
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        temporary.replace(args.output)
    except OSError as error:
        print(f'Cannot save gate evidence: {error}', file=sys.stderr)
        return 4
    print(json.dumps({'status': result['status'], 'action': result['action'],
                      'output': str(args.output.resolve())}))
    return {'pass': 0, 'early_stop': 3, 'review_required': 4}[result['status']]


if __name__ == '__main__':
    sys.exit(Main())
