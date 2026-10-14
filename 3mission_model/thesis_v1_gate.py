"""Frozen thesis-v1 CPU gates. No process control, training, I/O or threshold adaptation.

current/reference: production {epoch, metrics} validation records. reference is
accepted S0 for continuations; S0 uses the constants below, never a moving model.
previous: None at epoch 0, otherwise {'record': previous_record, 'gate': previous_gate}.
The runner separately binds checkpoint/config/manifest identity, images and pause
state. Healthy means the next planned epoch is allowed, not temporal superiority.
"""
import math

from tools.detection_screen_gate import CheckRecord, IsFiniteNumber, CLASSES

EXPECTED_TARGETS = 5973
EXPECTED_TOKEN_SHA256 = '4a9ba13f3483643ebd12ccd27ef8299e92ec7ff1eb57247790c3cc5f686dbc58'
GT_VALID_CELLS = 294874915
GT_HOLE_CELLS = 49169885
ALL_VALID_MIOU = 0.42854145
OLD_MAP = 0.6108642419
OLD_OCC_IOU = 0.355869959
OLD_DETECTION = {
    'maph_l2': 0.5809723337491354,
    'vehicle_ap_l2': 0.8045182824134827, 'vehicle_aph_l2': 0.7986391186714172,
    'pedestrian_ap_l2': 0.6315881609916687, 'pedestrian_aph_l2': 0.5686075091362,
    'cyclist_ap_l2': 0.3964862823486328, 'cyclist_aph_l2': 0.3756703734397888,
}
STAGES = ('S0', 'S', 'T', 'S-R', 'T-R')
COLLAPSE_KEYS = tuple(OLD_DETECTION)
TOLERANCE = 1e-12


def NonfinitePaths(value, path):
    if isinstance(value, dict):
        return [error for key, item in value.items() for error in NonfinitePaths(item, f'{path}.{key}')]
    if isinstance(value, (list, tuple)):
        return [error for index, item in enumerate(value) for error in NonfinitePaths(item, f'{path}[{index}]')]
    return [f'{path}: nonfinite'] if type(value) is float and not math.isfinite(value) else []


def CheckThesisRecord(record, name):
    errors = CheckRecord(record, name, EXPECTED_TARGETS) + NonfinitePaths(record, name)
    if not isinstance(record, dict) or not isinstance(record.get('metrics'), dict):
        return errors
    if type(record.get('epoch')) is int and record['epoch'] > 48:
        errors.append(f'{name}: epoch exceeds frozen 48-epoch schedule')
    metrics = record['metrics']
    audit = metrics.get('validation_audit', {})
    if not isinstance(audit, dict):
        audit = {}
    for key in ('unique_target_count', 'expected_target_count'):
        if type(audit.get(key)) is not int or audit[key] != EXPECTED_TARGETS:
            errors.append(f'{name}.validation_audit.{key}: must be {EXPECTED_TARGETS}')
    for key in ('token_sha256', 'expected_token_sha256'):
        if audit.get(key) != EXPECTED_TOKEN_SHA256:
            errors.append(f'{name}.validation_audit.{key}: frozen ordered target hash mismatch')
    if audit.get('b_complete') is not True:
        errors.append(f'{name}: actual token set not certified complete/deduplicated')
    for key in ('missing_count', 'unknown_count'):
        if key in audit and (type(audit[key]) is not int or audit[key] != 0):
            errors.append(f'{name}.validation_audit.{key}: must be zero')
    terrain, occ = metrics.get('terrain', {}), metrics.get('occ', {})
    if not isinstance(terrain, dict) or not isinstance(occ, dict):
        return errors + [f'{name}: terrain/occ must be mappings']
    validity = terrain.get('validity', {})
    if not isinstance(validity, dict):
        validity = {}
    for key in ('valid_recall', 'hole_iou', 'miou', 'joint_10cm_recall', 'joint_20cm_recall'):
        if not IsFiniteNumber(validity.get(key)) or not 0 <= validity[key] <= 1:
            errors.append(f'{name}.terrain.validity.{key}: missing/invalid finite fraction')
    for key in ('mae_m', 'rmse_m'):
        if IsFiniteNumber(terrain.get(key)) and terrain[key] < 0:
            errors.append(f'{name}.terrain.{key}: negative error')
    if validity.get('threshold') != 0.5:
        errors.append(f'{name}: terrain validity threshold must remain 0.5')
    count_keys = ('validity_tp', 'validity_fn', 'validity_fp', 'validity_tn',
                  'classification_count', 'gt_valid_count', 'ignore_count',
                  'joint_10cm_count', 'joint_20cm_count')
    b_valid_counts = True
    for key in count_keys:
        value = validity.get(key)
        if not IsFiniteNumber(value) or value < 0 or int(value) != value:
            errors.append(f'{name}.terrain.validity.{key}: missing/noninteger count')
            b_valid_counts = False
    if b_valid_counts:
        tp, fn, fp, tn = (validity['validity_' + key] for key in ('tp', 'fn', 'fp', 'tn'))
        if (tp + fn != GT_VALID_CELLS or tn + fp != GT_HOLE_CELLS
                or validity['classification_count'] != GT_VALID_CELLS + GT_HOLE_CELLS
                or validity['gt_valid_count'] != GT_VALID_CELLS or validity['ignore_count'] != 0
                or terrain.get('gt_valid_count') != GT_VALID_CELLS):
            errors.append(f'{name}: terrain GT/confusion counts differ from full frozen validation')
        if not 0 <= validity['joint_10cm_count'] <= validity['joint_20cm_count'] <= tp:
            errors.append(f'{name}: joint counts must satisfy 0 <= joint10 <= joint20 <= TP')
        expected = {'valid_recall': tp / GT_VALID_CELLS,
                    'hole_iou': tn / max(tn + fn + fp, 1),
                    'miou': (tp / max(tp + fn + fp, 1) + tn / max(tn + fn + fp, 1)) / 2,
                    'joint_10cm_recall': validity['joint_10cm_count'] / GT_VALID_CELLS,
                    'joint_20cm_recall': validity['joint_20cm_count'] / GT_VALID_CELLS}
        for key, value in expected.items():
            if IsFiniteNumber(validity.get(key)) and abs(validity[key] - value) > 1e-6:
                errors.append(f'{name}.terrain.validity.{key}: inconsistent global counts/denominator')
    for key in ('occupancy_iou', 'semantic_miou', 'occupied_root_recall'):
        if not IsFiniteNumber(occ.get(key)) or not 0 <= occ[key] <= 1:
            errors.append(f'{name}.occ.{key}: missing/invalid finite fraction')
    for key in ('occupied_root_count', 'occupied_root_kept', 'root_count', 'kept_root_count'):
        value = occ.get(key)
        if not IsFiniteNumber(value) or value < 0 or int(value) != value:
            errors.append(f'{name}.occ.{key}: missing/noninteger count')
    if all(IsFiniteNumber(occ.get(key)) for key in ('occupied_root_count', 'occupied_root_kept', 'occupied_root_recall')):
        count, kept = occ['occupied_root_count'], occ['occupied_root_kept']
        if count <= 0 or kept > count or abs(occ['occupied_root_recall'] - kept / max(count, 1)) > 1e-6:
            errors.append(f'{name}: occupied-root recall/counts inconsistent')
    root_keys = ('root_count', 'kept_root_count', 'occupied_root_count', 'occupied_root_kept')
    if all(IsFiniteNumber(occ.get(key)) for key in root_keys):
        if (occ['kept_root_count'] > occ['root_count'] or occ['occupied_root_count'] > occ['root_count']
                or occ['occupied_root_kept'] > occ['kept_root_count']):
            errors.append(f'{name}: root subset counts inconsistent')
    return errors


def CollapseReasons(det, reference_det):
    return [f'{key} below 20% of fixed reference' for key in COLLAPSE_KEYS
            if det[key] < 0.2 * reference_det[key] - TOLERANCE]


def Eligibility(record, reference, stage):
    metrics = record['metrics']
    det, terrain, occ = (metrics[key] for key in ('det', 'terrain', 'occ'))
    validity = terrain['validity']
    reasons = []
    if stage == 'S0':
        limits = [('mAP', det['map_l2'], OLD_MAP - .05),
                  ('OCC IoU', occ['occupancy_iou'], .28),
                  ('valid recall', validity['valid_recall'], .95)]
        if terrain['mae_m'] > .12 + TOLERANCE:
            reasons.append('weighted height MAE > 0.12m')
        if terrain['rmse_m'] > .32 + TOLERANCE:
            reasons.append('weighted height RMSE > 0.32m')
        reasons += CollapseReasons(det, OLD_DETECTION)
        score = (det['map_l2'] / OLD_MAP + occ['occupancy_iou'] / OLD_OCC_IOU + validity['joint_20cm_recall']) / 3
    else:
        baseline = reference['metrics']
        limits = [('mAP', det['map_l2'], baseline['det']['map_l2'] - .02),
                  ('OCC IoU', occ['occupancy_iou'], baseline['occ']['occupancy_iou'] - .02),
                  ('valid recall', validity['valid_recall'], max(.90, baseline['terrain']['validity']['valid_recall'] - .03))]
        if terrain['mae_m'] > 1.10 * baseline['terrain']['mae_m'] + TOLERANCE:
            reasons.append('weighted height MAE > 1.10 * S0')
        reasons += CollapseReasons(det, baseline['det'])
        score = (det['map_l2'] / baseline['det']['map_l2']
                 + occ['occupancy_iou'] / baseline['occ']['occupancy_iou']
                 + validity['joint_20cm_recall'] / baseline['terrain']['validity']['joint_20cm_recall']) / 3
    for name, actual, limit in limits:
        if actual < limit - TOLERANCE:
            reasons.append(f'{name} below eligibility limit {limit:.12g}')
    if validity['hole_iou'] < .20 - TOLERANCE:
        reasons.append('hole IoU below 0.20')
    if validity['miou'] <= ALL_VALID_MIOU:
        reasons.append('binary mIoU does not exceed frozen all-valid baseline')
    return reasons, score


def EvaluateThesisGate(current, reference, previous, stage):
    """Return health and numerical selection eligibility; never claim final success."""
    result = {'schema_version': 1, 'protocol': 'thesis_v1_20261011', 'stage': stage,
              'status': 'diagnostic_required', 'b_healthy': False, 'b_can_continue': False,
              'b_eligible': False, 'score': None, 'reasons': [], 'eligibility_reasons': [],
              'b_root_healthy_seen': False, 'last_healthy_root_recall': None,
              'b_final_experiment_complete': False,
              'scope': 'Numerical validation only; checkpoint/manifest/code identity, visual acceptance, manual pause and full schedule completion are separate runner obligations.'}
    if stage not in STAGES:
        result['reasons'] = ['unknown thesis stage']
        return result
    errors = CheckThesisRecord(current, 'current')
    if stage != 'S0':
        errors += CheckThesisRecord(reference, 'S0 reference')
        if not errors:
            reference_reasons, _ = Eligibility(reference, None, 'S0')
            if reference_reasons:
                errors += ['S0 reference is not numerically eligible: ' + reason for reason in reference_reasons]
            if reference['metrics']['terrain']['validity']['joint_20cm_recall'] <= 0:
                errors.append('S0 joint20 must be positive')
            if reference['metrics']['occ']['occupied_root_recall'] < .60 - TOLERANCE:
                errors.append('S0 reference has not reached healthy root recall')
            for key in ('protocol', 'frame_count', 'eligible_by_class', 'original_eligible_count',
                        'supplemental_eligible_count', 'supplemental_zero_point_count'):
                if current['metrics']['det']['ground_truth_audit'][key] != reference['metrics']['det']['ground_truth_audit'][key]:
                    errors.append(f'current/reference detection GT mismatch: {key}')
    if previous is not None:
        if not isinstance(previous, dict) or not isinstance(previous.get('gate'), dict):
            errors.append('previous requires record and gate mappings')
        else:
            errors += CheckThesisRecord(previous.get('record'), 'previous')
            errors += NonfinitePaths(previous['gate'], 'previous.gate')
            gate = previous['gate']
            if gate.get('stage') != stage or type(gate.get('b_root_healthy_seen')) is not bool:
                errors.append('previous gate stage/root health binding missing')
            if isinstance(previous.get('record'), dict) and gate.get('epoch') != previous['record'].get('epoch'):
                errors.append('previous gate epoch differs from previous record')
            if gate.get('b_healthy') is not True or gate.get('status') != 'pass':
                errors.append('previous validation gate requires diagnostic resolution before progression')
            if not errors and previous['record']['epoch'] != current['epoch'] - 1:
                errors.append('previous validation must be exactly preceding epoch')
            if gate.get('b_root_healthy_seen') and not IsFiniteNumber(gate.get('last_healthy_root_recall')):
                errors.append('previous cumulative root health missing finite recall')
            if (gate.get('stage') == stage and type(gate.get('b_root_healthy_seen')) is bool
                    and (not gate['b_root_healthy_seen'] or IsFiniteNumber(gate.get('last_healthy_root_recall')))):
                result['b_root_healthy_seen'] = gate['b_root_healthy_seen']
                result['last_healthy_root_recall'] = gate['last_healthy_root_recall']
    elif isinstance(current, dict) and current.get('epoch') != 0:
        errors.append('nonzero epoch requires preceding validation and gate')
    if errors:
        result['reasons'] = errors
        return result
    metrics = current['metrics']
    det, occ = metrics['det'], metrics['occ']
    result['epoch'] = current['epoch']
    if stage != 'S0':
        baseline = reference['metrics']['det']
        if baseline['map_l2'] - det['map_l2'] >= .05 - TOLERANCE:
            result['reasons'].append('mAP L2 drop >= 5 percentage points versus S0')
        result['reasons'] += CollapseReasons(det, baseline)
        if previous is None:
            result['b_root_healthy_seen'] = True
            result['last_healthy_root_recall'] = reference['metrics']['occ']['occupied_root_recall']
    if stage.endswith('-R'):
        if occ['kept_root_count'] != occ['root_count'] or occ['occupied_root_recall'] < 1 - TOLERANCE:
            result['reasons'].append('root-off stage did not retain all roots')
    elif result['b_root_healthy_seen']:
        recall = occ['occupied_root_recall']
        if recall < .60 - TOLERANCE:
            result['reasons'].append('occupied-root recall below 0.60 after healthy activation')
        if result['last_healthy_root_recall'] - recall > .25 + TOLERANCE:
            result['reasons'].append('occupied-root recall drop > 25 percentage points versus last healthy validation')
    eligibility_reasons, score = Eligibility(current, reference, stage)
    if stage == 'S0' and not eligibility_reasons and occ['occupied_root_recall'] < .60 - TOLERANCE:
        result['reasons'].append('first numerically eligible S0 has occupied-root recall below 0.60')
    result.update(eligibility_reasons=eligibility_reasons, score=score)
    if not result['reasons']:
        result.update(status='pass', b_healthy=True, b_can_continue=current['epoch'] < 48,
                      b_eligible=not eligibility_reasons)
        if (stage != 'S0' or not eligibility_reasons) and occ['occupied_root_recall'] >= .60:
            result['b_root_healthy_seen'] = True
        if result['b_root_healthy_seen']:
            result['last_healthy_root_recall'] = occ['occupied_root_recall']
    return result


def SelectThesisCheckpoint(records, reference, stage):
    """Evaluate every epoch; choose one eligible complete record, exact tie -> early epoch."""
    if not records or any(not isinstance(record, dict) or type(record.get('epoch')) is not int for record in records):
        return {'status': 'diagnostic_required', 'selected_epoch': None, 'reasons': ['missing/invalid records']}
    ordered = sorted(records, key=lambda record: record['epoch'])
    if [record['epoch'] for record in ordered] != list(range(len(ordered))):
        return {'status': 'diagnostic_required', 'selected_epoch': None, 'reasons': ['validation epochs must be unique consecutive 0..N']}
    previous, evaluations, eligible = None, [], []
    for record in ordered:
        gate = EvaluateThesisGate(record, reference, previous, stage)
        evaluations.append(gate)
        if gate['b_eligible']:
            eligible.append(gate)
        previous = {'record': record, 'gate': gate}
    selected = max(eligible, key=lambda gate: (gate['score'], -gate['epoch'])) if eligible else None
    b_any_diagnostic = any(not gate['b_healthy'] for gate in evaluations)
    return {'status': 'diagnostic_required' if b_any_diagnostic else 'selected_numerical_candidate' if selected else 'no_eligible_checkpoint',
            'selected_epoch': selected['epoch'] if selected else None,
            'selected_score': selected['score'] if selected else None,
            'last_epoch': ordered[-1]['epoch'], 'b_complete_48_epochs': ordered[-1]['epoch'] == 48,
            'evaluations': evaluations, 'b_any_diagnostic': b_any_diagnostic, 'b_final_experiment_complete': False,
            'selection_rule': 'maximum same-checkpoint three-task score among healthy eligible epochs; exact score tie selects earlier epoch; final epoch reported separately'}
