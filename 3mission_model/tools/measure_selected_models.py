"""Serial measurement of the seven approved inference conditions; no training."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT/'tools')]
from run_detection_screen import CheckActive
from run_options import WriteJsonAtomic


def Main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--final-48', dest='b_final_48', action='store_true')
    args = parser.parse_args()
    selection = json.loads((PROJECT/'log/server_preparation/x4_detection_screen_20261008/inference_performance_sample_selection.json').read_text())
    indices = [str(row['index']) for row in selection['selected']]
    assert len(indices) == 2 and selection['distribution']['frame_count'] == 5973
    cases = [('single_f5_single', 'single_f', 'single', 'on'),
             ('f5_cold_on', 'f', 'cold', 'on'), ('f5_online_on', 'f', 'online', 'on'),
             ('f5_cold_off', 'f', 'cold', 'off'), ('f5_online_off', 'f', 'online', 'off'),
             ('f_no_root5_cold_off', 'f_no_root', 'cold', 'off'),
             ('f_no_root5_online_off', 'f_no_root', 'online', 'off')]
    if args.b_final_48:
        selection = json.loads((PROJECT/'log/server_preparation/x4_selected_long48_20261010/f48_benchmark_selection.json').read_text())
        indices = [str(row['index']) for row in selection['samples']]
        assert len(indices) == 2
        cases = [(f'f48_{mode}_{root_filter}', 'f', mode, root_filter)
                 for root_filter in ('on', 'off') for mode in ('cold', 'online')]
    args.output_root.mkdir(parents=True, exist_ok=False)
    WriteJsonAtomic(args.output_root/'sample_selection.json', selection)
    for name, candidate, mode, root_filter in cases:
        CheckActive()
        output = args.output_root/name
        command = [sys.executable, '-u', str(PROJECT/'tools/benchmark_three_task_inference.py'),
                   '--config', str(PROJECT/f'configs/server_x4_screen_{candidate}.yaml'),
                   '--checkpoint', str(PROJECT/f'log/server_runs_x4_detection_screen_20261008/{candidate}/checkpoints/epoch_5.pth'),
                   '--mode', mode, '--root', root_filter, '--indices', *indices,
                   '--warmup', '3', '--repeats', '10', '--workers', '2', '--rss-interval', '.02',
                   '--output', str(output)]
        if args.b_final_48:
            command[command.index('--config')+1] = str(PROJECT/'configs/server_x4_selected_f_long48.yaml')
            command[command.index('--checkpoint')+1] = str(PROJECT/'log/server_runs_x4_selected_long48_20261010/f/checkpoints/epoch_48.pth')
        WriteJsonAtomic(args.output_root/'current_case.json', dict(case=name, command=command))
        subprocess.run(command, check=True)
        report = json.loads((output/'report.json').read_text())
        assert report['b_complete'] and len(report['cases']) == 2
        assert all(len(case['samples']) == 10 for case in report['cases'])
        assert report['checkpoint_meta']['epoch'] == (48 if args.b_final_48 else 5)
    images = []
    if args.b_final_48:
        config = PROJECT/'log/server_preparation/x4_selected_long48_20261010/f48_root_off_visualization.yaml'
        for epoch, examples in [(48, [('train', 10), ('val', 0), ('val', 100), ('val', 300), ('val', 2720)])]:
            for split, index in examples:
                CheckActive()
                image = args.output_root/f'root_off_epoch{epoch}/{split}_{index:03d}.png'
                command = [sys.executable, '-u', str(PROJECT/'tools/visualize_three_task.py'),
                           '--temporal', '--config', str(config), '--checkpoint',
                           str(PROJECT/f'log/server_runs_x4_selected_long48_20261010/f/checkpoints/epoch_{epoch}.pth'),
                           '--split', split, '--index', str(index), '--output', str(image)]
                WriteJsonAtomic(args.output_root/'current_case.json', dict(image=str(image), command=command))
                subprocess.run(command, check=True)
                if image.stat().st_size < 1000:
                    raise ValueError('Missing real root-off prediction image')
                images.append(str(image))
    WriteJsonAtomic(args.output_root/'completed.json', dict(b_complete=True,
                    cases=[case[0] for case in cases], images=images, b_training_performed=False))


if __name__ == '__main__':
    Main()
