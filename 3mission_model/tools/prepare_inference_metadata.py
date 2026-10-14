"""Create fixed-val input-only metadata from point-only infos; no GPU or GT infos."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference_metadata import PrepareInferenceMetadata, VerifyInferenceMetadata


def Main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--point-only-infos', default='waymo_infos_validation_point_only.pkl',
                        help='Relative to WAYMO_DATA_ROOT; must be the point-only metadata source')
    parser.add_argument('--target-tokens', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    root = Path(os.environ['WAYMO_DATA_ROOT']).resolve()
    relative = Path(args.point_only_infos)
    if relative.is_absolute() or '..' in relative.parts:
        parser.error('--point-only-infos must be relative to WAYMO_DATA_ROOT')
    manifest = PrepareInferenceMetadata(root / relative, args.target_tokens, args.output)
    binding = VerifyInferenceMetadata(args.output / 'manifest.json')
    print(json.dumps({'b_verified': True, 'target_count': manifest['target_count'],
                      'history_count': manifest['history_count'], 'binding': binding}, indent=2))


if __name__ == '__main__':
    Main()
