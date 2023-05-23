import argparse
import os
import glob
import tqdm
import json
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest

diarization_system = 'system_vA04D'
# diarization_dir = os.path.expanduser(f'~/scratch/chime7/chime7_diar_results/{diarization_system}')

output_dir = f'./alignments/{diarization_system}'

def main(diarization_dir: str, subsets: list = ['dev']):
    # Assumption:
    # Output of diarization is organized in 3 subdirectories, with each subdirectory corresponding to one scenario (chime6, dipco, mixer6)
    scenario_dirs = glob.glob(diarization_dir + '/*')
    assert len(scenario_dirs) == 3, f'Expected 3 subdirectories, found {len(scenario_dirs)}'

    for scenario in ['chime6', 'dipco', 'mixer6']:
        for subset in subsets:
            # Currently, subdirectories don't have a uniform naming scheme
            # Therefore, we pick the subdirectory that has both scenario and subset in its name
            scenario_subset_dir = [sd for sd in scenario_dirs if scenario in sd and subset in sd][0]

            # Grab manifests from the results of diarization
            manifests_dir = os.path.join(scenario_subset_dir, 'pred_jsons_with_overlap')
            manifests = glob.glob(manifests_dir + '/*.json')
            
            # Process each manifest
            for manifest in manifests:
                manifest_name = os.path.basename(manifest)
                session_name = manifest_name.replace(scenario, '').replace('dev', '').replace('.json', '').strip('-')
                new_manifest = os.path.join(output_dir, scenario, subset, session_name + '.json')
                
                if not os.path.isdir(os.path.dirname(new_manifest)):
                    os.makedirs(os.path.dirname(new_manifest))

                # read manifest
                data = read_manifest(manifest)

                for item in data:
                    # not required
                    item.pop('audio_filepath')
                    item.pop('words')
                    item.pop('text')
                    item.pop('duration')
                    item.pop('offset')
                    # set these to be consistent with the baseline falign manifests
                    item['session_id'] = session_name
                    item['words'] = 'placeholder'

                # dump the list in a json file (not JSONL as our manifests)
                with open(new_manifest, 'w') as f:
                    json.dump(data, f, indent=4, sort_keys=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--diarization-dir',
        type=str,
        required=True,
        help='Directory with output of diarization',
    )
    args = parser.parse_args()

    main(diarization_dir=args.diarization_dir)
