# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import time
from pathlib import Path
from typing import Optional

import torch
from lhotse import load_manifest_lazy

from lhotse.audio import set_audio_duration_mismatch_tolerance
from lhotse.cut import CutSet
from lhotse.utils import fastcopy

from .chime7_enhancers import FrontEnd_v1

logging.basicConfig(
    format="%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


ENHANCER_IMPL_CHOICES = ['gss', 'nemo_v1']


def enhance_cuts(
    cuts_per_recording: str,
    cuts_per_segment: str,
    enhanced_dir: str,
    bss_iterations: int,
    context_duration: float,
    use_garbage_class: bool,
    min_segment_length: float,
    max_segment_length: float,
    max_batch_duration: float,
    max_batch_cuts: int,
    num_buckets: int,
    num_workers: int,
    force_overwrite: bool,
    duration_tolerance: float,
    channels: Optional[str] = None,
    torchaudio_backend: str = 'soundfile',
    mc_mask_min_db: float = -60,
    mc_postmask_min_db: float = 0,
    dereverb_filter_length: int = 10,
    dereverb_prediction_delay: int = 2,
    dereverb_num_iterations: int = 3,
    mc_filter_type: str = 'pmwf',
    mc_filter_num_iterations: int = 5,
    mc_filter_postfilter: Optional[str] = 'ban',
):
    """
    Args:
        cuts_per_recording: Path to cuts manifest, e.g., cuts.jsonl.gz
        cuts_per_segment: Path to cuts_per_segment manifest, e.g., cuts_per_segment.JOB.jsonl.gz
        enhanced_dir: Output dir for enhanced audio
        bss_iterations: Number of BSS iterations. Default: 20
        context_duration: Context duration in seconds. Default: 15
        use_garbage_class: Set this flag to use garbage class
        min_segment_length: Minimum segment length in seconds
        max_segment_length: Maximum segment length in seconds
        max_batch_duration: Maximum batch duration in seconds
        max_batch_cuts: Maximum number of cuts per batch
        num_buckets: Number of buckets
        num_workers: Number of workers
        force_overwrite: Force overwrite
        duration_tolerance: Duration tolerance
        channels: Comma-separated list of channels
        torchaudio_backend: Backend used for torchaudio
        mc_mask_min_db: Minimum mask value in dB
        mc_postmask_min_db: Minimum postmask value in dB
        dereverb_filter_length: Dereverb filter length
        dereverb_prediction_delay: Prediction delay for dereverb
        dereverb_num_iterations: Number of iterations for dereverb
        mc_filter_type: Filter type
        mc_filter_num_iterations: Number of iterations for iterative filters
        mc_filter_postfilter: Postfilter type
    """
    logger.info('Enhance cuts')
    logger.info('\tcuts_per_recording: %s', cuts_per_recording)
    logger.info('\tcuts_per_segment:   %s', cuts_per_segment)
    logger.info('\tenhanced_dir:       %s', enhanced_dir)
    logger.info('\tbss_iterations:     %d', bss_iterations)
    logger.info('\tcontext_duration:   %f', context_duration)
    logger.info('\tuse_garbage_class:  %s', use_garbage_class)
    logger.info('\tmin_segment_length: %f', min_segment_length)
    logger.info('\tmax_segment_length: %f', max_segment_length)
    logger.info('\tmax_batch_duration: %f', max_batch_duration)
    logger.info('\tmax_batch_cuts:     %d', max_batch_cuts)
    logger.info('\tnum_buckets:        %d', num_buckets)
    logger.info('\tnum_workers:        %d', num_workers)
    logger.info('\tforce_overwrite:    %s', force_overwrite)
    logger.info('\tduration_tolerance: %f', duration_tolerance)
    logger.info('\tchannels:           %s', channels)
    logger.info('\ttorchaudio_backend: %s', torchaudio_backend)
    logger.info('\tmc_mask_min_db:     %f', mc_mask_min_db)
    logger.info('\tmc_postmask_min_db: %f', mc_postmask_min_db)
    logger.info('\tdereverb_filter_length: %d', dereverb_filter_length)
    logger.info('\tdereverb_prediction_delay: %d', dereverb_prediction_delay)
    logger.info('\tdereverb_num_iterations:   %d', dereverb_num_iterations)
    logger.info('\tmc_filter_type:            %s', mc_filter_type)
    logger.info('\tmc_filter_num_iterations:  %d', mc_filter_num_iterations)
    logger.info('\tmc_filter_postfilter:      %s', mc_filter_postfilter)

    # ########################################
    # Setup as in gss.bin.modes.enhance.cuts_
    # ########################################
    if duration_tolerance is not None:
        set_audio_duration_mismatch_tolerance(duration_tolerance)

    enhanced_dir = Path(enhanced_dir)
    enhanced_dir.mkdir(exist_ok=True, parents=True)

    cuts = load_manifest_lazy(cuts_per_recording)
    cuts_per_segment = load_manifest_lazy(cuts_per_segment)

    if channels is not None:
        channels = [int(c) for c in channels.split(",")]
        cuts_per_segment = CutSet.from_cuts(fastcopy(cut, channel=channels) for cut in cuts_per_segment)

    # Paranoia mode: ensure that cuts_per_recording have ids same as the recording_id
    cuts = CutSet.from_cuts(cut.with_id(cut.recording_id) for cut in cuts)

    logger.info("Aplying min/max segment length constraints")
    cuts_per_segment = cuts_per_segment.filter(lambda c: c.duration > min_segment_length).cut_into_windows(
        duration=max_segment_length
    )

    # ########################################
    # Initialize enhancer
    # ########################################
    enhancer = FrontEnd_v1(
        stft_fft_length=1024,
        stft_hop_length=256,
        dereverb_prediction_delay=dereverb_prediction_delay,
        dereverb_filter_length=dereverb_filter_length,
        dereverb_num_iterations=dereverb_num_iterations,
        bss_iterations=bss_iterations,
        mc_filter_type=mc_filter_type,
        mc_filter_beta=0,
        mc_filter_rank='one',
        mc_filter_num_iterations=mc_filter_num_iterations,
        mc_filter_postfilter=mc_filter_postfilter,
        mc_ref_channel='max_snr',
        mc_mask_min_db=mc_mask_min_db,
        mc_postmask_min_db=mc_postmask_min_db,
        use_dtype=torch.cfloat,
        cuts=cuts,
        context_duration=context_duration,
        activity_garbage_class=use_garbage_class,
    )

    # ########################################
    # Enhance cuts and save audio
    # ########################################
    logger.info(f'Enhancing {len(frozenset(c.id for c in cuts_per_segment))} segments')
    begin = time.time()
    num_errors, out_cuts = enhancer.enhance_cuts(
        cuts=cuts_per_segment,
        exp_dir=enhanced_dir,
        max_batch_duration=max_batch_duration,
        max_batch_cuts=max_batch_cuts,
        num_workers=num_workers,
        num_buckets=num_buckets,
        force_overwrite=force_overwrite,
        torchaudio_backend=torchaudio_backend,
    )
    end = time.time()

    if num_errors > 0:
        logger.error(f'Finished in {end-begin:.2f}s with {num_errors} errors')
    else:
        logger.info(f'Finished in {end-begin:.2f}s')


if __name__ == '__main__':
    """Simulate the interface of gss enhance cuts
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--cuts-per-recording', type=str, required=True, help='Path to cuts manifest, e.g., cuts.jsonl.gz',
    )
    parser.add_argument(
        '--cuts-per-segment',
        type=str,
        required=True,
        help='Path to cuts_per_segment manifest, e.g., cuts_per_segment.JOB.jsonl.gz',
    )
    parser.add_argument(
        '--enhanced-dir', type=str, required=True, help='Output dir for enhanced audio',
    )
    parser.add_argument(
        '--bss-iterations', type=int, default=20, help='Number of BSS iterations. Default: 20',
    )
    parser.add_argument(
        '--context-duration', type=float, default=15, help='Context duration in seconds. Default: 15',
    )
    parser.add_argument(
        '--use-garbage-class', action="store_true", help='Set this flag to use garbage class',
    )
    parser.add_argument(
        '--min-segment-length', type=float, default=0.0,
    )
    parser.add_argument(
        '--max-segment-length', type=float, default=200,
    )
    parser.add_argument(
        '--max-batch-duration', type=float, default=200,
    )
    parser.add_argument(
        '--max-batch-cuts', type=int, default=1,
    )
    parser.add_argument(
        '--num-buckets', type=int, default=4,
    )
    parser.add_argument(
        '--num-workers', type=int, default=4,
    )
    parser.add_argument(
        '--force-overwrite', action='store_true',
    )
    parser.add_argument(
        '--duration-tolerance', type=float, default=3.0,
    )
    parser.add_argument('--channels', type=str, default=None, help='Comma-separated list of channels')
    parser.add_argument(
        '--torchaudio-backend',
        type=str,
        choices=['sox_io', 'soundfile'],
        default='soundfile',  # faster than the defaulut sox_io
        help='Backend used for torchaudio',
    )
    parser.add_argument('--dereverb-filter-length', type=int, default=10, help='Dereverb filter length')
    parser.add_argument('--mc-mask-min-db', type=float, default=-60, help='Minimum mask value in dB')
    parser.add_argument('--mc-postmask-min-db', type=float, default=0, help='Minimum postmask value in dB')
    parser.add_argument('--dereverb-prediction-delay', type=int, default=2, help='Prediction delay for dereverb')
    parser.add_argument('--dereverb-num-iterations', type=int, default=3, help='Number of iterations for dereverb')
    parser.add_argument('--mc-filter-type', type=str, default='pmwf', help='Filter type')
    parser.add_argument(
        '--mc-filter-num-iterations', type=int, default=5, help='Number of iterations for iterative filters'
    )
    parser.add_argument(
        '--mc-filter-postfilter', type=lambda s: None if s == 'None' else str(s), default='ban', help='Postfilter type'
    )
    args = parser.parse_args()

    enhance_cuts(
        cuts_per_recording=args.cuts_per_recording,
        cuts_per_segment=args.cuts_per_segment,
        enhanced_dir=args.enhanced_dir,
        bss_iterations=args.bss_iterations,
        context_duration=args.context_duration,
        use_garbage_class=args.use_garbage_class,
        min_segment_length=args.min_segment_length,
        max_segment_length=args.max_segment_length,
        max_batch_duration=args.max_batch_duration,
        max_batch_cuts=args.max_batch_cuts,
        num_buckets=args.num_buckets,
        num_workers=args.num_workers,
        force_overwrite=args.force_overwrite,
        duration_tolerance=args.duration_tolerance,
        channels=args.channels,
        torchaudio_backend=args.torchaudio_backend,
        dereverb_filter_length=args.dereverb_filter_length,
        mc_mask_min_db=args.mc_mask_min_db,
        mc_postmask_min_db=args.mc_postmask_min_db,
        dereverb_prediction_delay=args.dereverb_prediction_delay,
        dereverb_num_iterations=args.dereverb_num_iterations,
        mc_filter_type=args.mc_filter_type,
        mc_filter_num_iterations=args.mc_filter_num_iterations,
        mc_filter_postfilter=args.mc_filter_postfilter,
    )
