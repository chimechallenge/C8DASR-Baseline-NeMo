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

import logging
import math
import pathlib
from abc import ABCMeta, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import numpy as np
import soundfile as sf
import torch
import torchaudio
from gss.core import Activity
from gss.utils.data_utils import GssDataset, create_sampler
from lhotse import CutSet, Recording, RecordingSet, SupervisionSegment, SupervisionSet
from lhotse.utils import add_durations, compute_num_samples
from torch.utils.data import DataLoader
import decimal

from nemo.collections.asr.modules.audio_modules import MaskBasedBeamformer, MaskBasedDereverbWPE, MaskEstimatorGSS
from nemo.collections.asr.modules.audio_preprocessing import AudioToSpectrogram, SpectrogramToAudio

logging.basicConfig(
    format="%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def output_file_name(recording_id: str, speaker: int, start: float, end: float) -> str:
    """Output file name for a cut

    Example:
        output_filename(recording_id, speaker, cut.start, cut.end)
    """
    return f'{recording_id}-{speaker}-{round(100*start):06d}_{round(100*end):06d}.flac'


def samples_to_frames(samples: int, fft_length: int, hop_length: int) -> int:
    """Convert samples to number of frames
    """
    frames = (samples - fft_length + hop_length) / hop_length
    return int(frames)


def activity_time_to_timefreq(a_time, win_length, hop_length):
    """Convert time-domain activity to TF domain.

    Args:
        a_time: (B, channel, time)
        analysis_transform: for example, stft

    Return:
        (B, channel, time frame)
    """
    assert a_time.ndim == 3
    a_tf = torch.nn.functional.pad(a_time.unsqueeze(-1), pad=(0, 0, win_length // 2, win_length // 2))
    a_tf = torch.nn.functional.unfold(a_tf, kernel_size=(win_length, 1), stride=(hop_length, 1))
    a_tf = a_tf.reshape(a_time.size(0), a_time.size(1), win_length, -1)
    a_tf = torch.abs(a_tf).any(axis=-2)
    return a_tf


def activity_time_to_timefreq_stft(a, stft):
    """This was the original prototype activity_time_to_timefreq.
    Keeping it here for reference since it worked a little better than
    the other.
    """
    A, _ = stft(input=a)
    A = torch.abs(A)
    A = torch.mean(A, axis=-2) > 0.5
    return A


def get_int_divisors(n):
    divs = [1]
    for i in range(2,int(math.sqrt(n))+1):
        if n%i == 0:
            divs.extend([i,n/i])
    divs.extend([int(n)])
    return sorted([int(x) for x in list(set(divs))])

def save_worker(exp_dir, orig_cuts, x_hat, recording_id, speaker, sample_rate, force_overwrite):
    """Save all cuts from orig_cuts into their respective files.
    This is mostly based on GSS's _save_worker function.

    Args:
        x_hat: single-channel output (time,)
    """
    out_dir = pathlib.Path(exp_dir) / recording_id
    enhanced_recordings = []
    enhanced_supervisions = []
    offset = 0
    for cut in orig_cuts:
        save_path = pathlib.Path(
            output_file_name(recording_id=recording_id, speaker=speaker, start=cut.start, end=cut.end)
        )

        if force_overwrite or not (out_dir / save_path).exists():
            n_start = compute_num_samples(offset, sample_rate)
            n_end = n_start + compute_num_samples(cut.duration, sample_rate)
            x_hat_cut = x_hat[n_start:n_end]

            sf.write(file=str(out_dir / save_path), data=x_hat_cut, samplerate=sample_rate, format='FLAC')

            # Update offset for the next cut
            offset = add_durations(offset, cut.duration, sampling_rate=sample_rate)
        else:
            logging.info(f'File {save_path} already exists. Skipping.')

        # add enhanced recording to list
        enhanced_recordings.append(Recording.from_file(out_dir / save_path))

        # modify supervision channels since enhanced recording has only 1 channel
        enhanced_supervisions.extend(
            [
                SupervisionSegment(
                    id=str(save_path),
                    recording_id=str(save_path),
                    start=segment.start,
                    duration=segment.duration,
                    channel=0,
                    text=segment.text,
                    language=segment.language,
                    speaker=segment.speaker,
                )
                for segment in cut.supervisions
            ]
        )
    return enhanced_recordings, enhanced_supervisions


class CutEnhancer(metaclass=ABCMeta):
    """Base class for cut enhancers.
    Each enhancer should implement it's init and enhance_batch methods.
    """

    def __init__(self, cuts, context_duration, activity_garbage_class):
        """This is a required part of cut_enhancener to prepare the data loop in enhance_cuts
        """
        assert len(cuts) > 0, f'No cuts found in {cuts}'
        self.sample_rate = cuts[0].recording.sampling_rate
        self.context_duration = context_duration
        self.activity = Activity(garbage_class=activity_garbage_class, cuts=cuts)

        logging.info('Initialized %s', self.__class__.__name__)
        logging.info('\tsample_rate:            %f', self.sample_rate)
        logging.info('\tcontext_duration:       %f', self.context_duration)
        logging.info('\tactivity_garbage_class: %s', activity_garbage_class)

    def enhance_cuts(
        self,
        cuts,
        exp_dir,
        max_batch_duration=None,
        max_batch_cuts=None,
        num_workers=1,
        num_buckets=2,
        force_overwrite=False,
        torchaudio_backend='soundfile',
    ):
        """Create data loaders and enhance cuts.
        This is mostly copied from from gss.core.enhancer.Enhancer.enhance_cuts.
        """
        torchaudio.set_audio_backend(torchaudio_backend)
        torchaudio_backend = torchaudio.get_audio_backend()

        logging.info('Preparing GSS dataset and data loader')
        logging.info('\tcuts:               %s', cuts)
        logging.info('\texp_dir:            %s', exp_dir)
        logging.info('\tmax_batch_duration: %f', max_batch_duration)
        logging.info('\tmax_batch_cuts:     %d', max_batch_cuts)
        logging.info('\tnum_workers:        %d', num_workers)
        logging.info('\tnum_buckets:        %d', num_buckets)
        logging.info('\tforce_overwrite:    %s', force_overwrite)
        logging.info('\ttorchaudio backend: %s', torchaudio_backend)

        gss_dataset = GssDataset(context_duration=self.context_duration, activity=self.activity)
        gss_sampler = create_sampler(
            cuts, max_duration=max_batch_duration, max_cuts=max_batch_cuts, num_buckets=num_buckets
        )
        dl = DataLoader(
            gss_dataset,
            sampler=gss_sampler,
            batch_size=None,
            num_workers=num_workers,
            persistent_workers=False,
            prefetch_factor=4,
        )

        exp_dir = pathlib.Path(exp_dir)
        logging.info('Output directory: %s', exp_dir)

        total_processed = 0
        num_errors = 0
        futures = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            logging.info('Started a pool with %d workers', num_workers)

            for batch_idx, batch in enumerate(dl):
                batch = SimpleNamespace(**batch)

                logging.info(
                    f'Processing batch {batch_idx + 1} {batch.recording_id, batch.speaker}: '
                    f'{len(batch.orig_cuts)} segments = {batch.duration}s (total: {total_processed} segments)'
                )
                total_processed += len(batch.orig_cuts)

                out_dir = exp_dir / batch.recording_id
                out_dir.mkdir(parents=True, exist_ok=True)

                file_exists = []
                if not force_overwrite:
                    for cut in batch.orig_cuts:
                        save_path = pathlib.Path(
                            output_file_name(
                                recording_id=batch.recording_id,
                                speaker=batch.speaker, start=cut.start,
                                end=cut.end
                            )
                        )
                        file_exists.append((out_dir / save_path).exists())

                    if all(file_exists):
                        logging.info(
                            "All files already exist. Skipping this batch.")
                        continue

                num_chunks_indx = 0
                num_chunks = get_int_divisors(self.fft_length // 2 + 1)

                # If hitting OOM, split the batch into smaller chunks
                # chunks are integer divisors here, makes it faster
                while True:
                    if num_chunks_indx >= len(num_chunks):
                        logging.error(
                            f'Please reduce "max_segment_length" ! '
                            f'Error enhancing batch OOM error.\nTotal audio length {batch.duration}, '
                            f'recording id {batch.recording_id}, speaker {batch.speaker}, '
                            f'start time {cut.start} s, end time {cut.end} s.\n.'
                            f'Maximum number of chunks reached {num_chunks[-1]}.\n'
                            f'Using channel 0 instead of enhanced signal.')
                        num_errors += 1
                        x_hat = batch.audio[0].cpu().numpy()
                        break
                    try:
                        x_hat = self.enhance_batch(batch.audio, batch.activity,
                                                   batch.speaker_idx,
                                                   num_chunks=num_chunks[
                                                       num_chunks_indx],
                                                   left_context=batch.left_context,
                                                   right_context=batch.right_context)
                        break  # succesfully processed the batch
                    except (
                            torch.cuda.OutOfMemoryError,
                            decimal.InvalidOperation) as e:
                        # try again with more chunks
                        num_chunks_indx += 1

                    except Exception as e:
                        logging.error(
                            f'Error enhancing batch: {e}, using channel 0 instead of enhanced signal.')
                        num_errors += 1
                        x_hat = batch.audio[0].cpu().numpy()
                        break

                # Save the enhanced cut to disk
                batch_save = executor.submit(save_worker,
                    exp_dir,
                    batch.orig_cuts,
                    x_hat,
                    batch.recording_id,
                    batch.speaker,
                    self.sample_rate,
                    force_overwrite,
                )
                futures.append(batch_save)


        # Prepare output
        out_recordings = []
        out_supervisions = []
        for future in futures:
            enhanced_recordings, enhanced_supervisions = future.result()
            out_recordings.extend(enhanced_recordings)
            out_supervisions.extend(enhanced_supervisions)

        out_recordings = RecordingSet.from_recordings(out_recordings)
        out_supervisions = SupervisionSet.from_segments(out_supervisions)
        return num_errors, CutSet.from_manifests(recordings=out_recordings, supervisions=out_supervisions)

    @abstractmethod
    def enhance_batch(self, mic, activity,
                      speaker_id,
                      num_chunks=1,
                      left_context=0,
                      right_context=0) -> np.ndarray:
        """Enhance a batch of cuts

        This method should be implemented by the child class.
        """
        pass


class FrontEnd_v1(CutEnhancer):
    """NeMo implementation of the GSS-based frontend.
    """

    def __init__(
        self,
        stft_fft_length=1024,
        stft_hop_length=256,
        dereverb_prediction_delay=2,
        dereverb_filter_length=10,
        dereverb_num_iterations=3,
        bss_iterations=20,
        mc_filter_type='pmwf',
        mc_filter_beta=0,
        mc_filter_rank='one',
        mc_filter_postfilter='ban',
        mc_filter_num_iterations=None,
        mc_ref_channel='max_snr',
        mc_mask_min_db=-200,
        mc_postmask_min_db=0,  # no postmasking by default
        use_dtype=torch.cfloat,
        *args,
        **kwargs,
    ):
        # Inititalize the base class
        super().__init__(*args, **kwargs)

        # Params to calculate transforms samples <-> frames
        self.fft_length = stft_fft_length
        self.hop_length = stft_hop_length

        # Inititalize blocks for this frontend
        self.analysis = AudioToSpectrogram(fft_length=stft_fft_length, hop_length=stft_hop_length).cuda()
        self.synthesis = SpectrogramToAudio(fft_length=stft_fft_length, hop_length=stft_hop_length).cuda()
        self.dereverb = MaskBasedDereverbWPE(
            filter_length=dereverb_filter_length,
            prediction_delay=dereverb_prediction_delay,
            num_iterations=dereverb_num_iterations,
            dtype=use_dtype,
        ).cuda()
        self.gss = MaskEstimatorGSS(num_iterations=bss_iterations, dtype=use_dtype).cuda()
        self.mc = MaskBasedBeamformer(
            filter_type=mc_filter_type,
            filter_beta=mc_filter_beta,
            filter_rank=mc_filter_rank,
            filter_postfilter=mc_filter_postfilter,
            ref_channel=mc_ref_channel,
            mask_min_db=mc_mask_min_db,
            postmask_min_db=mc_postmask_min_db,
        ).cuda()

    def enhance_batch(self, audio, activity, speaker_id, num_chunks=1,
                      left_context=0, right_context=0) -> np.ndarray:
        """Enhance batch, as implemented in GSS package

        Args:
            audio: (channels, samples)
            activity: (channels, samples)
        """
        # Move tensors to cuda
        audio = audio.cuda()
        activity = activity.cuda()

        # Used to drop context
        left_context_frames = samples_to_frames(
            samples=left_context, fft_length=self.fft_length, hop_length=self.hop_length
        )
        right_context_frames = samples_to_frames(
            samples=right_context, fft_length=self.fft_length, hop_length=self.hop_length
        )

        # Add batch dimension
        audio = audio.unsqueeze(0)
        activity = activity.unsqueeze(0)

        with torch.inference_mode():
            # Analysis
            x_enc, _ = self.analysis(input=audio)
            a_enc = activity_time_to_timefreq(activity, win_length=self.fft_length, hop_length=self.hop_length)

            # processing is running in chunks
            F = x_enc.size(-2)
            chunk_size = int(math.ceil(F / num_chunks))

            # run dereverb and gss on chunks
            mask = []
            for n in range(num_chunks):
                n_start = n * chunk_size
                n_end = min(F, (n + 1) * chunk_size)

                x_enc_n = x_enc[..., n_start:n_end, :]

                # dereverb
                x_enc_n, _ = self.dereverb(input=x_enc_n)
                x_enc[..., n_start:n_end, :] = x_enc_n

                # mask estimator
                mask_n = self.gss(x_enc_n, a_enc)

                # append mask to the list
                mask.append(mask_n)

            # concatenate estimated masks
            mask = torch.concatenate(mask, dim=-2)

            # drop context
            mask[..., :left_context_frames] = 0
            if right_context > 0:
                mask[..., -right_context_frames:] = 0

            # form mask for the target and the undesired signals
            mask_target = mask[:, speaker_id : speaker_id + 1, ...]
            mask_undesired = torch.sum(mask, dim=1, keepdim=True) - mask_target

            # run MCF on chunks
            target_enc = []
            for n in range(num_chunks):
                n_start = n * chunk_size
                n_end = min(F, (n + 1) * chunk_size)

                # multichannel filter
                target_enc_n, _ = self.mc(
                    input=x_enc[..., n_start:n_end, :],
                    mask=mask_target[..., n_start:n_end, :],
                    mask_undesired=mask_undesired[..., n_start:n_end, :],
                )

                # append target to the list
                target_enc.append(target_enc_n)
            # concatenate estimates
            target_enc = torch.concatenate(target_enc, axis=-2)
            target, _ = self.synthesis(input=target_enc)

        # drop context from the estimated audio
        target = target[0, 0].detach().cpu().numpy().squeeze()
        target = target[left_context:]
        if right_context > 0:
            target = target[:-right_context]
        return target
