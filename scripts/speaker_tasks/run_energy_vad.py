"""
# Script Usage:
python run_energy_vad.py \
    --manifest_file PATH_TO_MANIFEST \
    --out_dir PATH_TO_OUT_DIR \
    --threshold 10 \  # threshold for energy vad (dB above mean)
    --use_ctm # set this flag to save CTM files

# Function Usage:
## init vars
fname = "/media/data/datasets/vad_sd/multilingual_vad/mandarin/aishell2/audio/dev/wav/D0012/ID0012W0162.wav"
fs, sig = scipy.io.wavfile.read(fname)
sig = sig + 1e-10
# run naive vad
energy, vad, voiced = naive_frame_energy_vad(sig, fs, threshold=-35,
                                                win_len=0.025, win_hop=0.025)

## plot results
multi_plots(data=[sig, energy, vad, voiced],
            titles=["Input signal (voiced + silence)", "Short time energy",
                    "Voice activity detection", "Output signal (voiced only)"],
            fs=fs, plot_rows=4, step=1)

## save voiced signal
scipy.io.wavfile.write("naive_frame_energy_vad_no_silence_"+ fname,
                        fs,  np.array(voiced, dtype=sig.dtype))
"""

import argparse
import json
import multiprocessing as mp
import os
import random
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.io.wavfile
import scipy.signal
from tqdm import tqdm


def stride_trick(a, stride_length, stride_step):
    """
    apply framing using the stride trick from numpy.
    Source: https://superkogito.github.io/blog/2020/02/09/naive_vad.html
    Args:
        a (array) : signal array.
        stride_length (int) : length of the stride.
        stride_step (int) : stride step.

    Returns:
        blocked/framed array.
    """
    nrows = ((a.size - stride_length) // stride_step) + 1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows, stride_length), strides=(stride_step * n, n))


def framing(sig, fs=16000, win_len=0.025, win_hop=0.01):
    """
    transform a signal into a series of overlapping frames (=Frame blocking).
    Source: https://superkogito.github.io/blog/2020/02/09/naive_vad.html
    Args:
        sig     (array) : a mono audio signal (Nx1) from which to compute features.
        fs        (int) : the sampling frequency of the signal we are working with.
                          Default is 16000.
        win_len (float) : window length in sec.
                          Default is 0.025.
        win_hop (float) : step between successive windows in sec.
                          Default is 0.01.

    Returns:
        array of frames.
        frame length.

    Notes:
    ------
        Uses the stride trick to accelerate the processing.
    """
    # run checks and assertions
    if win_len < win_hop:
        print("ParameterError: win_len must be larger than win_hop.")

    # compute frame length and frame step (convert from seconds to samples)
    frame_length = win_len * fs
    frame_step = win_hop * fs
    signal_length = len(sig)
    frames_overlap = frame_length - frame_step

    # compute number of frames and left sample in order to pad if needed to make
    # sure all frames have equal number of samples  without truncating any samples
    # from the original signal
    rest_samples = np.abs(signal_length - frames_overlap) % np.abs(frame_length - frames_overlap)
    pad_signal = np.append(sig, np.array([0] * int(frame_step - rest_samples) * int(rest_samples != 0.0)))

    # apply stride trick
    frames = stride_trick(pad_signal, int(frame_length), int(frame_step))
    return frames, frame_length


def _calculate_normalized_short_time_energy(frames):
    """
    Source: https://superkogito.github.io/blog/2020/02/09/naive_vad.html
    """
    return np.sum(np.abs(np.fft.rfft(a=frames, n=len(frames))) ** 2, axis=-1) / len(frames) ** 2


def naive_frame_energy_vad(sig, fs, threshold=0, win_len=0.25, win_hop=0.25, E0=1e7, zero_mean=True):
    """
    Adapted from: https://superkogito.github.io/blog/2020/02/09/naive_vad.html
    """

    # Prevent zero signal values for STFFT
    sig = sig + 1e-10

    # framing
    frames, frames_len = framing(sig=sig, fs=fs, win_len=win_len, win_hop=win_hop)

    # compute short time energies to get voiced frames
    energy = _calculate_normalized_short_time_energy(frames)
    log_energy = 10 * np.log10(energy / E0)

    # normalize energy to 0 dB then filter and format
    if zero_mean:
        log_energy -= np.mean(log_energy)

    energy = scipy.signal.medfilt(log_energy, 5)
    energy = np.repeat(energy, frames_len)

    if threshold is None:
        threshold = np.mean(energy)

    # compute vad and get speech frames
    vad = np.array(energy > threshold, dtype=sig.dtype)
    vframes = np.array(frames.flatten()[np.where(vad == 1)], dtype=sig.dtype)
    return energy, vad, np.array(vframes, dtype=np.float64)


def multi_plots(data, titles, fs, plot_rows, step=1, colors=["b", "r", "m", "g", "b", "y"]):
    """
    Source: https://superkogito.github.io/blog/2020/02/09/naive_vad.html
    """
    # first fig
    plt.subplots(plot_rows, 1, figsize=(20, 10))
    plt.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.99, wspace=0.4, hspace=0.99)

    for i in range(plot_rows):
        plt.subplot(plot_rows, 1, i + 1)
        y = data[i]
        plt.plot([i / fs for i in range(0, len(y), step)], y, colors[i])
        plt.gca().set_title(titles[i])
    plt.show()

    # second fig
    sig, vad = data[0], data[-2]
    # plot VAD and orginal signal
    plt.subplots(1, 1, figsize=(20, 10))
    plt.plot([i / fs for i in range(len(sig))], sig, label="Signal")
    plt.plot([i / fs for i in range(len(vad))], max(sig) * vad, label="VAD")
    plt.legend(loc='best')
    plt.show()


def write_vad_to_rttm_file(audio_file, vad_list, output_file, sample_rate=16000, max_speakers=100):
    i = 0
    start = 0
    end = 0
    audio_key = Path(audio_file).stem
    if "/mls/" in str(audio_file):
        speaker_id = audio_key.split("_")[0]
    else:
        speaker_id = f"Fake_{np.random.choice(max_speakers)}"
    segments = []
    while i < len(vad_list):
        while i < len(vad_list) and vad_list[i] == 0:
            i += 1
        start = i
        while i < len(vad_list) and vad_list[i] == 1:
            i += 1
        end = i

        if start < end < len(vad_list):
            line = f"SPEAKER {audio_key} 1 {start/sample_rate:.2f} {(end - start)/sample_rate:.2f} <NA> <NA> {speaker_id} <NA> <NA>"
            segments.append(line)

    if len(segments) == 0:
        line = f"SPEAKER {audio_key} 1 0.0 0.0 <NA> <NA> {speaker_id} <NA> <NA>"
        segments.append(line)

    with open(output_file, "w") as fout:
        for line in segments:
            fout.write(f"{line}\n")
    return output_file


def load_and_split_segments(
    filename, decimals=2, global_duration_thres=0.5, duration_thres=1.0, seg_len_range=[0.4, 1.0], eps=0.1
):
    """
    Read RTTM file and split random segments
    """
    segment_lines = open(filename, 'r').readlines()
    segment_data_list = []
    utt_id = Path(filename).stem
    spk_id = "NA"
    stt_time = 0.0
    duration = 0.0
    for seg_line in segment_lines:
        seg_line_split = seg_line.split()
        utt_id = seg_line_split[1]
        spk_id = seg_line_split[7]
        stt_time = round(float(seg_line_split[3]), decimals)
        duration = round(float(seg_line_split[4]), decimals)
        if duration > global_duration_thres:
            if duration > duration_thres:
                cursor = stt_time
                curr_dur_sum = 0
                while cursor + (eps * 0.99) < (stt_time + duration):
                    if (stt_time + duration) - cursor < seg_len_range[1]:
                        curr_dur = round((stt_time + duration) - cursor, 2)
                    else:
                        curr_dur = round(random.uniform(seg_len_range[0], seg_len_range[1]), 2)
                    curr_dur_sum += curr_dur
                    if curr_dur > seg_len_range[0]:
                        segment_data_list.append((utt_id, spk_id, cursor, curr_dur))
                    cursor = round(cursor + curr_dur, 2)
                assert (
                    abs(curr_dur_sum - duration) < eps
                ), f"duration mismatch, curr_dur_sum:{curr_dur_sum:.2f}, duration:{duration:.2f}"

            else:
                segment_data_list.append((utt_id, spk_id, stt_time, duration))

    if len(segment_data_list) == 0:
        segment_data_list.append((utt_id, spk_id, stt_time, duration))
    return segment_data_list


def save_segments_to_ctm_file(segments_list, output_file):
    with open(output_file, 'w') as fout:
        for utt_id, spk_id, start_time, dur_time in segments_list:
            word = "<unk>"
            fout.write(f"{utt_id} {spk_id} {start_time:.2f} {dur_time:.2f} {word} 0\n")
    return output_file


def process_audio(inputs):
    audio_file, out_dir, threshold, enable_ctm = inputs
    rttm_file = os.path.join(out_dir, Path(audio_file).stem + ".rttm")
    ctm_file = os.path.join(out_dir, Path(audio_file).stem + ".ctm")
    sample_rate = 16000
    if not Path(audio_file).is_file():
        return audio_file, None, None

    try:
        sig, _ = librosa.load(audio_file, sr=sample_rate)
    except:
        print(f"Error reading {audio_file}")
        return audio_file, None, None

    energy, vad, voiced = naive_frame_energy_vad(
        sig, sample_rate, threshold=threshold, win_len=0.025, win_hop=0.025, zero_mean=True
    )
    write_vad_to_rttm_file(audio_file, vad, rttm_file, sample_rate=sample_rate, max_speakers=100)
    if not enable_ctm:
        return audio_file, rttm_file, None
    word_segments = load_and_split_segments(rttm_file)
    save_segments_to_ctm_file(word_segments, ctm_file)
    return audio_file, rttm_file, ctm_file


def run_energy_vad_on_manifest(manifest_file, out_dir, threshold=10, enable_ctm=False):
    if not Path(out_dir).is_dir():
        Path(out_dir).mkdir(parents=True, exist_ok=True)

    inputs = []
    print("Loading manifest file...")
    with open(manifest_file, 'r') as fin:
        for line in fin.readlines():
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            inputs.append((item["audio_filepath"], out_dir, threshold, enable_ctm))

    with mp.Pool(mp.cpu_count()) as pool:
        results = list(
            tqdm(pool.imap(process_audio, inputs), total=len(inputs), desc='Running energy VAD', leave=True,)
        )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_file", type=str, required=True, help="manifest file")
    parser.add_argument("--out_dir", type=str, required=True, help="output directory")
    parser.add_argument("--threshold", type=float, default=10, help="threshold for energy VAD")
    parser.add_argument("--enable_ctm", action="store_true", help="whether to get fake ctm file")

    args = parser.parse_args()
    run_energy_vad_on_manifest(**vars(args))
    print(f"Results saved in {args.out_dir}")
