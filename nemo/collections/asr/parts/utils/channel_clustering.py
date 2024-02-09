# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

# Copyright (c) 2007-2020 The scikit-learn developers.

# BSD 3-Clause License

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# NME-SC clustering is based on the implementation from the paper
# https://arxiv.org/pdf/2003.02405.pdf and the implementation from
# https://github.com/tango4j/Auto-Tuning-Spectral-Clustering.

import torch
import numpy as np

# from tests.collections.asr.test_diar_utils import generate_toy_data
from nemo.collections.asr.parts.utils.speaker_utils import (
    check_ranges,
    get_new_cursor_for_update,
    get_online_subsegments_from_buffer,
    get_speech_labels_for_update,
    get_subsegments,
    get_target_sig,
    merge_float_intervals,
    merge_int_intervals,
)
from nemo.collections.asr.parts.utils.offline_clustering import (
    NMESC,
    SpectralClustering,
    SpeakerClustering,
    get_scale_interpolated_embs,
    getAffinityGraphMat,
    getCosAffinityMatrix,
    split_input_data,
    cos_similarity,
)
from nemo.collections.asr.parts.utils.audio_utils import estimated_coherence
from nemo.collections.asr.modules.audio_preprocessing import AudioToSpectrogram


def generate_orthogonal_embs(total_spks, perturb_sigma, emb_dim):
    """Generate a set of artificial orthogonal embedding vectors from random numbers
    """
    gaus = torch.randn(emb_dim, emb_dim)
    _svd = torch.linalg.svd(gaus)
    orth = _svd[0] @ _svd[2]
    orth_embs = orth[:total_spks]
    # Assert orthogonality
    assert torch.abs(getCosAffinityMatrix(orth_embs) - torch.diag(torch.ones(total_spks))).sum() < 1e-4
    return orth_embs

def generate_toy_data(
    n_clusters=2,
    spk_dur=10,
    emb_dim=192,
    perturb_sigma=0.0,
    ms_window=[1.5],
    ms_shift=[0.75],
    torch_seed=0,
):
    torch.manual_seed(torch_seed)
    spk_timestamps = [(spk_dur * k, spk_dur) for k in range(n_clusters)]
    emb_list, seg_list = [], []
    multiscale_segment_counts = [0 for _ in range(len(ms_window))]
    ground_truth = []
    random_orthogonal_embs = generate_orthogonal_embs(n_clusters, perturb_sigma, emb_dim)
    for scale_idx, (window, shift) in enumerate(zip(ms_window, ms_shift)):
        for spk_idx, (offset, dur) in enumerate(spk_timestamps):
            segments_stt_dur = get_subsegments(offset=offset, window=window, shift=shift, duration=dur)
            segments = [[x[0], x[0] + x[1]] for x in segments_stt_dur]
            emb_cent = random_orthogonal_embs[spk_idx, :]
            emb = emb_cent.tile((len(segments), 1)) + 0.1 * torch.rand(len(segments), emb_dim)
            seg_list.extend(segments)
            emb_list.append(emb)
            multiscale_segment_counts[scale_idx] += emb.shape[0]

            if scale_idx == len(multiscale_segment_counts) - 1:
                ground_truth.extend([spk_idx] * emb.shape[0])

    emb_tensor = torch.concat(emb_list)
    multiscale_segment_counts = torch.tensor(multiscale_segment_counts)
    segm_tensor = torch.tensor(seg_list)
    multiscale_weights = torch.ones(len(ms_window)).unsqueeze(0)
    ground_truth = torch.tensor(ground_truth)
    return emb_tensor, segm_tensor, multiscale_segment_counts, multiscale_weights, spk_timestamps, ground_truth


def channel_cluster(
    mat: torch.Tensor,
    oracle_num_channels: int = -1,
    max_rp_threshold: float = 0.5,
    max_num_channels: int = 7,
    min_num_channels: int = 3,
    sparse_search: bool = True,
    sparse_search_volume: int = 30,
    fixed_thres: float = -1.0,
    force_fully_connected: bool = False,
    kmeans_random_trials: int = 10,
    cuda: bool = True,
) -> torch.LongTensor:

    nmesc = NMESC(
        mat,
        max_num_speakers=max_num_channels,
        max_rp_threshold=max_rp_threshold,
        sparse_search=sparse_search,
        sparse_search_volume=sparse_search_volume,
        fixed_thres=fixed_thres,
        nme_mat_size=100,
        maj_vote_spk_count=False,
        force_fully_connected=force_fully_connected,
        cuda=True,
    )
    device = mat.device
    min_samples_for_nmesc = 6
    # If there are less than `min_samples_for_nmesc` segments, est_num_of_spk is 1.
    if mat.shape[0] > min_samples_for_nmesc:
        est_num_of_spk, p_hat_value = nmesc.forward()
        affinity_mat = getAffinityGraphMat(mat, p_hat_value)
    else:
        nmesc.fixed_thres = max_rp_threshold
        est_num_of_spk, p_hat_value = nmesc.forward()
        affinity_mat = mat
    
    # Clip the estimated number of speakers to the range of [min_num_channels, max_num_channels]
    est_num_of_spk = torch.clamp(est_num_of_spk, min=min_num_channels, max=max_num_channels)

    # n_clusters is number of speakers estimated from spectral clustering.
    if oracle_num_channels > 0:
        n_clusters = int(oracle_num_channels)
    else:
        n_clusters = int(est_num_of_spk.item())

    spectral_model = SpectralClustering(
        n_clusters=n_clusters, n_random_trials=kmeans_random_trials, cuda=cuda, device=device
    )
    Y = spectral_model.forward(affinity_mat)
    return Y

def freq_to_subband(freq, fft_length, sample_rate):
    """Convert freq to subband

    Args:
        freq: frequency in Hz

    Returns:
        Subband index
    """
    num_subbands = fft_length//2 + 1
    return int(np.round(freq * fft_length / sample_rate))

def channel_cluster_from_coherence(
        audio_signal: torch.Tensor,
        sample_rate: int = 16000,
        fft_length: int = 1024,
        hop_length: int = 256,
        freq_min: float = 300,
        freq_max: float = 3500,
        mag_power: float = 2,
        output_coherence: bool = False,
        ) -> torch.LongTensor:
    """
    Args:
        audio_signal: Multichannel time domain signal, shape (channel, time)
        sample_rate: sample rate of the audio signal
        fft_length: length of the window and FFT for the STFT
        hop_length: hop length for the STFT
        freq_min: min frequency to consider, in Hz
        freq_max: max frequency to consider, in Hz
    Returns:
        Cluster assignments
    """
    k_min = freq_to_subband(freq_min, fft_length, sample_rate)
    k_max = freq_to_subband(freq_max, fft_length, sample_rate)

    stft = AudioToSpectrogram(fft_length=fft_length, hop_length=hop_length).to(device=audio_signal.device)

    # analysis transform
    A_spec, _ = stft(input=audio_signal[None, ...])

    # reshape to (freq, time, channel)
    A_spec = A_spec[0].permute(1,2,0).cpu().numpy()

    # estimate coherence
    coherence = estimated_coherence(A_spec[k_min:k_max, ...])
    # use (magnitude coherence)^power
    mag_coherence = np.abs(coherence) ** mag_power
    # average across subbands
    mag_coherence = np.mean(mag_coherence, axis=0)

    # run clustering
    clusters = channel_cluster(mat=torch.tensor(mag_coherence))

    if output_coherence:
        return clusters, mag_coherence
    else:
        return clusters

if __name__ == "__main__":
    for n_clusters in [2,3,4,5]:
        res = generate_toy_data(n_clusters=n_clusters)
        embs = res[0]
        mat = cos_similarity(embs, embs)
        Y = channel_cluster(mat=mat)

        print(f"Y: {Y}")
        print(f"len(set(Y)): {len(set(Y.cpu().tolist()))}, n_clusters: {n_clusters}")