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
# Code adapted from https://github.com/lhotse-speech/lhotse/blob/master/lhotse/bin/modes/cut.py


from typing import Optional
from lhotse.cut import CutSet
from lhotse.serialization import load_manifest_lazy_or_eager
from lhotse.utils import Pathlike


def simple_cut(
    output_cut_manifest: Pathlike,
    force_eager: bool,
    recording_manifest: Optional[Pathlike] = None,
    feature_manifest: Optional[Pathlike] = None,
    supervision_manifest: Optional[Pathlike] = None,
):
    """
    Create a CutSet stored in OUTPUT_CUT_MANIFEST. Depending on the provided options, it may contain any combination
    of recording, feature and supervision manifests.
    Either RECORDING_MANIFEST or FEATURE_MANIFEST has to be provided.
    When SUPERVISION_MANIFEST is provided, the cuts time span will correspond to that of the supervision segments.
    Otherwise, that time span corresponds to the one found in features, if available, otherwise recordings.

    .. hint::
        ``--force-eager`` must be used when the RECORDING_MANIFEST is not sorted by recording ID.

    Args: 
        output_cut_manifest: output path to store the CutSet manifest.
        recording_manifest: will be used to attach the recordings to the cuts
        feature_manifest: will be used to attach the features to the cuts.
        supervision_manifest: will be used to attach the supervisions to the cuts.
        force_eager: Force reading full manifests into memory before creating the manifests (useful when you are not sure about the input manifest sorting).
    """
    supervision_set, feature_set, recording_set = [
        load_manifest_lazy_or_eager(p) if p is not None else None
        for p in (supervision_manifest, feature_manifest, recording_manifest)
    ]

    if all(m is None or m.is_lazy for m in (supervision_set, feature_set, recording_set)) and not force_eager:
        # Create the CutSet lazily; requires sorting by recording_id
        CutSet.from_manifests(
            recordings=recording_set,
            supervisions=supervision_set,
            features=feature_set,
            output_path=output_cut_manifest,
            lazy=True,
        )
    else:
        cut_set = CutSet.from_manifests(recordings=recording_set, supervisions=supervision_set, features=feature_set)
        cut_set.to_file(output_cut_manifest)


def trim_to_supervisions(
    cuts: Pathlike,
    output_cuts: Pathlike,
    keep_overlapping: bool,
    min_duration: Optional[float] = None,
    context_direction: str = "center",
    keep_all_channels: bool = False,
):
    """
    Splits each input cut into as many cuts as there are supervisions.
    These cuts have identical start times and durations as the supervisions.
    When there are overlapping supervisions, they can be kept or discarded with options.

    \b
    For example, the following cut:
                Cut
        |-----------------|
         Sup1
        |----|  Sup2
           |-----------|

    \b
    is transformed into two cuts:
         Cut1
        |----|
         Sup1
        |----|
           Sup2
           |-|
                Cut2
           |-----------|
           Sup1
           |-|
                Sup2
           |-----------|

    Args:
        cuts: input cuts manifest
        output_cuts: output cuts manifest
        keep_overlapping: when `False`, it will discard parts of other supervisions that overlap with the main supervision. In the illustration, it would discard `Sup2` in `Cut1` and `Sup1` in `Cut2`.
        min_duration: An optional duration in seconds; specifying this argument will extend the cuts
            that would have been shorter than `min_duration` with actual acoustic context in the recording/features.
            If there are supervisions present in the context, they are kept when `keep_overlapping` is true.
            If there is not enough context, the returned cut will be shorter than `min_duration`.
            If the supervision segment is longer than `min_duration`, the return cut will be longer.
        context_diraction: Which direction should the cut be expanded towards to include context.
            The value of "center" implies equal expansion to left and right;
            "random" uniformly samples a value between "left" and "right".
        keep_all_channels: If ``True``, the output cut will have the same channels as the input cut. By default,
            the trimmed cut will have the same channels as the supervision.
    """
    assert context_direction in ("left", "center", "right", "random")
    cuts = CutSet.from_file(cuts)

    with CutSet.open_writer(output_cuts) as writer:
        for cut in cuts.trim_to_supervisions(
            keep_overlapping=keep_overlapping,
            min_duration=min_duration,
            context_direction=context_direction,
            keep_all_channels=keep_all_channels,
        ):
            writer.write(cut)
