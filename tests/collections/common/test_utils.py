# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import os
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import pytest

from nemo.collections.common.parts.preprocessing.manifest import get_full_path
from nemo.collections.common.parts.utils import flatten


class TestListUtils:
    @pytest.mark.unit
    def test_flatten(self):
        """Test flattening an iterable with different values: str, bool, int, float, complex.
        """
        test_cases = []
        test_cases.append({'input': ['aa', 'bb', 'cc'], 'golden': ['aa', 'bb', 'cc']})
        test_cases.append({'input': ['aa', ['bb', 'cc']], 'golden': ['aa', 'bb', 'cc']})
        test_cases.append({'input': ['aa', [['bb'], [['cc']]]], 'golden': ['aa', 'bb', 'cc']})
        test_cases.append({'input': ['aa', [[1, 2], [[3]], 4]], 'golden': ['aa', 1, 2, 3, 4]})
        test_cases.append({'input': [True, [2.5, 2.0 + 1j]], 'golden': [True, 2.5, 2.0 + 1j]})

        for n, test_case in enumerate(test_cases):
            assert flatten(test_case['input']) == test_case['golden'], f'Test case {n} failed!'


class TestPreprocessingUtils:
    @pytest.mark.unit
    def test_get_full_path_local(self, tmpdir):
        """Test with local paths
        """
        # Create a few files
        num_files = 10

        audio_files_relative_path = [f'file_{n}.test' for n in range(num_files)]
        audio_files_absolute_path = [os.path.join(tmpdir, a_file_rel) for a_file_rel in audio_files_relative_path]

        data_dir = tmpdir
        manifest_file = os.path.join(data_dir, 'manifest.json')

        # Context manager to create dummy files
        @contextmanager
        def create_files(paths):
            # Create files
            for a_file in paths:
                Path(a_file).touch()
            yield
            # Remove files
            for a_file in paths:
                Path(a_file).unlink()

        # 1) Test with absolute paths and while files don't exist.
        # Note: it's still expected the path will be resolved correctly, since it will be
        # expanded using manifest_file.parent or data_dir and relative path.
        # - single file
        for n in range(num_files):
            assert (
                get_full_path(audio_files_absolute_path[n], manifest_file=manifest_file)
                == audio_files_absolute_path[n]
            )
            assert get_full_path(audio_files_absolute_path[n], data_dir=data_dir) == audio_files_absolute_path[n]

        # - all files in a list
        assert get_full_path(audio_files_absolute_path, manifest_file=manifest_file) == audio_files_absolute_path
        assert get_full_path(audio_files_absolute_path, data_dir=data_dir) == audio_files_absolute_path

        # 2) Test with absolute paths and existing files.
        with create_files(audio_files_absolute_path):
            # - single file
            for n in range(num_files):
                assert (
                    get_full_path(audio_files_absolute_path[n], manifest_file=manifest_file)
                    == audio_files_absolute_path[n]
                )
                assert get_full_path(audio_files_absolute_path[n], data_dir=data_dir) == audio_files_absolute_path[n]

            # - all files in a list
            assert get_full_path(audio_files_absolute_path, manifest_file=manifest_file) == audio_files_absolute_path
            assert get_full_path(audio_files_absolute_path, data_dir=data_dir) == audio_files_absolute_path

        # 3) Test with relative paths while files don't exist.
        # This is a situation we may have with a tarred dataset.
        # In this case, we expect to return the relative path.
        # - single file
        for n in range(num_files):
            assert (
                get_full_path(audio_files_relative_path[n], manifest_file=manifest_file)
                == audio_files_relative_path[n]
            )
            assert get_full_path(audio_files_relative_path[n], data_dir=data_dir) == audio_files_relative_path[n]

        # - all files in a list
        assert get_full_path(audio_files_relative_path, manifest_file=manifest_file) == audio_files_relative_path
        assert get_full_path(audio_files_relative_path, data_dir=data_dir) == audio_files_relative_path

        # 4) Test with relative paths and existing files.
        # In this case, we expect to return the absolute path.
        with create_files(audio_files_absolute_path):
            # - single file
            for n in range(num_files):
                assert (
                    get_full_path(audio_files_relative_path[n], manifest_file=manifest_file)
                    == audio_files_absolute_path[n]
                )
                assert get_full_path(audio_files_relative_path[n], data_dir=data_dir) == audio_files_absolute_path[n]

            # - all files in a list
            assert get_full_path(audio_files_relative_path, manifest_file=manifest_file) == audio_files_absolute_path
            assert get_full_path(audio_files_relative_path, data_dir=data_dir) == audio_files_absolute_path

    @pytest.mark.unit
    def test_get_full_path_ais(self, tmpdir):
        """Test with paths on AIStore.
        """
        # Create a few files
        num_files = 10

        audio_files_relative_path = [f'file_{n}.test' for n in range(num_files)]
        audio_files_cache_path = [os.path.join(tmpdir, a_file_rel) for a_file_rel in audio_files_relative_path]

        ais_data_dir = 'ais://test'
        ais_manifest_file = os.path.join(ais_data_dir, 'manifest.json')

        # Context manager to create dummy files
        @contextmanager
        def create_files(paths):
            # Create files
            for a_file in paths:
                Path(a_file).touch()
            yield
            # Remove files
            for a_file in paths:
                Path(a_file).unlink()

        # Simulate caching in local tmpdir
        def datastore_path_to_cache_path_in_tmpdir(path):
            rel_path = os.path.relpath(path, start=os.path.dirname(ais_manifest_file))

            if rel_path in audio_files_relative_path:
                idx = audio_files_relative_path.index(rel_path)
                return audio_files_cache_path[idx]
            else:
                raise ValueError(f'Unexpected path {path}')

        with mock.patch(
            'nemo.collections.common.parts.preprocessing.manifest.datastore_path_to_local_path',
            datastore_path_to_cache_path_in_tmpdir,
        ):
            # Test with relative paths and existing cached files.
            # We expect to return the absolute path in the local cache.
            with create_files(audio_files_cache_path):
                # - single file
                for n in range(num_files):
                    assert (
                        get_full_path(audio_files_relative_path[n], manifest_file=ais_manifest_file)
                        == audio_files_cache_path[n]
                    )
                    assert (
                        get_full_path(audio_files_relative_path[n], data_dir=ais_data_dir) == audio_files_cache_path[n]
                    )

                # - all files in a list
                assert (
                    get_full_path(audio_files_relative_path, manifest_file=ais_manifest_file) == audio_files_cache_path
                )
                assert get_full_path(audio_files_relative_path, data_dir=ais_data_dir) == audio_files_cache_path
