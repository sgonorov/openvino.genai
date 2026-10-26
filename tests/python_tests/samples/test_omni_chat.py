# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import soundfile as sf

from conftest import SAMPLES_CPP_DIR, SAMPLES_PY_DIR
from test_utils import run_sample

# Qwen3-Omni speech output is 24kHz mono.
SPEECH_SAMPLE_RATE = 24000
# The second turn passes no media, so it checks that speech output still works on a follow-up turn.
QUESTIONS = "Describe <ov_genai_audio_0>.\nWhat happens in the video?"
TURNS = 2
VIDEO_FRAMES = 8
VIDEO_SIZE = 128


@pytest.fixture(scope="module")
def mjpeg_video(tmp_path_factory: pytest.TempPathFactory) -> str:
    """An MJPEG AVI clip, the only video format OpenCV decodes without FFmpeg.

    The OpenCV that the C++ samples build has no FFmpeg in CI, so omni_chat can't open an .mp4 there.
    """
    path = tmp_path_factory.mktemp("omni_video") / "video.avi"
    writer = cv2.VideoWriter(
        str(path), cv2.CAP_OPENCV_MJPEG, cv2.VideoWriter_fourcc(*"MJPG"), 4.0, (VIDEO_SIZE, VIDEO_SIZE)
    )
    assert writer.isOpened(), f"Could not create {path}"
    for frame_idx in range(VIDEO_FRAMES):
        writer.write(np.full((VIDEO_SIZE, VIDEO_SIZE, 3), frame_idx * 30, dtype=np.uint8))
    writer.release()
    return str(path)


def check_speech_output(sample_dir: Path) -> None:
    for turn in range(TURNS):
        waveform, sample_rate = sf.read(sample_dir / f"output_audio_{turn}.wav")
        assert sample_rate == SPEECH_SAMPLE_RATE
        assert waveform.ndim == 1, "Speech output must be mono"
        assert waveform.size > 0, f"Turn {turn} produced an empty waveform"


class TestOmniChat:
    @pytest.mark.qwen3_omni
    @pytest.mark.samples
    @pytest.mark.parametrize("convert_model", ["tiny-random-qwen3-omni"], indirect=True)
    @pytest.mark.parametrize("download_test_content", [["monalisa.jpg", "how_are_you_doing_today.wav"]], indirect=True)
    def test_sample_omni_chat(
        self, convert_model: str, download_test_content: list[str], mjpeg_video: str, tmp_path: Path
    ):
        image, audio = download_test_content
        args = [convert_model, image, audio, mjpeg_video]
        # Each sample writes output_audio_<turn>.wav to its working directory.
        cpp_dir = tmp_path / "cpp"
        py_dir = tmp_path / "py"
        cpp_dir.mkdir()
        py_dir.mkdir()

        cpp_result = run_sample([str(SAMPLES_CPP_DIR / "omni_chat"), *args], QUESTIONS, cwd=str(cpp_dir))
        py_result = run_sample(
            [sys.executable, str(SAMPLES_PY_DIR / "omni/omni_chat.py"), *args], QUESTIONS, cwd=str(py_dir)
        )

        assert py_result.stdout == cpp_result.stdout, "Results should match"
        check_speech_output(cpp_dir)
        check_speech_output(py_dir)
