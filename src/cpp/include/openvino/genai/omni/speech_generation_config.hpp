// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <limits>
#include <string>

#include "openvino/runtime/properties.hpp"
#include "openvino/genai/visibility.hpp"

namespace ov::genai {
class OPENVINO_GENAI_EXPORTS OmniSpeechGenerationConfig {
public:
    OmniSpeechGenerationConfig() = default;
    explicit OmniSpeechGenerationConfig(const std::filesystem::path& json_path);

    /// @brief Enable speech output generation (requires model with talker support).
    bool return_audio = true;
    /// @brief Speaker name for speech output. Empty string selects the model's default speaker.
    /// Available names are model-specific and listed under `talker_config.speaker_id` in the
    /// model's config.json.
    std::string speaker;
    /// @brief Number of codec frames to accumulate before streaming each audio chunk.
    /// Each frame is 80ms of audio at 24kHz (1920 samples). Default 1.
    /// Streaming is controlled by presence of an audio_streamer callback, not by this value.
    size_t audio_chunk_frames = 1;
};
static constexpr ov::Property<bool> return_audio{"return_audio"};
static constexpr ov::Property<std::string> speaker{"speaker"};
static constexpr ov::Property<size_t> audio_chunk_frames{"audio_chunk_frames"};
}  // namespace ov::genai
