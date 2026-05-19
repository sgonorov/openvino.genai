// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <limits>
#include <string>

#include "openvino/runtime/infer_request.hpp"

namespace ov::genai {
class OPENVINO_GENAI_EXPORTS OmniSpeechGenerationConfig {
public:
    size_t max_new_tokens = SIZE_MAX;
    size_t max_length = SIZE_MAX;
    bool ignore_eos = false;
    size_t min_new_tokens = 0;
    bool echo = false;
    size_t logprobs = 0;

    // EOS special token
    int64_t eos_token_id = -1;
    std::set<std::string> stop_strings;
    // Default setting in vLLM (and OpenAI API) is not to include stop string in the output
    bool include_stop_str_in_output = false;
    std::set<int64_t> stop_token_ids;

    // penalties (not used in beam search)
    float repetition_penalty = 1.0f;
    float presence_penalty = 0.0f;
    float frequency_penalty = 0.0f;

    // Beam search specific
    size_t num_beam_groups = 1;
    size_t num_beams = 1;
    float diversity_penalty = 0.0f;
    float length_penalty = 1.0f;
    size_t num_return_sequences = 1;
    size_t no_repeat_ngram_size = std::numeric_limits<size_t>::max();
    StopCriteria stop_criteria = StopCriteria::HEURISTIC;

    // Multinomial
    float temperature = 1.0f;
    float top_p = 1.0f;
    size_t top_k = std::numeric_limits<size_t>::max();
    float min_p = 0.0f;
    bool do_sample = false;
    size_t rng_seed = 0;

    // CDPruner config
    size_t pruning_ratio = 0;  // 0 means disabled, and values from 1 to 100 represent the percentage to prune.
    float relevance_weight = 0.5f;

    // Assisting generation parameters
    float assistant_confidence_threshold = 0.f;
    size_t num_assistant_tokens = 0;
    size_t max_ngram_size = 0;

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
}  // namespace ov::genai
