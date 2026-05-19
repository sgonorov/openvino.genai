// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>
#include <filesystem>

#include "openvino/genai/llm_pipeline.hpp"
#include "openvino/genai/streamer_base.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "openvino/genai/visual_language/perf_metrics.hpp"
#include "openvino/genai/visual_language/video_metadata.hpp"
#include "openvino/genai/omni/speech_generation_config.hpp"

namespace ov::genai {

class OPENVINO_GENAI_EXPORTS OmniDecodedResults : public VLMDecodedResults {
public:
    /// @brief Speech output waveforms (one per generated result).
    std::vector<ov::Tensor> speech_outputs;
};

/// @brief OmniPipeline is used to generate audio on top of VLMPipeline
class OPENVINO_GENAI_EXPORTS OmniPipeline {
public:
    /// @brief Construct a pipeline from a folder containing tokenizer
    /// and model IRs.
    /// @param models_path A folder to read tokenizer and model IRs.
    /// @param device Inference device. A tokenizer is always compiled
    /// for CPU.
    /// @param properties A config to pass to ov::Core::compile_model().
    OmniPipeline(
        const std::filesystem::path& models_dir,
        const std::string& device,
        const ov::AnyMap& properties = {}
    );

    /// @brief Construct a pipeline from a map of models and their weights.
    /// @param models_map A map where key is model name (e.g. "vision_embeddings", "text_embeddings", "language", "resampler")
    /// and value is a pair of model IR as string and weights as tensor.
    /// @param tokenizer A tokenizer.
    /// @param config_dir_path A path to directory containing config.json.
    /// @param device Inference device. A tokenizer is always compiled
    /// for CPU.
    /// @param properties A config to pass to ov::Core::compile_model().
    /// @param generation_config Optional generation configuration for the pipeline.
    OmniPipeline(
        const ModelsMap& models_map,
        const Tokenizer& tokenizer,
        const std::filesystem::path& config_dir,
        const std::string& device,
        const ov::AnyMap& properties = {},
        const ov::genai::GenerationConfig& generation_config = {},
        const ov::genai::OmniSpeechGenerationConfig& speech_generation_config = {}
    );

    /// @brief Construct a pipeline from a folder containing tokenizer
    /// and model IRs. Accepts arbitrary list of optional properties.
    /// @param models_path A folder to read tokenizer and model IRs.
    /// @param device Inference device. A tokenizer is always compiled
    /// for CPU.
    /// @param properties A config to pass to ov::Core::compile_model().
    template <typename... Properties, typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
    OmniPipeline(
        const std::filesystem::path& models_path,
        const std::string& device,
        Properties&&... properties
    ): OmniPipeline(models_path, device, ov::AnyMap{std::forward<Properties>(properties)...}) {}

    /// @brief Construct a pipeline from a map of models and their weights.
    /// @param models_map A map where key is model name (e.g. "vision_embeddings", "text_embeddings", "language", "resampler")
    /// and value is a pair of model IR as string and weights as tensor.
    /// @param tokenizer A tokenizer.
    /// @param config_dir_path A path to directory containing config.json.
    /// @param device Inference device. A tokenizer is always compiled
    /// for CPU.
    /// @param properties A config to pass to ov::Core::compile_model().
    template <typename... Properties, typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
    OmniPipeline(
        const ModelsMap& models_map,
        const Tokenizer& tokenizer,
        const std::filesystem::path& config_dir,
        const std::string& device,
        Properties&&... properties
    ): OmniPipeline(models_map, tokenizer, config_dir_path, device, ov::AnyMap{std::forward<Properties>(properties)...}) {}

    OmniPipeline(
        const std::shared_ptr<VLMPipelineBase>& vlm,
        const std::filesystem::path& audio_xml,
        const std::string& audio_device,
        const ov::genai::OmniSpeechGenerationConfig& speech_generation_config,
        const ov::AnyMap& properties = {}
    );

    /// @brief Default destructor.
    ~OmniPipeline();

    OmniDecodedResults generate(
        const ChatHistory& history,
        const std::vector<ov::Tensor>& images,
        const std::vector<ov::Tensor>& videos,
        const std::vector<ov::Tensor>& audios,
        const GenerationConfig& text_generation_config,
        const OmniSpeechGenerationConfig& speech_generation_config,
        const AudioStreamerVariant& streamer
    );

    /// @brief Generate a response given a chat history and arbitrary number
    /// of ov::Property instances.
    /// @param history Chat history with messages.
    /// For using image and video tags in prompt, see:
    /// https://openvinotoolkit.github.io/openvino.genai/docs/use-cases/image-processing/#use-image-or-video-tags-in-prompt
    /// @param config_map A config may contain GenerationConfig, values
    /// for its members, StreamerVariant a single image or multiple
    /// images/videos.
    /// @return OmniDecodedResults structure containing generated texts, scores and perf metrics.
    OmniDecodedResults generate(
        const ChatHistory& history,
        const ov::AnyMap& config_map
    );

    /// @brief Generate a response given a chat history and config.
    /// @param history Chat history with messages.
    /// For using image and video tags in prompt, see:
    /// https://openvinotoolkit.github.io/openvino.genai/docs/use-cases/image-processing/#use-image-or-video-tags-in-prompt
    /// @param ...properties ov::Property instances to be combined into
    /// ov::AnyMap.
    /// @return OmniDecodedResults structure containing generated texts, scores and perf metrics.
    template <typename... Properties>
    util::EnableIfAllStringAny<OmniDecodedResults, Properties...> generate(
        const ChatHistory& history,
        Properties&&... properties
    ) {
        return generate(history, AnyMap{std::forward<Properties>(properties)...});
    }

    OmniDecodedResults generate(
        const std::string& prompt,
        const ov::AnyMap& config_map
    );

    /// @brief Generate a response given a prompt and arbitrary number
    /// of ov::Property instances.
    /// Example:
    /// generate("text", image(rgb), do_sample(true));
    /// @param prompt A prompt to respond to.
    /// For using image and video tags in prompt, see:
    /// https://openvinotoolkit.github.io/openvino.genai/docs/use-cases/image-processing/#use-image-or-video-tags-in-prompt
    /// @param ...properties ov::Property instances to be combined into
    /// ov::AnyMap.
    /// @return OmniDecodedResults structure containing generated texts, scores and perf metrics.
    /// chat_template will be applied to the prompt, run pipe.set_chat_template(custom_chat_template) to update it.
    /// To disable it for non-chat mode, please, use custom_chat_template eq "" or set generation_config.apply_chat_template to false.
    template <typename... Properties>
    util::EnableIfAllStringAny<OmniDecodedResults, Properties...> generate(
        const std::string& prompt,
        Properties&&... properties
    ) {
        return generate(
            prompt, AnyMap{std::forward<Properties>(properties)...}
        );
    }

    std::shared_ptr<ov::genai::VLMPipeline> get_vlm_pipeline() const;

    /// @brief Extract OmniSpeechGenerationConfig used to get default values.
    /// @return Default values used.
    OmniSpeechGenerationConfig get_speech_generation_config() const;

    /// @brief Override default values for OmniSpeechGenerationConfig
    /// @param new_config A config to override default values with.
    void set_speech_generation_config(const OmniSpeechGenerationConfig& new_config);

private:
    class OmniPipelineImpl;
    std::unique_ptr<OmniPipelineImpl> m_pimpl;
};

static constexpr ov::Property<ov::Tensor> return_audio{"return_audio"};
}
