// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/visibility.hpp"

namespace ov::genai {

class OPENVINO_GENAI_EXPORTS TalkerBase {
public:
    virtual ~TalkerBase() = default;

    virtual OmniDecodedResults generate(
        const VLMDecodedResults& vlm_results,
        const OmniSpeechGenerationConfig& speech_generation_config,
        const SpeechStreamerVariant& streamer = std::monostate{}
    ) = 0;

    virtual OmniDecodedResults generate(
        const VLMDecodedResults& vlm_results,
        const AnyMap& properties = {}
    ) = 0;

    template <typename... Properties>
    util::EnableIfAllStringAny<OmniDecodedResults, Properties...> generate(
        const VLMDecodedResults& vlm_results,
        Properties&&... properties
    ) {
        return generate(vlm_results, AnyMap{std::forward<Properties>(properties)...});
    }
};

}  // namespace ov::genai
