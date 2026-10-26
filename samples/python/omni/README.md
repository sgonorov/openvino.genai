# Qwen3-Omni Chat Sample (Python)

> **Preview:** The Qwen3-Omni API (`OmniPipeline` and related types) is a preview feature and is subject to change in future releases.

This example demonstrates interactive multimodal chat with Qwen3-Omni models: text, image, audio, and video input producing text and optionally synthesized speech output. The sample features `openvino_genai.OmniPipeline` and configures it for the chat scenario using the `ChatHistory` API.

The following are sample files:
 - [`omni_chat.py`](./omni_chat.py) demonstrates multimodal chat with optional speech synthesis.

## Download and convert the model and tokenizers

The `--upgrade-strategy eager` option is needed to ensure `optimum-intel` is upgraded to the latest version.

Install [../../export-requirements.txt](../../export-requirements.txt) to convert a model.

```sh
pip install --upgrade-strategy eager -r ../../export-requirements.txt
```

Then export a Qwen3-Omni model to OpenVINO format using the Optimum Intel CLI or Python API.

Install [deployment-requirements.txt](../../deployment-requirements.txt) via `pip install -r ../../deployment-requirements.txt` to run the sample.

## Get test image, audio and video

[This image](https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/d5fbbd1a-d484-415c-88cb-9986625b7b11) can be used as a sample image.

Download an example 16kHz mono WAV file:

```sh
wget https://storage.openvinotoolkit.org/models_contrib/speech/2021.2/librispeech_s5/how_are_you_doing_today.wav
```

Download an example video file:

```sh
wget https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/video/Coco%20Walking%20in%20Berkeley.mp4
```

## Run the sample

```sh
python omni_chat.py <MODEL_DIR> <IMAGE_FILE_OR_DIR> <AUDIO_FILE> <VIDEO_FILE>
```

**Parameters:**
- `<MODEL_DIR>` — Path to the exported Qwen3-Omni OpenVINO model directory.
- `<IMAGE_FILE_OR_DIR>` — Path to an input image or a directory of images for visual context.
- `<AUDIO_FILE>` — Path to an input audio file (16kHz mono WAV). Refer to it in a question as `<ov_genai_audio_0>`.
- `<VIDEO_FILE>` — Path to an input video file.

**Example:**

```sh
python omni_chat.py ./qwen3-omni-ov ./coco.jpg ./how_are_you_doing_today.wav "./Coco Walking in Berkeley.mp4"
```

Images, video and audio are loaded once at startup and sent with the first question. Type questions and press Enter; the model responds with streaming text and, when speech output is enabled, 24kHz mono PCM samples in `OmniDecodedResults.speech_result.waveforms`. Each turn's speech is saved to `output_audio_<turn>.wav` in the working directory. Press Ctrl+D to exit.

## Audio placement

Every audio you pass gets an index, starting at zero. Write `<ov_genai_audio_N>` in a question to
put audio N at that exact position, for example `Describe <ov_genai_audio_0> in one sentence`.
The full tag rules are in
[Use Media Tags in Prompt](https://openvinotoolkit.github.io/openvino.genai/docs/use-cases/visual-processing/#use-media-tags-in-prompt).

Omit the tags and the audio is prepended to your text instead. That default is also the layout
Qwen3-Omni was trained on, so prefer it unless you specifically need the audio elsewhere. Placing
audio between spans of text works, but it is off the training distribution and answers may be
weaker.

Media attaches to the turn you supply it on. Later turns pass no media and refer to earlier
media through the chat history rather than sending the tensors again.

## Speech synthesis

Text decoding (the "thinker" phase) and speech output (the "talker" phase) are configured separately:

```python
text_config = openvino_genai.GenerationConfig()
text_config.max_new_tokens = 256

talker_speech_config = openvino_genai.OmniTalkerSpeechConfig(model_dir)
talker_speech_config.return_audio = True  # Enable speech synthesis
talker_speech_config.speaker = "Cherry"   # Select voice (optional)
```

Available voices vary by checkpoint. MoE models typically expose `"Ethan"`, `"Chelsie"`, `"Aiden"`, `"Cherry"`. Check `talker_config.speaker_id` in the model's `config.json` for the full list. Leaving `speaker` empty selects the default voice. Set `talker_speech_config.return_audio = False` for text-only responses.

## GPU inference

Qwen3-Omni speech synthesis requires FP32 precision on GPU — FP16 causes numerical drift that corrupts codec tokens and distorts audio. The pipeline enforces `INFERENCE_PRECISION_HINT=f32` automatically for GPU devices. Change the device by passing it to the `OmniPipeline` constructor:

```python
pipe = openvino_genai.OmniPipeline(model_dir, "GPU")
```

MoE models (30B-A3B) may exceed available GPU memory; use CPU for MoE inference.

## Speaker API features

The sample shows speaker enumeration and voice blending at startup:

1. **List speakers** — shows all named voices in the model
2. **Voice blending** — when 2+ speakers are available, creates a 50/50 blend and uses it for generation

**Custom blending:** Retrieve speaker embeddings via `talker.get_speaker_embedding(name)`, mix with custom ratios (e.g., 0.75A + 0.25B), and pass the result via `OmniTalkerSpeechConfig.speaker`.

## See Also

- [C++ Qwen3-Omni chat sample](../../cpp/omni/)
- [Qwen3-Omni model documentation](https://github.com/QwenLM/Qwen3-Omni)
