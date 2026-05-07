# Gemma MTP and Local Provider Notes

This document describes the local development changes that add Gemma MTP testing support to AI Playground and expose eligible local chat models through the OpenAI-compatible provider surface.

## Purpose

The goal is to evaluate Google's Gemma multi-token prediction (MTP) flow on Intel hardware from inside AI Playground, without bypassing the app's normal backend and model-selection flow. The implementation keeps the user-facing model hierarchy explicit:

- Primary: the main chat model that produces the final response.
- Assistant: the smaller MTP model used to propose additional tokens for acceleration.

## Supported Paths

### Official Gemma MTP - Transformers

The `gemmaMTP` backend loads official Google Gemma primary and assistant model pairs with Hugging Face Transformers. It runs through the AI Playground `ai-backend` service and selects Intel XPU when available.

The current default pair is:

- Primary: `google/gemma-4-E2B-it`
- Assistant: `google/gemma-4-E2B-it-assistant`

The larger `google/gemma-4-E4B-it` pair remains listed, but it is no longer the default because it was too large for the target local test machine.

### GGUF Quantized Models

AI Playground still supports GGUF models through the Llama.cpp backend. The GGUF route is the appropriate path when testing quantized Q4-style memory footprints. Google's published Q4 memory examples should not be compared directly to the official Transformers/safetensors downloads used by the `gemmaMTP` backend.

## Why the Default Changed to E2B

The local system downloaded these official Transformers model folders:

| Model folder | Local size |
| --- | ---: |
| `google---gemma-4-E4B-it` | about 14.92 GB |
| `google---gemma-4-E4B-it-assistant` | about 0.18 GB |

The observed size explains why the original E4B official pair did not behave like a roughly 5 GB Q4 model. The smaller E2B primary and assistant pair is more appropriate for validating the official MTP code path on this system.

## User Workflow

1. Open Chat Settings.
2. Select the `Gemma MTP - Transformers` backend.
3. Select `google/gemma-4-E2B-it` as the Primary model.
4. Confirm the Assistant model is shown as `google/gemma-4-E2B-it-assistant`.
5. Use Load Model to pre-load the selected primary/assistant pair before the first chat prompt, or send a prompt and allow AI Playground to load it on demand.
6. During model preparation, review the loading details for backend, primary model, assistant model, assistant-token count, device, and context size.
7. After the selected pair is loaded, the same control changes to Unload Model.
8. Use Unload Model in Chat Settings to release the loaded Gemma MTP pair without restarting AI Playground.

The Load Model / Unload Model button reflects the selected Gemma MTP primary model, assistant model, and context settings. If the user changes to a different pair or context, the button returns to Load Model because the selected configuration is no longer prepared.

## Local Provider Behavior

The local provider model list includes chat-capable Llama.cpp, OpenVINO, and Gemma MTP models. Official Gemma MTP model IDs are preserved so clients can request the same model names shown in AI Playground.

This provider is intended to behave similarly to local model servers such as Ollama or LM Studio for compatible OpenAI-style clients, while still routing inference through AI Playground's managed backends.

## Verification Performed

The local feature branch was verified with:

- `python -m unittest service.tests.test_gemma_mtp_runtime`
- `npm run test`
- `npm run type-check`
- `npm run lint:ci`
- JSON validation for modified model and preset files
- Git diff whitespace checks
- Secret-pattern scans for Hugging Face token formats

## Credits and AI Assistance Disclosure

These local feature changes were developed under human direction with AI-assisted engineering support from OpenAI Codex. Codex was used to help research implementation options, edit code and documentation, run local verification commands, and summarize the resulting changes. Human review remains responsible for validating behavior, license compatibility, and suitability before any upstream contribution.

References:

- Google AI for Developers, "Gemma models." https://ai.google.dev/gemma/docs/core
- Google AI for Developers, "Gemma multi-token prediction." https://ai.google.dev/gemma/docs/mtp/mtp
- OpenAI, "Codex." https://openai.com/codex/
