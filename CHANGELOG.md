# Changelog

All notable local development changes are documented in this file.

## 3.1.8-alpha.0-local - 2026-05-07

### Added

- Added a Gemma MTP - Transformers backend for testing Google's official Gemma MTP primary and assistant model flow through AI Playground.
- Added Gemma MTP support to the OpenAI-compatible local provider surface so compatible clients can discover and call locally hosted models.
- Added model readiness loading for Gemma MTP pairs, including primary model, assistant model, assistant-token count, selected device, and context size details.
- Added an Unload Model action for Gemma MTP chat settings to release the active primary/assistant model pair without restarting the app.
- Added a smaller official default Gemma MTP pair: `google/gemma-4-E2B-it` with `google/gemma-4-E2B-it-assistant`.
- Added clearer MTP model labels that describe the hierarchy as Primary and Assistant.

### Changed

- Replaced the indeterminate loading bar during Gemma MTP startup with concrete loading details so users can see what is being prepared.
- Switched the default official Gemma MTP preset from `google/gemma-4-E4B-it` to `google/gemma-4-E2B-it` because the E4B official Transformers weights were too large for comfortable local testing on the target system.
- Bumped the local development app version to `3.1.8-alpha.0`.

### Fixed

- Fixed Gemma MTP runtime option routing so chat requests use the selected primary and assistant metadata.
- Fixed app window controls and development zoom scaling in the local development build.
- Fixed setup packaging resource paths for portable Git resources used during backend setup.
- Fixed media path/version alignment for the local development package.

### Notes

- The official Google Gemma MTP Transformers path downloads safetensors/Transformers model weights. These are not the same footprint as Google's quantized Q4 memory examples. On the local test system, `google/gemma-4-E4B-it` occupied about 14.92 GB on disk, while the assistant occupied about 0.18 GB. Use the smaller E2B pair for official MTP testing on constrained systems, or use GGUF models when testing quantized Q4 behavior.
- The branch remains local-fork oriented. It should not be treated as an upstream Intel AI Playground release until reviewed and accepted by the upstream maintainers.

## Credits and AI Assistance Disclosure

These local feature changes were developed under human direction with AI-assisted engineering support from OpenAI Codex. Codex was used to help research implementation options, edit code and documentation, run local verification commands, and summarize the resulting changes. Human review remains responsible for validating behavior, license compatibility, and suitability before any upstream contribution.

References:

- Google AI for Developers, "Gemma models." https://ai.google.dev/gemma/docs/core
- Google AI for Developers, "Gemma multi-token prediction." https://ai.google.dev/gemma/docs/mtp/mtp
- OpenAI, "Codex." https://openai.com/codex/
