import { describe, expect, it } from 'vitest'
import {
  buildLlamaCppSpeculativeArgs,
  getOpenVinoSpeculativeConfigPath,
} from '../../subprocesses/speculativeDecoding'

describe('speculativeDecoding', () => {
  it('builds llama.cpp draft model args from assistant metadata', () => {
    const args = buildLlamaCppSpeculativeArgs(
      {
        assistantModel: 'google/gemma-4-E4B-it-assistant-GGUF/gemma-4-E4B-it-assistant-Q8_0.gguf',
        numAssistantTokens: 5,
      },
      (repoId) => `C:/models/${repoId.split('/').join('---')}`,
    )

    expect(args).toEqual([
      '--model-draft',
      'C:/models/google---gemma-4-E4B-it-assistant-GGUF---gemma-4-E4B-it-assistant-Q8_0.gguf',
      '--draft-max',
      '5',
    ])
  })

  it('omits llama.cpp draft token count when it is not configured', () => {
    const args = buildLlamaCppSpeculativeArgs(
      {
        assistantModel: 'draft/repo/model.gguf',
      },
      (repoId) => repoId,
    )

    expect(args).toEqual(['--model-draft', 'draft/repo/model.gguf'])
  })

  it('uses a model-local OVMS config for speculative OpenVINO models', () => {
    const configPath = getOpenVinoSpeculativeConfigPath(
      'OpenVINO/Gemma-4-E4B-it-mtp-ov',
      'C:/aipg/models/LLM/openvino',
    )

    expect(configPath).toBe(
      'C:/aipg/models/LLM/openvino/OpenVINO---Gemma-4-E4B-it-mtp-ov/config.json',
    )
  })
})
