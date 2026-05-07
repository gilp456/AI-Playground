import { describe, expect, it } from 'vitest'
import { isGemmaMtpSelectionLoaded } from './textInference'

describe('isGemmaMtpSelectionLoaded', () => {
  it('returns true only when the selected primary and assistant match the loaded pair', () => {
    expect(
      isGemmaMtpSelectionLoaded({
        activeModel: 'google/gemma-4-E2B-it',
        activeAssistantModel: 'google/gemma-4-E2B-it-assistant',
        lastUsedModel: 'google/gemma-4-E2B-it',
        lastUsedAssistantModel: 'google/gemma-4-E2B-it-assistant',
      }),
    ).toBe(true)

    expect(
      isGemmaMtpSelectionLoaded({
        activeModel: 'google/gemma-4-E2B-it',
        activeAssistantModel: 'google/gemma-4-E2B-it-assistant',
        lastUsedModel: 'google/gemma-4-E2B-it',
        lastUsedAssistantModel: 'google/gemma-4-E4B-it-assistant',
      }),
    ).toBe(false)
  })

  it('returns false when no model has been loaded in the current session', () => {
    expect(
      isGemmaMtpSelectionLoaded({
        activeModel: 'google/gemma-4-E2B-it',
        activeAssistantModel: 'google/gemma-4-E2B-it-assistant',
        lastUsedModel: null,
        lastUsedAssistantModel: null,
      }),
    ).toBe(false)
  })
})
