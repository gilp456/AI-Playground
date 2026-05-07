import { describe, expect, it } from 'vitest'
import { metricsEnabledFromSavedSetting } from './textInference'
import { withGemmaMtpRequestFlag } from './openAiCompatibleChat'

describe('Gemma MTP app telemetry helpers', () => {
  it('enables metrics by default when a preset has no saved preference', () => {
    expect(metricsEnabledFromSavedSetting(undefined)).toBe(true)
    expect(metricsEnabledFromSavedSetting(false)).toBe(false)
  })

  it('adds the selected MTP flag to Gemma request bodies', () => {
    const body = JSON.stringify({ model: 'google/gemma-4-E2B-it', messages: [] })

    expect(JSON.parse(withGemmaMtpRequestFlag(body, 'gemmaMTP', false))).toMatchObject({
      model: 'google/gemma-4-E2B-it',
      mtp: false,
    })
    expect(JSON.parse(withGemmaMtpRequestFlag(body, 'llamaCPP', false))).not.toHaveProperty('mtp')
  })
})
