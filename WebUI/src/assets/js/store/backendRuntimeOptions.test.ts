import { describe, expect, it } from 'vitest'
import { cloneBackendRuntimeOptionsForIpc } from './backendRuntimeOptions'

describe('cloneBackendRuntimeOptionsForIpc', () => {
  it('returns a structured-cloneable speculative options object from a proxy source', () => {
    const source = {
      speculative: new Proxy(
        {
          assistantModel: 'google/gemma-4-E4B-it-assistant',
          numAssistantTokens: 4,
        },
        {},
      ),
    }

    expect(() => structuredClone(source)).toThrow()

    const cloned = cloneBackendRuntimeOptionsForIpc(source)

    expect(structuredClone(cloned)).toEqual({
      speculative: {
        assistantModel: 'google/gemma-4-E4B-it-assistant',
        numAssistantTokens: 4,
      },
    })
  })

  it('omits empty speculative options', () => {
    expect(cloneBackendRuntimeOptionsForIpc({ speculative: undefined })).toBeUndefined()
  })
})
