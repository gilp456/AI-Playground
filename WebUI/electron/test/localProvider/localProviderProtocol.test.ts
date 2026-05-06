import { describe, expect, it } from 'vitest'
import {
  createChatCompletionPayload,
  createModelListResponse,
  createResponsePayload,
  toOpenAiSse,
} from '../../localProvider/localProviderProtocol'

describe('localProviderProtocol', () => {
  it('lists chat-capable models in OpenAI model-list format', () => {
    const response = createModelListResponse([
      { name: 'OpenVINO/Gemma-4-E4B-it-mtp-ov', type: 'openVINO', downloaded: true },
      { name: 'OpenVINO/bge-base-en-v1.5-fp16-ov', type: 'embedding', downloaded: true },
      { name: 'missing/model.gguf', type: 'llamaCPP', downloaded: false },
    ])

    expect(response).toEqual({
      object: 'list',
      data: [
        {
          id: 'OpenVINO/Gemma-4-E4B-it-mtp-ov',
          object: 'model',
          owned_by: 'ai-playground',
        },
      ],
    })
  })

  it('adds OpenVINO speculative defaults to chat completion requests', () => {
    const payload = createChatCompletionPayload(
      {
        model: 'OpenVINO/Gemma-4-E4B-it-mtp-ov',
        messages: [{ role: 'user', content: 'hello' }],
        stream: true,
      },
      {
        name: 'OpenVINO/Gemma-4-E4B-it-mtp-ov',
        type: 'openVINO',
        downloaded: true,
        speculative: {
          assistantModel: 'google/gemma-4-E4B-it-assistant',
          numAssistantTokens: 5,
        },
      },
    )

    expect(payload).toMatchObject({
      model: 'OpenVINO---Gemma-4-E4B-it-mtp-ov',
      messages: [{ role: 'user', content: 'hello' }],
      stream: true,
      num_assistant_tokens: 5,
    })
  })

  it('converts a basic Responses API request into a chat completion payload', () => {
    const payload = createResponsePayload(
      {
        model: 'llama-model',
        instructions: 'You are concise.',
        input: 'Say hi',
        stream: false,
      },
      {
        name: 'llama-model',
        type: 'llamaCPP',
        downloaded: true,
      },
    )

    expect(payload).toEqual({
      model: 'llama-model',
      messages: [
        { role: 'system', content: 'You are concise.' },
        { role: 'user', content: 'Say hi' },
      ],
      stream: false,
    })
  })

  it('normalizes Responses text content parts into chat-compatible text parts', () => {
    const payload = createResponsePayload(
      {
        model: 'llama-model',
        input: [
          {
            role: 'user',
            content: [{ type: 'input_text', text: 'Say hi' }],
          },
        ],
      },
      {
        name: 'llama-model',
        type: 'llamaCPP',
        downloaded: true,
      },
    )

    expect(payload).toMatchObject({
      messages: [
        {
          role: 'user',
          content: [{ type: 'text', text: 'Say hi' }],
        },
      ],
    })
  })

  it('converts chat completion stream chunks to basic Responses API SSE events', () => {
    const events = toOpenAiSse('resp_123', 'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\n')

    expect(events).toBe(
      'event: response.output_text.delta\n' +
        'data: {"type":"response.output_text.delta","response_id":"resp_123","delta":"Hi"}\n\n',
    )
  })
})
