import getPort from 'get-port'
import { afterEach, describe, expect, it, vi } from 'vitest'
import {
  createLocalProviderServer,
  type LocalProviderServer,
} from '../../localProvider/localProviderServer'
import type { BackendRuntimeOptions } from '../../subprocesses/speculativeDecoding'

describe('localProviderServer', () => {
  let server: LocalProviderServer | null = null

  afterEach(async () => {
    await server?.close()
    server = null
  })

  it('lists downloaded models and proxies chat completions through the selected backend', async () => {
    const port = await getPort()
    const ensureBackendReadiness =
      vi.fn<
        (
          llmModelName: string,
          embeddingModelName?: string,
          contextSize?: number,
          runtimeOptions?: BackendRuntimeOptions,
        ) => Promise<void>
      >()
    const fetchImpl = vi.fn<typeof fetch>().mockResolvedValue(
      new Response(JSON.stringify({ choices: [{ message: { content: 'pong' } }] }), {
        headers: { 'Content-Type': 'application/json' },
        status: 200,
      }),
    )

    server = await createLocalProviderServer({
      host: '127.0.0.1',
      port,
      fetch: fetchImpl,
      getModels: async () => [
        {
          name: 'OpenVINO/Gemma-4-E4B-it-mtp-ov',
          type: 'openVINO',
          downloaded: true,
          speculative: {
            assistantModel: 'OpenVINO/Gemma-4-E4B-it-assistant-ov',
            numAssistantTokens: 4,
          },
        },
      ],
      getService: () => ({
        get_info: () =>
          ({
            baseUrl: 'http://127.0.0.1:9999',
          }) as ApiServiceInformation,
        ensureBackendReadiness,
      }),
    })

    const models = await fetch(`${server.url}/models`).then((response) => response.json())
    expect(models).toMatchObject({
      data: [{ id: 'OpenVINO/Gemma-4-E4B-it-mtp-ov' }],
    })

    const completion = await fetch(`${server.url}/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'OpenVINO/Gemma-4-E4B-it-mtp-ov',
        messages: [{ role: 'user', content: 'ping' }],
        context_size: 4096,
      }),
    }).then((response) => response.json())

    expect(completion).toEqual({ choices: [{ message: { content: 'pong' } }] })
    expect(ensureBackendReadiness).toHaveBeenCalledWith(
      'OpenVINO/Gemma-4-E4B-it-mtp-ov',
      undefined,
      4096,
      {
        speculative: {
          assistantModel: 'OpenVINO/Gemma-4-E4B-it-assistant-ov',
          numAssistantTokens: 4,
        },
      },
    )
    expect(fetchImpl).toHaveBeenCalledWith('http://127.0.0.1:9999/v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'OpenVINO---Gemma-4-E4B-it-mtp-ov',
        messages: [{ role: 'user', content: 'ping' }],
        context_size: 4096,
        num_assistant_tokens: 4,
      }),
    })
  })

  it('streams chat completion responses without converting them to JSON', async () => {
    const port = await getPort()
    const fetchImpl = vi.fn<typeof fetch>().mockResolvedValue(
      new Response(streamFrom(['data: {"choices":[{"delta":{"content":"Hi"}}]}\n\n']), {
        headers: { 'Content-Type': 'text/event-stream' },
        status: 200,
      }),
    )

    server = await createServer(port, fetchImpl)

    const streamText = await fetch(`${server.url}/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'OpenVINO/Gemma-4-E4B-it-mtp-ov',
        messages: [{ role: 'user', content: 'ping' }],
        stream: true,
      }),
    }).then((response) => response.text())

    expect(streamText).toBe('data: {"choices":[{"delta":{"content":"Hi"}}]}\n\n')
  })

  it('converts split backend chat SSE chunks into Responses SSE events', async () => {
    const port = await getPort()
    const fetchImpl = vi.fn<typeof fetch>().mockResolvedValue(
      new Response(streamFrom(['data: {"choices":[{"delta"', ':{"content":"Hi"}}]}\n\n']), {
        headers: { 'Content-Type': 'text/event-stream' },
        status: 200,
      }),
    )

    server = await createServer(port, fetchImpl)

    const streamText = await fetch(`${server.url}/responses`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'OpenVINO/Gemma-4-E4B-it-mtp-ov',
        input: 'ping',
        stream: true,
      }),
    }).then((response) => response.text())

    expect(streamText).toContain('event: response.output_text.delta')
    expect(streamText).toContain('"delta":"Hi"')
    expect(streamText).toContain('event: response.completed')
  })
})

async function createServer(port: number, fetchImpl: typeof fetch) {
  return createLocalProviderServer({
    host: '127.0.0.1',
    port,
    fetch: fetchImpl,
    getModels: async () => [
      {
        name: 'OpenVINO/Gemma-4-E4B-it-mtp-ov',
        type: 'openVINO',
        downloaded: true,
      },
    ],
    getService: () => ({
      get_info: () =>
        ({
          baseUrl: 'http://127.0.0.1:9999',
        }) as ApiServiceInformation,
      ensureBackendReadiness: async () => {},
    }),
  })
}

function streamFrom(chunks: string[]) {
  const encoder = new TextEncoder()
  return new ReadableStream({
    start(controller) {
      for (const chunk of chunks) {
        controller.enqueue(encoder.encode(chunk))
      }
      controller.close()
    },
  })
}
