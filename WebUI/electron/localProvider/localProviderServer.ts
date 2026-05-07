import http, { IncomingMessage, ServerResponse } from 'node:http'
import { randomUUID } from 'node:crypto'
import {
  createChatCompletionPayload,
  createModelListResponse,
  createResponsePayload,
  toOpenAiSse,
  type ChatCompletionRequest,
  type LocalProviderModel,
  type ResponsesRequest,
} from './localProviderProtocol'
import type { BackendRuntimeOptions } from '../subprocesses/speculativeDecoding.ts'

type LocalProviderService = {
  get_info(): ApiServiceInformation
  ensureBackendReadiness(
    llmModelName: string,
    embeddingModelName?: string,
    contextSize?: number,
    runtimeOptions?: BackendRuntimeOptions,
  ): Promise<void>
}

type LocalProviderServerOptions = {
  host: string
  port: number
  getModels(): Promise<LocalProviderModel[]>
  getService(serviceName: BackendServiceName): LocalProviderService | undefined
  fetch?: typeof fetch
}

export type LocalProviderServer = {
  url: string
  close(): Promise<void>
}

export async function createLocalProviderServer(
  options: LocalProviderServerOptions,
): Promise<LocalProviderServer> {
  const fetchImpl = options.fetch ?? fetch
  const server = http.createServer(async (request, response) => {
    try {
      applyCors(request, response)

      if (request.method === 'OPTIONS') {
        response.writeHead(204)
        response.end()
        return
      }

      const url = new URL(request.url ?? '/', `http://${options.host}:${options.port}`)
      if (request.method === 'GET' && url.pathname === '/v1/models') {
        writeJson(response, 200, createModelListResponse(await options.getModels()))
        return
      }

      if (request.method === 'POST' && url.pathname === '/v1/chat/completions') {
        await handleChatCompletions(request, response, options, fetchImpl)
        return
      }

      if (request.method === 'POST' && url.pathname === '/v1/responses') {
        await handleResponses(request, response, options, fetchImpl)
        return
      }

      writeJson(response, 404, { error: { message: 'Not found' } })
    } catch (error) {
      writeJson(response, 500, {
        error: { message: error instanceof Error ? error.message : String(error) },
      })
    }
  })

  await new Promise<void>((resolve, reject) => {
    server.once('error', reject)
    server.listen(options.port, options.host, () => {
      server.off('error', reject)
      resolve()
    })
  })

  return {
    url: `http://${options.host}:${options.port}/v1`,
    close: () =>
      new Promise<void>((resolve, reject) => {
        server.close((error) => (error ? reject(error) : resolve()))
      }),
  }
}

async function handleChatCompletions(
  request: IncomingMessage,
  response: ServerResponse,
  options: LocalProviderServerOptions,
  fetchImpl: typeof fetch,
) {
  const body = await readJson(request)
  if (!isChatCompletionRequest(body)) {
    writeJson(response, 400, { error: { message: 'Expected model and messages' } })
    return
  }

  const model = await resolveModel(body.model, options)
  if (!model) {
    writeJson(response, 404, { error: { message: `Unknown model: ${body.model}` } })
    return
  }

  const service = options.getService(serviceNameForModel(model))
  if (!service) {
    writeJson(response, 503, { error: { message: `Backend not available for ${model.type}` } })
    return
  }

  await service.ensureBackendReadiness(model.name, undefined, optionalNumber(body.context_size), {
    speculative: model.speculative,
  })

  const backendResponse = await fetchImpl(`${service.get_info().baseUrl}/v1/chat/completions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(createChatCompletionPayload(body, model)),
  })

  await pipeBackendResponse(response, backendResponse, body.stream === true)
}

async function handleResponses(
  request: IncomingMessage,
  response: ServerResponse,
  options: LocalProviderServerOptions,
  fetchImpl: typeof fetch,
) {
  const body = await readJson(request)
  if (!isResponsesRequest(body)) {
    writeJson(response, 400, { error: { message: 'Expected model and input' } })
    return
  }

  const model = await resolveModel(body.model, options)
  if (!model) {
    writeJson(response, 404, { error: { message: `Unknown model: ${body.model}` } })
    return
  }

  const service = options.getService(serviceNameForModel(model))
  if (!service) {
    writeJson(response, 503, { error: { message: `Backend not available for ${model.type}` } })
    return
  }

  await service.ensureBackendReadiness(model.name, undefined, optionalNumber(body.context_size), {
    speculative: model.speculative,
  })

  const chatPayload = createResponsePayload(body, model)
  const backendResponse = await fetchImpl(`${service.get_info().baseUrl}/v1/chat/completions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(chatPayload),
  })

  if (body.stream) {
    await pipeResponsesStream(response, backendResponse)
    return
  }

  const chatCompletion = (await backendResponse.json()) as {
    choices?: Array<{ message?: { content?: string } }>
  }
  const text = chatCompletion.choices?.[0]?.message?.content ?? ''
  writeJson(response, backendResponse.status, {
    id: `resp_${randomUUID()}`,
    object: 'response',
    created_at: Math.floor(Date.now() / 1000),
    model: body.model,
    status: 'completed',
    output: [
      {
        type: 'message',
        role: 'assistant',
        content: [{ type: 'output_text', text }],
      },
    ],
    output_text: text,
  })
}

async function resolveModel(
  requestedModel: unknown,
  options: LocalProviderServerOptions,
): Promise<LocalProviderModel | undefined> {
  if (typeof requestedModel !== 'string') return undefined
  return (await options.getModels()).find(
    (model) => model.name === requestedModel && model.downloaded,
  )
}

function isChatCompletionRequest(body: Record<string, unknown>): body is ChatCompletionRequest {
  return typeof body.model === 'string' && Array.isArray(body.messages)
}

function isResponsesRequest(body: Record<string, unknown>): body is ResponsesRequest {
  return (
    typeof body.model === 'string' && (typeof body.input === 'string' || Array.isArray(body.input))
  )
}

function optionalNumber(value: unknown): number | undefined {
  return typeof value === 'number' ? value : undefined
}

function serviceNameForModel(model: LocalProviderModel): BackendServiceName {
  if (model.type === 'openVINO') return 'openvino-backend'
  if (model.type === 'gemmaMTP') return 'ai-backend'
  return 'llamacpp-backend'
}

async function pipeBackendResponse(
  response: ServerResponse,
  backendResponse: Response,
  shouldStream = false,
) {
  response.writeHead(backendResponse.status, {
    'Content-Type': backendResponse.headers.get('content-type') ?? 'application/json',
  })
  if (shouldStream || backendResponse.headers.get('content-type')?.includes('text/event-stream')) {
    await streamResponseBody(response, backendResponse)
    return
  }

  response.end(Buffer.from(await backendResponse.arrayBuffer()))
}

async function pipeResponsesStream(response: ServerResponse, backendResponse: Response) {
  const responseId = `resp_${randomUUID()}`
  response.writeHead(backendResponse.status, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    Connection: 'keep-alive',
  })
  response.write(
    `event: response.created\ndata: ${JSON.stringify({ type: 'response.created', response_id: responseId })}\n\n`,
  )

  if (!backendResponse.body) {
    response.end()
    return
  }

  const reader = backendResponse.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const events = buffer.split('\n\n')
    buffer = events.pop() ?? ''
    response.write(toOpenAiSse(responseId, events.join('\n\n')))
  }
  buffer += decoder.decode()
  if (buffer.trim()) {
    response.write(toOpenAiSse(responseId, buffer))
  }
  response.write(
    `event: response.completed\ndata: ${JSON.stringify({ type: 'response.completed', response_id: responseId })}\n\n`,
  )
  response.write('data: [DONE]\n\n')
  response.end()
}

async function streamResponseBody(response: ServerResponse, backendResponse: Response) {
  if (!backendResponse.body) {
    response.end()
    return
  }

  const reader = backendResponse.body.getReader()
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    response.write(Buffer.from(value))
  }
  response.end()
}

async function readJson(request: IncomingMessage): Promise<Record<string, unknown>> {
  const chunks: Buffer[] = []
  for await (const chunk of request) {
    chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk))
  }
  return JSON.parse(Buffer.concat(chunks).toString('utf8') || '{}')
}

function writeJson(response: ServerResponse, status: number, body: unknown) {
  response.writeHead(status, { 'Content-Type': 'application/json' })
  response.end(JSON.stringify(body))
}

function applyCors(request: IncomingMessage, response: ServerResponse) {
  const origin = request.headers.origin
  const allowedOrigin =
    origin && /^http:\/\/(localhost|127\.0\.0\.1)(:\d+)?$/.test(origin)
      ? origin
      : 'http://localhost'
  response.setHeader('Access-Control-Allow-Origin', allowedOrigin)
  response.setHeader('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
  response.setHeader('Access-Control-Allow-Headers', 'content-type,authorization')
}
