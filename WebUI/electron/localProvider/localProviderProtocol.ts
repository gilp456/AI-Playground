export type LocalProviderModel = {
  name: string
  type: string
  downloaded?: boolean
  speculative?: {
    assistantModel: string
    numAssistantTokens?: number
  }
}

export type ChatMessage = {
  role: string
  content: unknown
}

export type ChatCompletionRequest = {
  model: string
  messages: ChatMessage[]
  stream?: boolean
  [key: string]: unknown
}

export type ResponsesRequest = {
  model: string
  instructions?: string
  input: string | Array<{ role?: string; content?: unknown }>
  stream?: boolean
  [key: string]: unknown
}

export function createModelListResponse(models: LocalProviderModel[]) {
  return {
    object: 'list',
    data: models
      .filter(
        (model) =>
          model.downloaded &&
          (model.type === 'llamaCPP' || model.type === 'openVINO' || model.type === 'gemmaMTP'),
      )
      .map((model) => ({
        id: model.name,
        object: 'model',
        owned_by: 'ai-playground',
      })),
  }
}

export function normalizeBackendModelName(model: LocalProviderModel): string {
  return model.type === 'openVINO' ? model.name.split('/').join('---') : model.name
}

export function createChatCompletionPayload(
  request: ChatCompletionRequest,
  model: LocalProviderModel,
): Record<string, unknown> {
  return {
    ...request,
    model: normalizeBackendModelName(model),
    ...speculativeRequestParams(model),
  }
}

export function createResponsePayload(
  request: ResponsesRequest,
  model: LocalProviderModel,
): Record<string, unknown> {
  return {
    model: normalizeBackendModelName(model),
    messages: responseInputToMessages(request),
    stream: request.stream ?? false,
    ...copySamplingParams(request),
    ...speculativeRequestParams(model),
  }
}

function responseInputToMessages(request: ResponsesRequest): ChatMessage[] {
  const messages: ChatMessage[] = []
  if (request.instructions) {
    messages.push({ role: 'system', content: request.instructions })
  }

  if (typeof request.input === 'string') {
    messages.push({ role: 'user', content: request.input })
    return messages
  }

  for (const item of request.input) {
    const role =
      item.role === 'assistant' || item.role === 'system' || item.role === 'tool'
        ? item.role
        : 'user'
    messages.push({ role, content: normalizeResponseContent(item.content) })
  }
  return messages
}

function normalizeResponseContent(content: unknown): unknown {
  if (!Array.isArray(content)) return content ?? ''
  return content.map((part) => {
    if (!part || typeof part !== 'object') return part

    const responsePart = part as { type?: unknown; text?: unknown; [key: string]: unknown }
    if (responsePart.type === 'input_text' || responsePart.type === 'output_text') {
      return {
        ...responsePart,
        type: 'text',
      }
    }
    return responsePart
  })
}

function copySamplingParams(request: ResponsesRequest): Record<string, unknown> {
  const copied: Record<string, unknown> = {}
  for (const key of ['temperature', 'top_p', 'max_output_tokens', 'max_tokens']) {
    if (request[key] !== undefined) {
      copied[key === 'max_output_tokens' ? 'max_tokens' : key] = request[key]
    }
  }
  return copied
}

function speculativeRequestParams(model: LocalProviderModel): Record<string, number> {
  const numAssistantTokens = model.speculative?.numAssistantTokens
  return model.type === 'openVINO' && numAssistantTokens
    ? { num_assistant_tokens: numAssistantTokens }
    : {}
}

export function toOpenAiSse(responseId: string, chunk: string): string {
  return chunk
    .split('\n\n')
    .map((event) => event.trim())
    .filter((event) => event.startsWith('data: ') && event !== 'data: [DONE]')
    .map((event) => {
      const raw = event.slice('data: '.length)
      const parsed = JSON.parse(raw) as { choices?: Array<{ delta?: { content?: string } }> }
      const delta = parsed.choices?.[0]?.delta?.content
      if (!delta) return ''
      return (
        'event: response.output_text.delta\n' +
        `data: ${JSON.stringify({ type: 'response.output_text.delta', response_id: responseId, delta })}\n\n`
      )
    })
    .join('')
}
