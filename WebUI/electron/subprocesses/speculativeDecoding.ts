import path from 'node:path'

export type SpeculativeDecodingOptions = {
  assistantModel: string
  numAssistantTokens?: number
}

export type BackendRuntimeOptions = {
  speculative?: SpeculativeDecodingOptions
}

export function buildLlamaCppSpeculativeArgs(
  options: SpeculativeDecodingOptions | undefined,
  resolveModelPath: (modelRepoId: string) => string,
): string[] {
  if (!options?.assistantModel) return []

  const args = ['--model-draft', resolveModelPath(options.assistantModel)]
  if (options.numAssistantTokens !== undefined) {
    args.push('--draft-max', options.numAssistantTokens.toString())
  }
  return args
}

export function getOpenVinoSpeculativeConfigPath(
  modelRepoId: string,
  modelRepositoryPath: string,
): string {
  return path
    .join(modelRepositoryPath, modelRepoId.split('/').join('---'), 'config.json')
    .replace(/\\/g, '/')
}

export function toOpenVinoChatRequestParameters(
  options: SpeculativeDecodingOptions | undefined,
): Record<string, number> {
  if (!options?.numAssistantTokens) return {}
  return { num_assistant_tokens: options.numAssistantTokens }
}
