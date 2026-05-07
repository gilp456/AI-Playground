export function cloneBackendRuntimeOptionsForIpc(
  runtimeOptions?: BackendRuntimeOptions,
): BackendRuntimeOptions | undefined {
  const speculative = runtimeOptions?.speculative
  if (!speculative?.assistantModel) return undefined

  return {
    speculative: {
      assistantModel: speculative.assistantModel,
      numAssistantTokens: speculative.numAssistantTokens,
    },
  }
}
