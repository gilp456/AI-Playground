import { afterEach, describe, expect, it, vi } from 'vitest'

vi.mock('electron', () => ({
  app: {
    isPackaged: false,
  },
  BrowserWindow: vi.fn(),
}))

vi.mock('../../subprocesses/uvBasedBackends/uv.ts', () => ({
  aipgBaseDir: 'C:/aipg',
  checkBackend: vi.fn(),
  installBackend: vi.fn(),
}))

import { AiBackendService } from '../../subprocesses/aiBackendService'

describe('AiBackendService Gemma MTP unloading', () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('posts to the unload endpoint and clears the cached MTP model pair', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response('{}', { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)
    const service = Object.create(AiBackendService.prototype)
    Object.assign(service, {
      baseUrl: 'http://127.0.0.1:59000',
      currentStatus: 'running',
      currentMtpModel: 'google/gemma-4-E2B-it',
      currentMtpAssistantModel: 'google/gemma-4-E2B-it-assistant',
      currentMtpDevice: 'xpu:0',
    })

    const result = await service.unloadGemmaMtp()

    expect(result).toEqual({ success: true })
    expect(fetchMock).toHaveBeenCalledWith('http://127.0.0.1:59000/api/gemmaMtp/unload', {
      method: 'POST',
    })
    expect(service.currentMtpModel).toBeNull()
    expect(service.currentMtpAssistantModel).toBeNull()
    expect(service.currentMtpDevice).toBeNull()
  })

  it('clears cached MTP state without a backend request when service is stopped', async () => {
    const fetchMock = vi.fn()
    vi.stubGlobal('fetch', fetchMock)
    const service = Object.create(AiBackendService.prototype)
    Object.assign(service, {
      currentStatus: 'notYetStarted',
      currentMtpModel: 'google/gemma-4-E2B-it',
      currentMtpAssistantModel: 'google/gemma-4-E2B-it-assistant',
      currentMtpDevice: 'xpu:0',
    })

    const result = await service.unloadGemmaMtp()

    expect(result).toEqual({ success: true })
    expect(fetchMock).not.toHaveBeenCalled()
    expect(service.currentMtpModel).toBeNull()
    expect(service.currentMtpAssistantModel).toBeNull()
    expect(service.currentMtpDevice).toBeNull()
  })
})
