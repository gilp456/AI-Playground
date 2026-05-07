import { describe, expect, it } from 'vitest'
import {
  DEFAULT_ZOOM_LEVEL,
  MAX_ZOOM_LEVEL,
  MIN_ZOOM_LEVEL,
  nextZoomLevel,
  previousZoomLevel,
} from '../windowControls'

describe('window control zoom helpers', () => {
  it('clamps zoom in at the maximum level', () => {
    expect(nextZoomLevel(MAX_ZOOM_LEVEL)).toBe(MAX_ZOOM_LEVEL)
    expect(nextZoomLevel(MAX_ZOOM_LEVEL - 0.5)).toBe(MAX_ZOOM_LEVEL)
  })

  it('clamps zoom out at the minimum level', () => {
    expect(previousZoomLevel(MIN_ZOOM_LEVEL)).toBe(MIN_ZOOM_LEVEL)
    expect(previousZoomLevel(MIN_ZOOM_LEVEL + 0.5)).toBe(MIN_ZOOM_LEVEL)
  })

  it('uses a readable default zoom level', () => {
    expect(DEFAULT_ZOOM_LEVEL).toBeGreaterThan(0)
    expect(DEFAULT_ZOOM_LEVEL).toBeLessThanOrEqual(MAX_ZOOM_LEVEL)
  })
})
