export const MIN_ZOOM_LEVEL = -1
export const MAX_ZOOM_LEVEL = 2
export const ZOOM_LEVEL_STEP = 0.5
export const DEFAULT_ZOOM_LEVEL = 0.5

export function clampZoomLevel(level: number): number {
  return Math.min(MAX_ZOOM_LEVEL, Math.max(MIN_ZOOM_LEVEL, level))
}

export function nextZoomLevel(currentLevel: number): number {
  return clampZoomLevel(currentLevel + ZOOM_LEVEL_STEP)
}

export function previousZoomLevel(currentLevel: number): number {
  return clampZoomLevel(currentLevel - ZOOM_LEVEL_STEP)
}
