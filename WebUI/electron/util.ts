import { app } from 'electron'
import path from 'node:path'

export const externalResourcesDir = () =>
  path.resolve(app.isPackaged ? process.resourcesPath : path.join(__dirname, '../../external/'))

export const getMediaDir = () => {
  const mediaFolder = app.isPackaged ? 'AI-Playground' : 'AI-Playground-Codex-Dev'
  let mediaDir: string
  if (process.env.USERPROFILE) {
    mediaDir = path.join(process.env.USERPROFILE, 'Documents', mediaFolder, 'media')
  } else if (process.env.HOME) {
    mediaDir = path.join(process.env.HOME, mediaFolder, 'media')
  } else {
    mediaDir = path.join(externalResourcesDir(), 'service', 'static', 'sd_out')
  }
  return mediaDir
}
