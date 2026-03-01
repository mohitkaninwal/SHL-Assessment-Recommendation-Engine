import { defineConfig } from 'vite'

export default defineConfig({
  server: {
    host: '0.0.0.0',
    port: Number(process.env.FRONTEND_PORT || 8501)
  }
})
