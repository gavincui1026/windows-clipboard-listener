import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// https://vite.dev/config/
export default defineConfig({
  plugins: [vue()],
  server: {
    // 允许所有域名访问（开发环境）
    host: true,
    // 或者指定具体域名
    // allowedHosts: ['clickboardlsn.top', '.clickboardlsn.top', 'localhost']
  }
})
