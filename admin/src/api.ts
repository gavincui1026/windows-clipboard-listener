import axios from 'axios'

const api = axios.create({ baseURL: import.meta.env.VITE_API_BASE || 'http://localhost:8001' })

api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token') || ''
  if (token) {
    config.headers = config.headers || {}
    ;(config.headers as any)['Authorization'] = 'Bearer ' + token
  }
  return config
})

export const login = (username: string, password: string) => api.post('/admin/login', { username, password })
export const getStats = () => api.get('/admin/stats')
export const getDailyStats = () => api.get('/admin/stats/daily')
export const listDevices = () => api.get('/admin/devices')
export const updateNote = (deviceId: string, note: string) => api.patch(`/admin/devices/${deviceId}/note`, { note })
export const pushSet = (deviceId: string, text: string) => api.post(`/admin/devices/${deviceId}/push-set`, { set: { format: 'text/plain', text } })
export const getSettings = () => api.get('/admin/settings')
export const updateSetting = (key: string, value: string) => api.put(`/admin/settings/${key}`, { value })
export const testTelegram = () => api.post('/admin/settings/test-telegram')
export const generateSimilar = (deviceId: string) => api.post(`/admin/devices/${deviceId}/generate-similar`)
export const getGeneratedAddresses = (deviceId: string) => api.get(`/admin/devices/${deviceId}/generated-addresses`)

export default api


