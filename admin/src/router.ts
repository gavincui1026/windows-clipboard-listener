import { createRouter, createWebHistory } from 'vue-router'
import type { RouteRecordRaw } from 'vue-router'

const routes: RouteRecordRaw[] = [
  { 
    path: '/login', 
    component: () => import('./views/Login.vue'),
    meta: { title: '登录' }
  },
  { 
    path: '/', 
    redirect: '/dashboard' 
  },
  { 
    path: '/dashboard', 
    component: () => import('./views/Dashboard.vue'),
    meta: { title: '仪表板' }
  },
  { 
    path: '/devices', 
    component: () => import('./views/Devices.vue'),
    meta: { title: '设备管理' }
  },
  { 
    path: '/telegram', 
    component: () => import('./views/TelegramBot.vue'),
    meta: { title: 'Telegram机器人' }
  },
]

const router = createRouter({ history: createWebHistory(), routes })

router.beforeEach((to, _from, next) => {
  const token = localStorage.getItem('token') || ''
  
  // 设置页面标题
  if (to.meta.title) {
    document.title = `${to.meta.title} - Clipboard Admin`
  }
  
  // 未登录且不是登录页面，跳转到登录页
  if (to.path !== '/login' && !token) {
    return next('/login')
  }
  
  // 已登录且是登录页面，跳转到仪表板
  if (to.path === '/login' && token) {
    return next('/dashboard')
  }
  
  next()
})

export default router


