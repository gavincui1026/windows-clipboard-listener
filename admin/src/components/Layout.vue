<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useAuthStore } from '../store/auth'
import { ElMessage, ElMessageBox } from 'element-plus'
import { useDark, useToggle } from '@vueuse/core'

const route = useRoute()
const router = useRouter()
const authStore = useAuthStore()

// 侧边栏折叠状态
const isCollapse = ref(false)
const isMobile = ref(false)
const showMobileSidebar = ref(false)

// 暗色模式
const isDark = useDark()
const toggleDark = useToggle(isDark)

// 面包屑
const breadcrumbs = computed(() => {
  const matched = route.matched.filter(item => item.meta && item.meta.title)
  return matched.map(item => ({
    title: item.meta.title,
    path: item.path
  }))
})

// 菜单项
const menuItems = [
  {
    index: '/dashboard',
    title: '仪表板',
    icon: 'DataAnalysis'
  },
  {
    index: '/devices',
    title: '设备管理',
    icon: 'Monitor'
  },
  {
    index: '/telegram',
    title: 'TG机器人',
    icon: 'ChatDotSquare'
  }
]

// 检查是否移动端
const checkMobile = () => {
  isMobile.value = window.innerWidth <= 768
  if (isMobile.value) {
    isCollapse.value = true
  }
}

// 切换侧边栏
const toggleSidebar = () => {
  if (isMobile.value) {
    showMobileSidebar.value = !showMobileSidebar.value
  } else {
    isCollapse.value = !isCollapse.value
  }
}

// 退出登录
const handleLogout = async () => {
  try {
    await ElMessageBox.confirm(
      '确定要退出登录吗？',
      '退出确认',
      {
        confirmButtonText: '退出',
        cancelButtonText: '取消',
        type: 'warning'
      }
    )
    authStore.logout()
    ElMessage.success('退出成功')
    router.push('/login')
  } catch {
    // 用户取消
  }
}

onMounted(() => {
  checkMobile()
  window.addEventListener('resize', checkMobile)
})
</script>

<template>
  <div class="layout-container">
    <!-- 侧边栏 -->
    <aside 
      :class="[
        'sidebar-container',
        { 'collapse': isCollapse && !isMobile },
        { 'mobile-show': showMobileSidebar && isMobile }
      ]"
    >
      <div class="sidebar-logo">
        <Transition name="fade" mode="out-in">
          <h1 v-if="!isCollapse || isMobile">
            <Setting style="width: 20px; height: 20px; margin-right: 8px; vertical-align: middle;" />
            剪贴板管理系统
          </h1>
          <Setting v-else style="width: 24px; height: 24px; color: white;" />
        </Transition>
      </div>
      <el-scrollbar height="calc(100vh - 60px)">
        <el-menu
          :default-active="route.path"
          :collapse="isCollapse && !isMobile"
          :collapse-transition="false"
          background-color="#304156"
          text-color="#bfcbd9"
          active-text-color="#409eff"
          router
        >
          <el-menu-item 
            v-for="item in menuItems" 
            :key="item.index"
            :index="item.index"
          >
            <template #default>
              <component :is="item.icon" style="width: 20px; height: 20px;" />
              <span style="margin-left: 10px;">{{ item.title }}</span>
            </template>
          </el-menu-item>
        </el-menu>
      </el-scrollbar>
    </aside>

    <!-- 遮罩层（移动端） -->
    <div 
      v-if="isMobile && showMobileSidebar" 
      class="drawer-bg"
      @click="showMobileSidebar = false"
    />

    <!-- 主内容区 -->
    <div :class="['main-container', { 'collapse': isCollapse && !isMobile }]">
      <!-- 顶部导航栏 -->
      <header class="navbar">
        <div class="navbar-left">
          <div 
            class="hamburger" 
            @click="toggleSidebar"
            style="cursor: pointer; font-size: 20px; color: var(--text-regular);"
          >
            <Fold v-if="!isCollapse && !isMobile" />
            <Expand v-else />
          </div>
          
          <!-- 面包屑 -->
          <el-breadcrumb separator="/" class="breadcrumb-container">
            <el-breadcrumb-item 
              v-for="(item, index) in breadcrumbs" 
              :key="item.path"
              :to="index === breadcrumbs.length - 1 ? '' : { path: item.path }"
            >
              {{ item.title }}
            </el-breadcrumb-item>
          </el-breadcrumb>
        </div>

        <div class="navbar-right">
          <!-- 暗色模式切换 -->
          <div class="theme-switch" @click="toggleDark()">
            <Sunny v-if="isDark" />
            <Moon v-else />
          </div>
          
          <!-- 用户下拉菜单 -->
          <el-dropdown trigger="click">
            <div style="display: flex; align-items: center; cursor: pointer;">
              <el-avatar 
                :size="32" 
                style="background-color: var(--primary-color);"
              >
                <UserFilled />
              </el-avatar>
              <span style="margin-left: 8px; color: var(--text-regular);">管理员</span>
              <ArrowDown style="margin-left: 4px; width: 12px; height: 12px;" />
            </div>
            <template #dropdown>
              <el-dropdown-menu>
                <el-dropdown-item @click="handleLogout">
                  <SwitchButton style="margin-right: 8px;" />
                  退出登录
                </el-dropdown-item>
              </el-dropdown-menu>
            </template>
          </el-dropdown>
        </div>
      </header>

      <!-- 页面内容 -->
      <main class="app-main">
        <Transition name="slide-fade" mode="out-in">
          <router-view />
        </Transition>
      </main>
    </div>
  </div>
</template>

<style scoped>
.layout-container {
  height: 100vh;
  width: 100%;
  display: flex;
  overflow: hidden;
}

.hamburger {
  transition: all 0.3s;
}

.hamburger:hover {
  color: var(--primary-color) !important;
}

.drawer-bg {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background-color: rgba(0, 0, 0, 0.3);
  z-index: 999;
}

:deep(.el-menu) {
  border: none;
}

:deep(.el-menu-item) {
  font-size: 14px;
  height: 50px;
  line-height: 50px;
}

:deep(.el-menu-item.is-active) {
  background-color: rgba(64, 158, 255, 0.2) !important;
}
</style>
