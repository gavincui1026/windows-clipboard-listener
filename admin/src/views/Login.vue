<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { login } from '../api'
import { useAuthStore } from '../store/auth'
import { ElMessage } from 'element-plus'

const username = ref('admin')
const password = ref('admin')
const loading = ref(false)
const router = useRouter()
const store = useAuthStore()

// 动画状态
const showForm = ref(false)

const onSubmit = async () => {
  if (!username.value || !password.value) {
    ElMessage.warning('请输入用户名和密码')
    return
  }

  loading.value = true
  try {
    const { data } = await login(username.value, password.value)
    store.setToken(data.token)
    ElMessage.success('登录成功')
    router.push('/dashboard')
  } catch (error: any) {
    ElMessage.error(error.response?.data?.detail || '登录失败')
  } finally {
    loading.value = false
  }
}

// 处理回车键
const handleKeyup = (e: KeyboardEvent) => {
  if (e.key === 'Enter') {
    onSubmit()
  }
}

onMounted(() => {
  // 延迟显示表单，创建动画效果
  setTimeout(() => {
    showForm.value = true
  }, 100)
})
</script>

<template>
  <div class="login-container">
    <!-- 背景装饰 -->
    <div class="login-bg">
      <div class="bg-shape bg-shape-1"></div>
      <div class="bg-shape bg-shape-2"></div>
      <div class="bg-shape bg-shape-3"></div>
    </div>

    <!-- 登录表单 -->
    <Transition name="bounce" appear>
      <el-card v-if="showForm" class="login-card animate__animated animate__fadeInUp">
        <div class="login-header">
          <div class="logo-container">
            <Lock class="logo-icon" />
          </div>
          <h2 class="login-title">剪贴板管理系统</h2>
          <p class="login-subtitle">欢迎回来！请登录您的账户。</p>
        </div>

        <el-form 
          @submit.prevent="onSubmit" 
          class="login-form"
          size="large"
        >
          <el-form-item>
            <el-input 
              v-model="username" 
              placeholder="用户名"
              prefix-icon="User"
              clearable
              @keyup.enter="onSubmit"
            />
          </el-form-item>

          <el-form-item>
            <el-input 
              v-model="password" 
              type="password" 
              placeholder="密码"
              prefix-icon="Lock"
              show-password
              clearable
              @keyup.enter="onSubmit"
            />
          </el-form-item>

          <el-form-item>
            <el-button 
              type="primary" 
              :loading="loading" 
              class="login-button"
              @click="onSubmit"
            >
              <span v-if="!loading">登录</span>
              <span v-else>登录中...</span>
            </el-button>
          </el-form-item>
        </el-form>

        <div class="login-footer">
          <el-text type="info" size="small">
            默认账号：admin / admin
          </el-text>
        </div>
      </el-card>
    </Transition>

    <!-- 版权信息 -->
    <div class="copyright">
      <el-text type="info">
        © 2024 剪贴板管理系统。保留所有权利。
      </el-text>
    </div>
  </div>
</template>

<style scoped>
.login-container {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 100vh;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  position: relative;
  overflow: hidden;
}

/* 背景装饰 */
.login-bg {
  position: absolute;
  width: 100%;
  height: 100%;
  overflow: hidden;
  z-index: 0;
}

.bg-shape {
  position: absolute;
  border-radius: 50%;
  background: rgba(255, 255, 255, 0.1);
  animation: float 20s infinite ease-in-out;
}

.bg-shape-1 {
  width: 400px;
  height: 400px;
  top: -200px;
  left: -100px;
  animation-delay: 0s;
}

.bg-shape-2 {
  width: 300px;
  height: 300px;
  bottom: -150px;
  right: -100px;
  animation-delay: 5s;
}

.bg-shape-3 {
  width: 200px;
  height: 200px;
  top: 50%;
  left: 50%;
  animation-delay: 10s;
}

@keyframes float {
  0%, 100% {
    transform: translate(0, 0) rotate(0deg);
  }
  33% {
    transform: translate(30px, -30px) rotate(120deg);
  }
  66% {
    transform: translate(-20px, 20px) rotate(240deg);
  }
}

/* 登录卡片 */
.login-card {
  width: 400px;
  max-width: 90%;
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.3);
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.2);
  border-radius: 16px;
  z-index: 1;
}

:deep(.el-card__body) {
  padding: 40px 32px;
}

/* 登录头部 */
.login-header {
  text-align: center;
  margin-bottom: 32px;
}

.logo-container {
  width: 80px;
  height: 80px;
  margin: 0 auto 20px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
  animation: pulse 2s infinite;
}

.logo-icon {
  width: 40px;
  height: 40px;
  color: white;
}

@keyframes pulse {
  0% {
    box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
  }
  50% {
    box-shadow: 0 10px 40px rgba(102, 126, 234, 0.6);
  }
  100% {
    box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
  }
}

.login-title {
  font-size: 28px;
  font-weight: 700;
  color: #303133;
  margin: 0 0 8px;
}

.login-subtitle {
  font-size: 14px;
  color: #909399;
  margin: 0;
}

/* 表单样式 */
.login-form {
  margin-bottom: 24px;
}

:deep(.el-input__wrapper) {
  background-color: #f5f7fa;
  border: 1px solid transparent;
  box-shadow: none;
  transition: all 0.3s;
}

:deep(.el-input__wrapper:hover),
:deep(.el-input__wrapper.is-focus) {
  background-color: white;
  border-color: #667eea;
  box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.1);
}

.login-button {
  width: 100%;
  height: 48px;
  font-size: 16px;
  font-weight: 600;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  border: none;
  transition: all 0.3s;
  box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
}

.login-button:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
}

.login-button:active {
  transform: translateY(0);
}

/* 页脚 */
.login-footer {
  text-align: center;
  padding-top: 16px;
  border-top: 1px solid #ebeef5;
}

/* 版权信息 */
.copyright {
  position: absolute;
  bottom: 20px;
  left: 50%;
  transform: translateX(-50%);
  z-index: 1;
}

.copyright .el-text {
  color: rgba(255, 255, 255, 0.8);
}

/* 动画效果 */
.bounce-enter-active {
  animation: bounce-in 0.5s;
}

@keyframes bounce-in {
  0% {
    transform: scale(0.9);
    opacity: 0;
  }
  50% {
    transform: scale(1.05);
  }
  100% {
    transform: scale(1);
    opacity: 1;
  }
}

/* 响应式设计 */
@media (max-width: 480px) {
  .login-card {
    width: 100%;
    margin: 20px;
  }
  
  :deep(.el-card__body) {
    padding: 32px 24px;
  }
  
  .login-title {
    font-size: 24px;
  }
}
</style>