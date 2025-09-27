<script setup lang="ts">
import { onMounted, onUnmounted, ref, computed } from 'vue'
import { getStats, getDailyStats } from '../api'
import { ElMessage } from 'element-plus'

const stats = ref<{ total: number; online: number } | null>(null)
const dailyStats = ref<{ dates: string[]; values: number[] } | null>(null)
const loading = ref(false)
const chartLoading = ref(false)

// 计算离线设备数
const offlineCount = computed(() => {
  if (!stats.value) return 0
  return stats.value.total - stats.value.online
})

// 计算在线率
const onlineRate = computed(() => {
  if (!stats.value || stats.value.total === 0) return 0
  return Math.round((stats.value.online / stats.value.total) * 100)
})

// 计算图表最大值
const maxValue = computed(() => {
  if (!dailyStats.value) return 10
  return Math.max(...dailyStats.value.values, 10)
})

// 计算折线图路径
const chartPath = computed(() => {
  if (!dailyStats.value || dailyStats.value.values.length === 0) return ''
  
  const width = 600
  const height = 250
  const padding = 40
  const chartWidth = width - padding * 2
  const chartHeight = height - padding * 2
  
  const points = dailyStats.value.values.map((value, index) => {
    const x = padding + (index / (dailyStats.value!.values.length - 1)) * chartWidth
    const y = padding + (1 - value / maxValue.value) * chartHeight
    return `${x},${y}`
  })
  
  return `M ${points.join(' L ')}`
})

// 自动刷新定时器
const refreshTimer = ref<number | null>(null)

const load = async (silent = false) => {
  if (!silent) {
    loading.value = true
  }
  try {
    const { data } = await getStats()
    stats.value = data
  } catch (error) {
    if (!silent) {
      ElMessage.error('加载统计数据失败')
    }
  } finally {
    loading.value = false
  }
}

const loadDailyStats = async () => {
  chartLoading.value = true
  try {
    const { data } = await getDailyStats()
    dailyStats.value = data
  } catch (error) {
    ElMessage.error('加载每日统计失败')
  } finally {
    chartLoading.value = false
  }
}

// 图表数据
const chartData = computed(() => {
  if (!stats.value) return []
  return [
    { name: 'Online', value: stats.value.online },
    { name: 'Offline', value: offlineCount.value }
  ]
})

onMounted(() => {
  load()
  loadDailyStats()
  // 每1.5秒静默刷新一次数据
  refreshTimer.value = window.setInterval(() => load(true), 1500)
  // 每5分钟刷新一次每日统计
  window.setInterval(() => loadDailyStats(), 300000)
})

onUnmounted(() => {
  if (refreshTimer.value) {
    clearInterval(refreshTimer.value)
  }
})

// 时间问候
const greeting = computed(() => {
  const hour = new Date().getHours()
  if (hour < 12) return '早上好'
  if (hour < 18) return '下午好'
  return '晚上好'
})
</script>

<template>
  <div class="dashboard-container">
    <!-- 欢迎标题 -->
    <div class="dashboard-header animate__animated animate__fadeInDown">
      <h1 class="dashboard-title">{{ greeting }}，管理员</h1>
      <p class="dashboard-subtitle">以下是您的剪贴板设备今日概况。</p>
    </div>

    <!-- 统计卡片 -->
    <el-row :gutter="20" class="stat-cards">
      <!-- 总设备数 -->
      <el-col :xs="24" :sm="12" :lg="8">
        <div class="stat-card stat-card-primary animate__animated animate__fadeInLeft">
          <div class="stat-icon">
            <Monitor />
          </div>
          <div class="stat-content">
            <div class="stat-value" v-loading="loading">
              {{ stats?.total ?? '-' }}
            </div>
            <div class="stat-label">总设备数</div>
          </div>
          <div class="stat-bg">
            <DataAnalysis />
          </div>
        </div>
      </el-col>

      <!-- 在线设备 -->
      <el-col :xs="24" :sm="12" :lg="8">
        <div class="stat-card stat-card-success animate__animated animate__fadeInUp">
          <div class="stat-icon">
            <Connection />
          </div>
          <div class="stat-content">
            <div class="stat-value" v-loading="loading">
              {{ stats?.online ?? '-' }}
            </div>
            <div class="stat-label">在线设备</div>
          </div>
          <div class="stat-bg">
            <TrendCharts />
          </div>
        </div>
      </el-col>

      <!-- 在线率 -->
      <el-col :xs="24" :sm="24" :lg="8">
        <div class="stat-card stat-card-info animate__animated animate__fadeInRight">
          <div class="stat-icon">
            <PieChart />
          </div>
          <div class="stat-content">
            <div class="stat-value" v-loading="loading">
              {{ onlineRate }}<span style="font-size: 0.7em; margin-left: 2px;">%</span>
            </div>
            <div class="stat-label">在线率</div>
          </div>
          <div class="stat-bg">
            <DataLine />
          </div>
        </div>
      </el-col>
    </el-row>

    <!-- 详细信息卡片 -->
    <el-row :gutter="20" style="margin-top: 20px;">
      <!-- 设备状态分布 -->
      <el-col :xs="24" :lg="12">
        <el-card class="detail-card animate__animated animate__fadeInLeft animate__delay-1s">
          <template #header>
            <div class="card-header">
              <span>设备状态分布</span>
              <el-button text @click="load()">
                <Refresh />
                刷新
              </el-button>
            </div>
          </template>
          <div v-loading="loading" style="min-height: 320px;">
            <div v-if="stats" class="chart-container">
              <!-- 简单的环形图实现 -->
              <div class="donut-chart">
                <div class="donut-hole">
                  <div class="donut-value">{{ stats.total }}</div>
                  <div class="donut-label">总计</div>
                </div>
                <svg width="200" height="200" viewBox="0 0 200 200">
                  <circle
                    cx="100"
                    cy="100"
                    r="80"
                    fill="none"
                    stroke="#e4e7ed"
                    stroke-width="20"
                  />
                  <circle
                    cx="100"
                    cy="100"
                    r="80"
                    fill="none"
                    stroke="#67c23a"
                    stroke-width="20"
                    :stroke-dasharray="`${(stats.online / stats.total) * 502.4} 502.4`"
                    stroke-dashoffset="125.6"
                    transform="rotate(-90 100 100)"
                    style="transition: stroke-dasharray 0.6s ease;"
                  />
                </svg>
              </div>
              <div class="legend">
                <div class="legend-item">
                  <span class="legend-dot" style="background: #67c23a;"></span>
                  <span>在线 ({{ stats.online }})</span>
                </div>
                <div class="legend-item">
                  <span class="legend-dot" style="background: #e4e7ed;"></span>
                  <span>离线 ({{ offlineCount }})</span>
                </div>
              </div>
            </div>
            <el-empty v-else description="暂无数据" />
          </div>
        </el-card>
      </el-col>

      <!-- 每日活跃设备统计 -->
      <el-col :xs="24" :lg="12">
        <el-card class="detail-card animate__animated animate__fadeInRight animate__delay-1s">
          <template #header>
            <div class="card-header">
              <span>最近7天活跃设备数</span>
              <el-button text @click="loadDailyStats">
                <Refresh />
                刷新
              </el-button>
            </div>
          </template>
          <div v-loading="chartLoading" style="min-height: 320px; padding: 20px 0;">
            <div v-if="dailyStats" class="line-chart-container">
              <!-- SVG 折线图 -->
              <svg width="100%" height="250" viewBox="0 0 600 250" preserveAspectRatio="xMidYMid meet">
                <!-- 网格线 -->
                <g class="grid">
                  <line v-for="i in 5" :key="'h-' + i" 
                    :x1="40" 
                    :y1="40 + (i - 1) * 42.5" 
                    :x2="560" 
                    :y2="40 + (i - 1) * 42.5" 
                    stroke="#e4e7ed" 
                    stroke-dasharray="2,2"
                  />
                </g>
                
                <!-- Y轴标签 -->
                <g class="y-labels">
                  <text v-for="i in 5" :key="'y-' + i"
                    :x="30"
                    :y="45 + (i - 1) * 42.5"
                    text-anchor="end"
                    fill="#909399"
                    font-size="12"
                  >
                    {{ Math.round(maxValue * (1 - (i - 1) * 0.25)) }}
                  </text>
                </g>
                
                <!-- X轴标签 -->
                <g class="x-labels">
                  <text v-for="(date, index) in dailyStats.dates" :key="'x-' + index"
                    :x="40 + index * 520 / 6"
                    y="235"
                    text-anchor="middle"
                    fill="#909399"
                    font-size="12"
                  >
                    {{ date }}
                  </text>
                </g>
                
                <!-- 数据线 -->
                <path
                  :d="chartPath"
                  fill="none"
                  stroke="#409eff"
                  stroke-width="2"
                  class="chart-line"
                />
                
                <!-- 数据点 -->
                <g class="data-points">
                  <circle v-for="(value, index) in dailyStats.values" :key="'point-' + index"
                    :cx="40 + index * 520 / 6"
                    :cy="40 + (1 - value / maxValue) * 170"
                    r="4"
                    fill="#409eff"
                    class="data-point"
                  >
                    <title>{{ dailyStats.dates[index] }}: {{ value }} 设备</title>
                  </circle>
                </g>
              </svg>
              
              <!-- 统计信息 -->
              <div class="chart-stats">
                <div class="stat-item">
                  <span class="stat-label">平均活跃数：</span>
                  <span class="stat-value">{{ Math.round(dailyStats.values.reduce((a, b) => a + b, 0) / dailyStats.values.length) }}</span>
                </div>
                <div class="stat-item">
                  <span class="stat-label">最高活跃数：</span>
                  <span class="stat-value">{{ Math.max(...dailyStats.values) }}</span>
                </div>
                <div class="stat-item">
                  <span class="stat-label">今日活跃数：</span>
                  <span class="stat-value">{{ dailyStats.values[dailyStats.values.length - 1] }}</span>
                </div>
              </div>
            </div>
            <el-empty v-else description="暂无数据" />
          </div>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<style scoped>
.dashboard-container {
  max-width: 1400px;
  margin: 0 auto;
  padding: 0 24px;
  width: 100%;
  box-sizing: border-box;
}

/* 头部样式 */
.dashboard-header {
  margin-bottom: 30px;
}

.dashboard-title {
  font-size: 32px;
  font-weight: 700;
  color: var(--text-primary);
  margin: 0 0 8px;
}

.dashboard-subtitle {
  font-size: 16px;
  color: var(--text-secondary);
  margin: 0;
}

/* 统计卡片 */
.stat-cards {
  margin-bottom: 20px;
}

.stat-card {
  min-height: 140px;
  padding: 28px 32px;
  border-radius: 16px;
  color: white;
  position: relative;
  overflow: hidden;
  cursor: pointer;
  transition: all 0.3s;
  margin-bottom: 20px;
  display: flex;
  align-items: center;
  backdrop-filter: blur(10px);
}

.stat-card:hover {
  transform: translateY(-5px);
}

.stat-card-primary {
  background: linear-gradient(135deg, #5b63d3 0%, #8b5cd6 100%);
  box-shadow: 0 10px 30px rgba(91, 99, 211, 0.25);
}

.stat-card-primary:hover {
  box-shadow: 0 15px 40px rgba(91, 99, 211, 0.35);
}

.stat-card-success {
  background: linear-gradient(135deg, #4caf50 0%, #81c784 100%);
  box-shadow: 0 10px 30px rgba(76, 175, 80, 0.25);
}

.stat-card-success:hover {
  box-shadow: 0 15px 40px rgba(76, 175, 80, 0.35);
}

.stat-card-info {
  background: linear-gradient(135deg, #2196f3 0%, #64b5f6 100%);
  box-shadow: 0 10px 30px rgba(33, 150, 243, 0.25);
}

.stat-card-info:hover {
  box-shadow: 0 15px 40px rgba(33, 150, 243, 0.35);
}

.stat-icon {
  position: absolute;
  top: 50%;
  right: 32px;
  transform: translateY(-50%);
  font-size: 64px;
  opacity: 0.15;
}

.stat-content {
  position: relative;
  z-index: 1;
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  justify-content: center;
}

.stat-card .stat-value {
  font-size: 52px;
  font-weight: 700;
  line-height: 1;
  margin-bottom: 12px;
  text-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
  display: block;
}

.stat-card .stat-label {
  font-size: 14px;
  font-weight: 600;
  opacity: 1;
  text-transform: uppercase;
  letter-spacing: 1px;
  text-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
}

.stat-bg {
  display: none;
}

/* 详细卡片 */
.detail-card {
  height: 100%;
  transition: all 0.3s;
}

.detail-card:hover {
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-weight: 600;
  color: var(--text-primary);
}

/* 图表样式 */
.chart-container {
  display: flex;
  align-items: center;
  justify-content: space-around;
  height: 100%;
}

.donut-chart {
  position: relative;
  width: 200px;
  height: 200px;
}

.donut-hole {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  text-align: center;
}

.donut-value {
  font-size: 36px;
  font-weight: 700;
  color: var(--text-primary);
}

.donut-label {
  font-size: 14px;
  color: var(--text-secondary);
}

.legend {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 14px;
  color: var(--text-regular);
}

.legend-dot {
  width: 12px;
  height: 12px;
  border-radius: 50%;
}

/* 折线图样式 */
.line-chart-container {
  width: 100%;
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.chart-line {
  animation: drawLine 1s ease-out;
}

@keyframes drawLine {
  from {
    stroke-dasharray: 1000;
    stroke-dashoffset: 1000;
  }
  to {
    stroke-dasharray: 1000;
    stroke-dashoffset: 0;
  }
}

.data-point {
  transition: all 0.3s;
  cursor: pointer;
}

.data-point:hover {
  r: 6;
  filter: drop-shadow(0 0 5px rgba(64, 158, 255, 0.5));
}

/* 统计信息 */
.chart-stats {
  display: flex;
  justify-content: space-around;
  padding: 20px;
  background-color: var(--bg-page);
  border-radius: 8px;
  margin: 0 20px;
}

.stat-item {
  text-align: center;
}

.chart-stats .stat-label {
  font-size: 14px;
  color: var(--text-secondary);
  margin-right: 8px;
}

.chart-stats .stat-value {
  font-size: 20px;
  font-weight: 600;
  color: var(--primary-color);
}

/* 响应式设计 */
@media (max-width: 1440px) {
  .dashboard-container {
    max-width: 1200px;
  }
}

@media (max-width: 1200px) {
  .dashboard-container {
    max-width: 960px;
  }
}

@media (max-width: 992px) {
  .dashboard-container {
    padding: 0 16px;
  }
}

@media (max-width: 768px) {
  .dashboard-title {
    font-size: 24px;
  }
  
  .stat-card .stat-value {
    font-size: 36px;
  }
  
  .chart-container {
    flex-direction: column;
    gap: 20px;
  }
  
  .dashboard-container {
    padding: 0 12px;
  }
  
  .chart-stats {
    flex-direction: column;
    gap: 12px;
  }
  
  .stat-item {
    display: flex;
    justify-content: center;
  }
}
</style>