<script setup>
import { ref, computed } from 'vue'
import { sendQuery } from './api/backend_connection.js'

const history = ref([])
const answer = ref(null)
const question = ref('')
const isLoading = ref(false)
const error = ref('')

const intentConfig = {
  outliers: {
    icon: '🔍',
    title: 'Outlier Detection',
    color: '#f093fb',
    gradient: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
  },
  top: {
    icon: '🏆',
    title: 'Top Values Analysis',
    color: '#4facfe',
    gradient: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
  },
  cluster: {
    icon: '📊',
    title: 'Country Clustering',
    color: '#43e97b',
    gradient: 'linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)',
  },
  trend: {
    icon: '📈',
    title: 'Trend Analysis',
    color: '#fa709a',
    gradient: 'linear-gradient(135deg, #fa709a 0%, #fee140 100%)',
  },
}

const hasValidResponse = computed(() => {
  return answer.value && answer.value.intent
})

async function handleQuery() {
  if (!question.value.trim()) {
    error.value = 'Please enter a question'
    return
  }

  isLoading.value = true
  error.value = ''
  answer.value = null

  try {
    const response = await sendQuery(question.value)

    if (response.error) {
      error.value = response.error
      answer.value = null
    } else {
      answer.value = response

      history.value.unshift({
        question: question.value,
        response: response,
        timestamp: new Date().toLocaleTimeString(),
      })
    }

    question.value = ''
  } catch (err) {
    error.value = 'Failed to get response. Please try again.'
    console.error(err)
  } finally {
    isLoading.value = false
  }
}

function clearHistory() {
  history.value = []
  answer.value = null
}

function getConfidenceLevel(score) {
  if (score >= 0.8) return 'High'
  if (score >= 0.6) return 'Medium'
  return 'Low'
}

function getConfidenceColor(score) {
  if (score >= 0.8) return '#43e97b'
  if (score >= 0.6) return '#fee140'
  return '#f5576c'
}

function formatValue(value) {
  if (typeof value === 'number') {
    return value.toFixed(2)
  }
  return value
}

function getClusterGroups(result) {
  const clusters = {}
  result.forEach((item) => {
    const clusterId = item.Cluster
    if (!clusters[clusterId]) {
      clusters[clusterId] = {
        id: clusterId,
        countries: [],
      }
    }
    clusters[clusterId].countries.push(item)
  })
  return Object.values(clusters).sort((a, b) => a.id - b.id)
}

function getClusterColor(clusterId) {
  const colors = [
    'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
    'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
    'linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)',
    'linear-gradient(135deg, #fa709a 0%, #fee140 100%)',
  ]
  return colors[clusterId % colors.length]
}

function getBarHeight(value, allValues) {
  const max = Math.max(...allValues)
  const min = Math.min(...allValues)
  const range = max - min
  if (range === 0) return '50%'
  return ((value - min) / range) * 80 + 10 + '%'
}
</script>

<template>
  <div class="app-container">
    <div class="main-content">
      <header class="header">
        <div class="header-icon">🍎</div>
        <h1 class="title">Food Inflation Insights</h1>
        <p class="subtitle">Ask questions about food prices and inflation trends</p>
      </header>

      <div class="query-section">
        <div class="input-container">
          <textarea
            v-model="question"
            placeholder="Ask about food inflation trends, prices, or economic impacts..."
            rows="4"
            class="question-input"
            @keydown.ctrl.enter="handleQuery"
            :disabled="isLoading"
          ></textarea>
          <div class="input-footer">
            <span class="hint">Press Ctrl + Enter to submit</span>
            <button
              @click="handleQuery"
              class="submit-btn"
              :disabled="isLoading || !question.trim()"
            >
              <span v-if="!isLoading">Ask AI</span>
              <span v-else class="loading-text">
                <span class="spinner"></span>
                Processing...
              </span>
            </button>
          </div>
        </div>

        <div v-if="error" class="error-message">⚠️ {{ error }}</div>

        <!-- Response Display -->
        <div v-if="hasValidResponse" class="response-container">
          <!-- Intent Header -->
          <div class="intent-header" :style="{ background: intentConfig[answer.intent]?.gradient }">
            <div class="intent-icon">{{ intentConfig[answer.intent]?.icon }}</div>
            <div class="intent-info">
              <h2 class="intent-title">{{ intentConfig[answer.intent]?.title }}</h2>
              <div class="intent-meta">
                <span class="meta-item"> <strong>Detected:</strong> {{ answer.intent }} </span>
                <span class="meta-divider">•</span>
                <span class="meta-item">
                  <strong>Confidence:</strong>
                  <span
                    class="confidence-badge"
                    :style="{ background: getConfidenceColor(answer.score) }"
                  >
                    {{ getConfidenceLevel(answer.score) }} ({{ (answer.score * 100).toFixed(1) }}%)
                  </span>
                </span>
              </div>
            </div>
          </div>

          <!-- Outliers Display -->
          <div v-if="answer.intent === 'outliers'" class="results-content">
            <div class="section-title">
              <span class="title-icon">🔍</span>
              Detected Outliers
              <span class="count-badge">{{ answer.result?.length || 0 }} found</span>
            </div>
            <div v-if="answer.result && answer.result.length > 0" class="outliers-grid">
              <div v-for="(item, idx) in answer.result" :key="idx" class="outlier-card">
                <div class="outlier-country">{{ item.REF_AREA_LABEL || item.Country }}</div>
                <div class="outlier-details">
                  <span class="detail-label">Period:</span>
                  <span class="detail-value">{{ item.TIME_PERIOD }}</span>
                </div>
                <div class="outlier-value">{{ formatValue(item.OBS_VALUE) }}%</div>
              </div>
            </div>
            <div v-else class="empty-state">No outliers detected</div>
          </div>

          <!-- Top Values Display -->
          <div v-if="answer.intent === 'top'" class="results-content">
            <div class="section-title">
              <span class="title-icon">🏆</span>
              Top Inflation Values
              <span class="count-badge">Top {{ answer.result?.length || 0 }}</span>
            </div>
            <div v-if="answer.result && answer.result.length > 0" class="top-list">
              <div v-for="(item, idx) in answer.result" :key="idx" class="top-item">
                <div class="rank-badge" :class="'rank-' + (idx + 1)">
                  {{ idx + 1 }}
                </div>
                <div class="top-content">
                  <div class="top-country">{{ item.REF_AREA_LABEL }}</div>
                  <div class="top-period">{{ item.TIME_PERIOD }}</div>
                </div>
                <div class="top-value">{{ formatValue(item.OBS_VALUE) }}%</div>
              </div>
            </div>
            <div v-else class="empty-state">No data available</div>
          </div>

          <!-- Cluster Display -->
          <div v-if="answer.intent === 'cluster'" class="results-content">
            <div class="section-title">
              <span class="title-icon">📊</span>
              Country Clusters
              <span class="count-badge">{{ answer.result?.length || 0 }} countries</span>
            </div>
            <div v-if="answer.result && answer.result.length > 0">
              <div
                v-for="cluster in getClusterGroups(answer.result)"
                :key="cluster.id"
                class="cluster-group"
              >
                <div class="cluster-header">
                  <span class="cluster-badge" :style="{ background: getClusterColor(cluster.id) }">
                    Cluster {{ cluster.id }}
                  </span>
                  <span class="cluster-count">{{ cluster.countries.length }} countries</span>
                </div>
                <div class="cluster-items">
                  <div
                    v-for="(country, idx) in cluster.countries"
                    :key="idx"
                    class="cluster-country"
                  >
                    <span class="country-name">{{ country.Country }}</span>
                    <span class="country-inflation"
                      >Avg: {{ formatValue(country.MeanInflation) }}%</span
                    >
                  </div>
                </div>
              </div>
            </div>
            <div v-else class="empty-state">No clustering data available</div>
          </div>

          <!-- Trend Display -->
          <div v-if="answer.intent === 'trend'" class="results-content">
            <div class="section-title">
              <span class="title-icon">📈</span>
              Trend Analysis
            </div>
            <div class="trend-stats">
              <div class="stat-card">
                <div class="stat-label">Trend Direction</div>
                <div class="stat-value" :class="answer.slope > 0 ? 'positive' : 'negative'">
                  {{ answer.slope > 0 ? '📈 Increasing' : '📉 Decreasing' }}
                </div>
              </div>
              <div class="stat-card">
                <div class="stat-label">Slope</div>
                <div class="stat-value">{{ formatValue(answer.slope) }}</div>
              </div>
              <div class="stat-card">
                <div class="stat-label">Intercept</div>
                <div class="stat-value">{{ formatValue(answer.intercept) }}</div>
              </div>
              <div class="stat-card">
                <div class="stat-label">Data Points</div>
                <div class="stat-value">{{ answer.predicted?.length || 0 }}</div>
              </div>
            </div>
            <div v-if="answer.predicted && answer.predicted.length > 0" class="trend-chart">
              <div class="chart-header">Predicted Trend Line</div>
              <div class="simple-chart">
                <div
                  v-for="(val, idx) in answer.predicted.slice(0, 20)"
                  :key="idx"
                  class="chart-bar"
                  :style="{
                    height: getBarHeight(val, answer.predicted),
                    background: `hsl(${200 + idx * 5}, 70%, 60%)`,
                  }"
                  :title="`Point ${idx + 1}: ${formatValue(val)}`"
                ></div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- History Section -->
      <div v-if="history.length > 0" class="history-section">
        <div class="history-header">
          <h2>Conversation History</h2>
          <button @click="clearHistory" class="clear-btn">Clear All</button>
        </div>
        <div class="history-list">
          <div v-for="(item, index) in history" :key="index" class="history-item">
            <div class="history-question">
              <span class="badge">Q</span>
              <span class="text">{{ item.question }}</span>
              <span class="timestamp">{{ item.timestamp }}</span>
            </div>
            <div class="history-answer">
              <span class="badge">A</span>
              <div class="text">
                <div
                  class="history-intent-tag"
                  :style="{ background: intentConfig[item.response.intent]?.gradient }"
                >
                  {{ intentConfig[item.response.intent]?.icon }}
                  {{ intentConfig[item.response.intent]?.title }}
                </div>
                <div class="history-summary">
                  <span v-if="item.response.intent === 'outliers'">
                    Found {{ item.response.result?.length || 0 }} outliers
                  </span>
                  <span v-else-if="item.response.intent === 'top'">
                    Top {{ item.response.result?.length || 0 }} values analyzed
                  </span>
                  <span v-else-if="item.response.intent === 'cluster'">
                    {{ item.response.result?.length || 0 }} countries clustered
                  </span>
                  <span v-else-if="item.response.intent === 'trend'">
                    Trend: {{ item.response.slope > 0 ? 'Increasing' : 'Decreasing' }} (slope:
                    {{ formatValue(item.response.slope) }})
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
* {
  box-sizing: border-box;
}

.app-container {
  min-height: 100vh;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  padding: 2rem 1rem;
  font-family:
    -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
}

.main-content {
  max-width: 900px;
  margin: 0 auto;
}

.header {
  text-align: center;
  margin-bottom: 3rem;
  color: white;
}

.header-icon {
  font-size: 4rem;
  margin-bottom: 1rem;
  animation: bounce 2s infinite;
}

@keyframes bounce {
  0%,
  100% {
    transform: translateY(0);
  }
  50% {
    transform: translateY(-10px);
  }
}

.title {
  font-size: 2.5rem;
  font-weight: 700;
  margin: 0 0 0.5rem 0;
  text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
}

.subtitle {
  font-size: 1.1rem;
  opacity: 0.95;
  margin: 0;
  font-weight: 300;
}

.query-section {
  background: white;
  border-radius: 16px;
  padding: 2rem;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
  margin-bottom: 2rem;
}

.input-container {
  margin-bottom: 1rem;
}

.question-input {
  width: 100%;
  padding: 1rem;
  border: 2px solid #e0e0e0;
  border-radius: 12px;
  font-size: 1rem;
  font-family: inherit;
  resize: vertical;
  transition: all 0.3s ease;
  margin-bottom: 0.75rem;
}

.question-input:focus {
  outline: none;
  border-color: #667eea;
  box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
}

.question-input:disabled {
  background: #f5f5f5;
  cursor: not-allowed;
}

.input-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.hint {
  font-size: 0.85rem;
  color: #999;
}

.submit-btn {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  padding: 0.75rem 2rem;
  border-radius: 8px;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
  box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
}

.submit-btn:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 6px 16px rgba(102, 126, 234, 0.4);
}

.submit-btn:active:not(:disabled) {
  transform: translateY(0);
}

.submit-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
  transform: none;
}

.loading-text {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.spinner {
  width: 16px;
  height: 16px;
  border: 2px solid rgba(255, 255, 255, 0.3);
  border-top-color: white;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

.error-message {
  background: #fee;
  color: #c33;
  padding: 1rem;
  border-radius: 8px;
  margin-bottom: 1rem;
  border-left: 4px solid #c33;
}

.response-card {
  background: #f8f9ff;
  border-radius: 12px;
  overflow: hidden;
  border: 1px solid #e0e7ff;
}

.card-header {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 1rem 1.5rem;
}

.card-header h2 {
  margin: 0;
  font-size: 1.25rem;
  font-weight: 600;
}

.card-content {
  padding: 1.5rem;
}

.card-content p {
  margin: 0;
  line-height: 1.6;
  color: #333;
  white-space: pre-wrap;
}

.history-section {
  background: white;
  border-radius: 16px;
  padding: 2rem;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
}

.history-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1.5rem;
  padding-bottom: 1rem;
  border-bottom: 2px solid #f0f0f0;
}

.history-header h2 {
  margin: 0;
  font-size: 1.5rem;
  color: #333;
}

.clear-btn {
  background: #f5f5f5;
  border: 1px solid #ddd;
  padding: 0.5rem 1rem;
  border-radius: 6px;
  font-size: 0.9rem;
  cursor: pointer;
  transition: all 0.2s ease;
}

.clear-btn:hover {
  background: #e0e0e0;
  border-color: #ccc;
}

.history-list {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.history-item {
  padding: 1rem;
  background: #fafafa;
  border-radius: 10px;
  border: 1px solid #e0e0e0;
  transition: all 0.2s ease;
}

.history-item:hover {
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
  transform: translateY(-2px);
}

.history-question,
.history-answer {
  display: flex;
  gap: 0.75rem;
  margin-bottom: 0.5rem;
  align-items: flex-start;
}

.history-answer {
  margin-bottom: 0;
}

.badge {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  width: 28px;
  height: 28px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 700;
  font-size: 0.85rem;
  flex-shrink: 0;
}

.history-answer .badge {
  background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
}

.text {
  flex: 1;
  line-height: 1.6;
  color: #333;
}

.timestamp {
  font-size: 0.75rem;
  color: #999;
  white-space: nowrap;
}

.history-intent-tag {
  display: inline-block;
  color: white;
  padding: 0.25rem 0.75rem;
  border-radius: 12px;
  font-size: 0.85rem;
  font-weight: 600;
  margin-bottom: 0.5rem;
}

.history-summary {
  color: #666;
  font-size: 0.95rem;
}

/* Response Display Styles */
.response-container {
  margin-top: 1.5rem;
  border-radius: 16px;
  overflow: hidden;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
}

.intent-header {
  color: white;
  padding: 1.5rem;
  display: flex;
  align-items: center;
  gap: 1rem;
}

.intent-icon {
  font-size: 3rem;
  line-height: 1;
}

.intent-info {
  flex: 1;
}

.intent-title {
  margin: 0 0 0.5rem 0;
  font-size: 1.5rem;
  font-weight: 700;
}

.intent-meta {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  font-size: 0.9rem;
  opacity: 0.95;
}

.meta-item {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.meta-divider {
  opacity: 0.5;
}

.confidence-badge {
  padding: 0.25rem 0.5rem;
  border-radius: 4px;
  font-weight: 600;
  color: white;
}

.results-content {
  background: white;
  padding: 2rem;
}

.section-title {
  font-size: 1.25rem;
  font-weight: 700;
  color: #333;
  margin-bottom: 1.5rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.title-icon {
  font-size: 1.5rem;
}

.count-badge {
  margin-left: auto;
  background: #f0f0f0;
  color: #666;
  padding: 0.25rem 0.75rem;
  border-radius: 12px;
  font-size: 0.85rem;
  font-weight: 600;
}

.empty-state {
  text-align: center;
  padding: 3rem 1rem;
  color: #999;
  font-size: 1.1rem;
}

/* Outliers Styles */
.outliers-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 1rem;
}

.outlier-card {
  background: linear-gradient(135deg, #fff5f5 0%, #ffe5e5 100%);
  border: 2px solid #ffcccb;
  border-radius: 12px;
  padding: 1.25rem;
  transition: all 0.3s ease;
}

.outlier-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 6px 20px rgba(245, 87, 108, 0.2);
}

.outlier-country {
  font-size: 1.1rem;
  font-weight: 700;
  color: #c33;
  margin-bottom: 0.75rem;
}

.outlier-details {
  display: flex;
  gap: 0.5rem;
  margin-bottom: 0.75rem;
  font-size: 0.9rem;
}

.detail-label {
  color: #666;
}

.detail-value {
  color: #333;
  font-weight: 600;
}

.outlier-value {
  font-size: 1.75rem;
  font-weight: 700;
  color: #f5576c;
  text-align: right;
}

/* Top Values Styles */
.top-list {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.top-item {
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 1rem;
  background: linear-gradient(135deg, #f8f9ff 0%, #e8f4ff 100%);
  border-radius: 10px;
  border: 1px solid #d0e8ff;
  transition: all 0.2s ease;
}

.top-item:hover {
  transform: translateX(4px);
  box-shadow: 0 4px 12px rgba(79, 172, 254, 0.2);
}

.rank-badge {
  width: 40px;
  height: 40px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 700;
  font-size: 1.1rem;
  color: white;
  background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
  flex-shrink: 0;
}

.rank-badge.rank-1 {
  background: linear-gradient(135deg, #ffd700 0%, #ffed4e 100%);
  box-shadow: 0 4px 12px rgba(255, 215, 0, 0.4);
}

.rank-badge.rank-2 {
  background: linear-gradient(135deg, #c0c0c0 0%, #e8e8e8 100%);
  box-shadow: 0 4px 12px rgba(192, 192, 192, 0.4);
}

.rank-badge.rank-3 {
  background: linear-gradient(135deg, #cd7f32 0%, #e8a87c 100%);
  box-shadow: 0 4px 12px rgba(205, 127, 50, 0.4);
}

.top-content {
  flex: 1;
}

.top-country {
  font-weight: 700;
  font-size: 1.05rem;
  color: #333;
  margin-bottom: 0.25rem;
}

.top-period {
  font-size: 0.85rem;
  color: #666;
}

.top-value {
  font-size: 1.5rem;
  font-weight: 700;
  color: #00a8e8;
}

/* Cluster Styles */
.cluster-group {
  margin-bottom: 1.5rem;
  border-radius: 12px;
  overflow: hidden;
  border: 2px solid #e0e0e0;
}

.cluster-group:last-child {
  margin-bottom: 0;
}

.cluster-header {
  padding: 1rem 1.25rem;
  background: #f8f8f8;
  display: flex;
  justify-content: space-between;
  align-items: center;
  border-bottom: 1px solid #e0e0e0;
}

.cluster-badge {
  color: white;
  padding: 0.5rem 1rem;
  border-radius: 20px;
  font-weight: 700;
  font-size: 0.95rem;
}

.cluster-count {
  color: #666;
  font-size: 0.9rem;
}

.cluster-items {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 0.75rem;
  padding: 1rem;
  background: white;
}

.cluster-country {
  padding: 0.75rem;
  background: #fafafa;
  border-radius: 8px;
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
  transition: all 0.2s ease;
}

.cluster-country:hover {
  background: #f0f0f0;
  transform: translateY(-2px);
}

.country-name {
  font-weight: 600;
  color: #333;
  font-size: 0.95rem;
}

.country-inflation {
  font-size: 0.85rem;
  color: #666;
}

/* Trend Styles */
.trend-stats {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 1rem;
  margin-bottom: 2rem;
}

.stat-card {
  background: linear-gradient(135deg, #fff9f0 0%, #ffe8d0 100%);
  border: 2px solid #ffd4a3;
  border-radius: 12px;
  padding: 1.25rem;
  text-align: center;
}

.stat-label {
  font-size: 0.85rem;
  color: #666;
  margin-bottom: 0.5rem;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.stat-value {
  font-size: 1.5rem;
  font-weight: 700;
  color: #333;
}

.stat-value.positive {
  color: #43e97b;
}

.stat-value.negative {
  color: #f5576c;
}

.trend-chart {
  background: #f8f9ff;
  border-radius: 12px;
  padding: 1.5rem;
  border: 1px solid #e0e7ff;
}

.chart-header {
  font-weight: 600;
  color: #333;
  margin-bottom: 1rem;
  font-size: 1.05rem;
}

.simple-chart {
  display: flex;
  align-items: flex-end;
  gap: 4px;
  height: 200px;
  padding: 1rem;
  background: white;
  border-radius: 8px;
}

.chart-bar {
  flex: 1;
  min-width: 8px;
  border-radius: 4px 4px 0 0;
  transition: all 0.3s ease;
  cursor: pointer;
}

.chart-bar:hover {
  opacity: 0.8;
  transform: scaleY(1.05);
}

/* Responsive Design */
@media (max-width: 768px) {
  .app-container {
    padding: 1rem 0.5rem;
  }

  .title {
    font-size: 1.75rem;
  }

  .subtitle {
    font-size: 0.95rem;
  }

  .query-section,
  .history-section {
    padding: 1.5rem;
    border-radius: 12px;
  }

  .input-footer {
    flex-direction: column;
    gap: 0.75rem;
    align-items: stretch;
  }

  .submit-btn {
    width: 100%;
  }

  .hint {
    text-align: center;
  }

  .history-header {
    flex-direction: column;
    gap: 1rem;
    align-items: flex-start;
  }

  .clear-btn {
    width: 100%;
  }

  .timestamp {
    display: block;
    margin-top: 0.25rem;
  }

  .intent-header {
    flex-direction: column;
    text-align: center;
  }

  .intent-meta {
    flex-direction: column;
    gap: 0.5rem;
  }

  .meta-divider {
    display: none;
  }

  .outliers-grid {
    grid-template-columns: 1fr;
  }

  .trend-stats {
    grid-template-columns: repeat(2, 1fr);
  }

  .cluster-items {
    grid-template-columns: 1fr;
  }

  .simple-chart {
    height: 150px;
  }
}

@media (max-width: 480px) {
  .header-icon {
    font-size: 3rem;
  }

  .title {
    font-size: 1.5rem;
  }

  .question-input {
    font-size: 16px; /* Prevents zoom on iOS */
  }

  .intent-icon {
    font-size: 2rem;
  }

  .intent-title {
    font-size: 1.25rem;
  }

  .trend-stats {
    grid-template-columns: 1fr;
  }

  .section-title {
    font-size: 1.1rem;
    flex-wrap: wrap;
  }

  .count-badge {
    margin-left: 0;
  }
}
</style>
