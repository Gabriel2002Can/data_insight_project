<script setup>
import { ref } from 'vue'
import { sendQuery } from './api/backend_connection.js'

const history = ref([])
const answer = ref('')
const question = ref('')
const isLoading = ref(false)

async function handleQuery() {
  isLoading.value = true
  answer.value = await sendQuery(question.value)

  history.value.push({ question: question.value, response: answer.value })
  isLoading.value = false
}
</script>

<template>
  <h1>Ask your question about food inflation to the AI!</h1>
  <div>
    <div>
      <textarea
        v-model="question"
        placeholder="Type your question here..."
        rows="4"
        cols="50"
      ></textarea>
    </div>
    <div>
      <button @click="handleQuery">Ask AI</button>
    </div>
    <div v-if="isLoading">Loading...</div>
    <div v-else>
      <h2>Response:</h2>
      <p>{{ answer }}</p>
    </div>
    <div v-if="history != null && history.length > 0">
      <h2>History:</h2>
      <ul>
        <li v-for="(item, index) in history" :key="index">
          <strong>Q:</strong> {{ item.question }}<br />
          <strong>A:</strong> {{ item.response }}
        </li>
      </ul>
    </div>
  </div>
</template>

<style scoped></style>
