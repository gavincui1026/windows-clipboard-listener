<script setup lang="ts">
import { useRoute } from 'vue-router'
import { computed } from 'vue'
import Layout from './components/Layout.vue'

const route = useRoute()
const isLogin = computed(() => route.path === '/login')
</script>

<template>
  <div id="app">
    <router-view v-if="isLogin" />
    <Layout v-else />
  </div>
</template>

<style>
/* 确保应用占满整个视口 */
#app {
  width: 100%;
  height: 100vh;
  overflow: hidden;
}
</style>

