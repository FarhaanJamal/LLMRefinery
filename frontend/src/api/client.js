import axios from "axios";

const api = axios.create({
  baseURL: "http://localhost:8080/api",
});

export async function uploadDataset(file) {
  const formData = new FormData();
  formData.append("file", file);
  const res = await api.post("/dataset/upload", formData);
  return res.data;
}

export async function startExperiment(config) {
  const res = await api.post("/experiment/start", config);
  return res.data;
}

export async function getJobStatus(jobId) {
  const res = await api.get(`/job/${jobId}`);
  return res.data;
}

export async function getActiveJobs() {
  const res = await api.get("/jobs/active");
  return res.data.jobs;
}

export async function getResults() {
  const res = await api.get("/experiments/results");
  return res.data.experiments;
}

export async function deleteExperiment(runId) {
  const res = await api.delete(`/experiments/${runId}`);
  return res.data;
}

export async function deployModel(runId) {
  const res = await api.post("/models/deploy", { run_id: runId });
  return res.data;
}

export async function undeployModel() {
  const res = await api.delete("/models/deploy");
  return res.data;
}

export async function getServingStatus() {
  const res = await api.get("/models/serving-status");
  return res.data;
}

export function subscribeEvents(onEvent) {
  const es = new EventSource("http://localhost:8080/api/events");
  es.onmessage = (e) => {
    try {
      onEvent(JSON.parse(e.data));
    } catch { /* ignore parse errors */ }
  };
  return es;
}

export async function chatCompletions(messages, { stream = false, maxTokens = 512, temperature = 0.7 } = {}) {
  if (!stream) {
    const res = await api.post("/chat/completions", {
      messages,
      max_tokens: maxTokens,
      temperature,
      stream: false,
    });
    return res.data;
  }

  // Streaming — return a ReadableStream of SSE chunks
  const resp = await fetch("http://localhost:8080/api/chat/completions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      messages,
      max_tokens: maxTokens,
      temperature,
      stream: true,
    }),
  });

  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(text || resp.statusText);
  }

  return resp.body;
}

export default api;
