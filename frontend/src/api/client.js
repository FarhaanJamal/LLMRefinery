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

export async function getResults() {
  const res = await api.get("/experiments/results");
  return res.data.experiments;
}

export default api;
