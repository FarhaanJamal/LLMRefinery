import { useState, useEffect, useCallback } from "react";
import { uploadDataset, startExperiment, getJobStatus, getActiveJobs } from "../api/client";

const STORAGE_KEY = "llm-refinery-active-jobs";

const MODELS = [
  "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "meta-llama/Meta-Llama-3-8B",
  "mistralai/Mistral-7B-v0.1",
  "google/gemma-7b",
];

const ALL_STEPS = [
  { key: "queued",      label: "Queued" },
  { key: "training",    label: "Training" },
  { key: "quantizing",  label: "Quantizing" },
  { key: "evaluating",  label: "Evaluating" },
  { key: "completed",   label: "Done" },
];

function getSteps(quantType) {
  if (quantType === "none") {
    return ALL_STEPS.filter((s) => s.key !== "quantizing");
  }
  return ALL_STEPS;
}

function JobTracker({ jobId, onDismiss }) {
  const [job, setJob] = useState(null);

  useEffect(() => {
    let active = true;
    async function poll() {
      try {
        const data = await getJobStatus(jobId);
        if (active) setJob(data);
      } catch { /* ignore */ }
    }
    poll();
    const id = setInterval(poll, 4000);
    return () => { active = false; clearInterval(id); };
  }, [jobId]);

  const status = job?.status || "queued";
  const isFailed = status === "failed";
  const isDone = status === "completed";
  const quantType = job?.params?.quant_type || "awq";
  const steps = getSteps(quantType);
  const currentIdx = steps.findIndex((s) => s.key === status);

  return (
    <div className="mt-5 rounded border border-gray-600 bg-gray-800 p-4 space-y-3">
      <div className="flex items-center justify-between">
        <p className="text-sm text-gray-300">
          Job <span className="font-mono text-white">{jobId.slice(0, 8)}</span>
        </p>
        <button onClick={onDismiss} className="text-xs text-gray-500 hover:text-white" title="Dismiss">✕</button>
      </div>

      {/* Step progress */}
      <div className="flex items-center gap-1">
        {steps.map((step, i) => {
          const done = i < currentIdx || isDone;
          const active = i === currentIdx && !isDone && !isFailed;
          return (
            <div key={step.key} className="flex-1 flex flex-col items-center gap-1">
              <div
                className={`h-2 w-full rounded-full transition-colors ${
                  done
                    ? "bg-green-500"
                    : active
                      ? "bg-indigo-500 animate-pulse"
                      : isFailed && i === currentIdx
                        ? "bg-red-500"
                        : "bg-gray-600"
                }`}
              />
              <span className={`text-[10px] ${done ? "text-green-400" : active ? "text-indigo-300" : "text-gray-500"}`}>
                {step.label}
              </span>
            </div>
          );
        })}
      </div>

      {isFailed && (
        <p className="text-red-400 text-xs">Job failed. Check logs for details.</p>
      )}

      {isDone && (
        <p className="text-green-400 text-xs">Pipeline complete — check Results tab.</p>
      )}
    </div>
  );
}

export default function UploadForm() {
  const [file, setFile] = useState(null);
  const [model, setModel] = useState(MODELS[0]);
  const [r, setR] = useState(16);
  const [alpha, setAlpha] = useState(32);
  const [quantType, setQuantType] = useState("awq");
  const [evalMode, setEvalMode] = useState("quick");
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activeJobs, setActiveJobs] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem(STORAGE_KEY)) || [];
    } catch { return []; }
  });

  // Persist to localStorage whenever activeJobs changes
  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(activeJobs));
  }, [activeJobs]);

  // On mount, also fetch any active jobs from backend (handles other tabs / cleared storage)
  useEffect(() => {
    async function loadActive() {
      try {
        const jobs = await getActiveJobs();
        if (jobs.length > 0) {
          setActiveJobs((prev) => {
            const existing = new Set(prev);
            const merged = [...prev];
            for (const j of jobs) {
              if (!existing.has(j.job_id)) merged.push(j.job_id);
            }
            return merged;
          });
        }
      } catch { /* ignore */ }
    }
    loadActive();
  }, []);

  const dismissJob = useCallback((jobId) => {
    setActiveJobs((prev) => prev.filter((id) => id !== jobId));
  }, []);

  async function handleSubmit(e) {
    e.preventDefault();
    if (!file) return setStatus({ type: "error", msg: "Select a .jsonl file." });

    setLoading(true);
    setStatus(null);

    try {
      const uploadRes = await uploadDataset(file);
      const experimentRes = await startExperiment({
        model,
        task: "qlora",
        params: { r, alpha, quant_type: quantType, eval_mode: evalMode },
        dataset_path: uploadRes.s3_path,
      });
      setActiveJobs((prev) => [experimentRes.job_id, ...prev]);
    } catch (err) {
      setStatus({
        type: "error",
        msg: err.response?.data?.detail || "Something went wrong.",
      });
    } finally {
      setLoading(false);
    }
  }

  return (
    <form onSubmit={handleSubmit} className="max-w-lg mx-auto space-y-5">
      {/* File */}
      <div>
        <label className="block text-sm font-medium text-gray-300 mb-1">
          Dataset (.jsonl)
        </label>
        <input
          type="file"
          accept=".jsonl"
          onChange={(e) => setFile(e.target.files[0])}
          className="block w-full text-sm text-gray-400 file:mr-4 file:py-2 file:px-4
                     file:rounded file:border-0 file:text-sm file:font-semibold
                     file:bg-indigo-600 file:text-white hover:file:bg-indigo-500"
        />
      </div>

      {/* Model */}
      <div>
        <label className="block text-sm font-medium text-gray-300 mb-1">
          Base Model
        </label>
        <select
          value={model}
          onChange={(e) => setModel(e.target.value)}
          className="w-full rounded bg-gray-800 border border-gray-600 px-3 py-2 text-sm text-white"
        >
          {MODELS.map((m) => (
            <option key={m} value={m}>{m}</option>
          ))}
        </select>
      </div>

      {/* LoRA Rank & Alpha */}
      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-1">
            LoRA Rank (r)
          </label>
          <input
            type="number"
            min={1}
            value={r}
            onChange={(e) => setR(Number(e.target.value))}
            className="w-full rounded bg-gray-800 border border-gray-600 px-3 py-2 text-sm text-white"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-1">
            LoRA Alpha
          </label>
          <input
            type="number"
            min={1}
            value={alpha}
            onChange={(e) => setAlpha(Number(e.target.value))}
            className="w-full rounded bg-gray-800 border border-gray-600 px-3 py-2 text-sm text-white"
          />
        </div>
      </div>

      {/* Quantization */}
      <div>
        <label className="block text-sm font-medium text-gray-300 mb-1">
          Quantization Type
        </label>
        <select
          value={quantType}
          onChange={(e) => setQuantType(e.target.value)}
          className="w-full rounded bg-gray-800 border border-gray-600 px-3 py-2 text-sm text-white"
        >
          <option value="awq">AWQ (4-bit)</option>
          <option value="none">None (FP16)</option>
        </select>
      </div>

      {/* Eval Mode */}
      <div>
        <label className="block text-sm font-medium text-gray-300 mb-1">
          Evaluation Mode
        </label>
        <select
          value={evalMode}
          onChange={(e) => setEvalMode(e.target.value)}
          className="w-full rounded bg-gray-800 border border-gray-600 px-3 py-2 text-sm text-white"
        >
          <option value="quick">Quick (ROUGE-L only)</option>
          <option value="full">Full (ROUGE-L + MMLU benchmarks)</option>
        </select>
      </div>

      {/* Submit */}
      <button
        type="submit"
        disabled={loading}
        className="w-full rounded bg-indigo-600 px-4 py-2.5 text-sm font-semibold text-white
                   hover:bg-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {loading ? "Submitting..." : "Upload & Start Experiment"}
      </button>

      {/* Status */}
      {status && (
        <div
          className={`rounded px-4 py-3 text-sm ${
            status.type === "success"
              ? "bg-green-900/50 text-green-300 border border-green-700"
              : "bg-red-900/50 text-red-300 border border-red-700"
          }`}
        >
          {status.msg}
        </div>
      )}

      {/* Active job trackers */}
      {activeJobs.map((jobId) => (
        <JobTracker
          key={jobId}
          jobId={jobId}
          onDismiss={() => dismissJob(jobId)}
        />
      ))}
    </form>
  );
}
