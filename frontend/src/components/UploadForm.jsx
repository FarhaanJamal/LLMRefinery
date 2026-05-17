import { useState, useEffect, useCallback } from "react";
import { uploadDataset, startExperiment, getJobStatus, getActiveJobs, subscribeEvents } from "../api/client";

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

    // Initial fetch
    getJobStatus(jobId).then((data) => { if (active) setJob(data); }).catch(() => {});

    // SSE-driven updates
    const es = subscribeEvents((event) => {
      if (event.type === "job_status" && event.job_id === jobId) {
        // Re-fetch full job data on status change
        getJobStatus(jobId).then((data) => { if (active) setJob(data); }).catch(() => {});
      }
    });

    // Fallback poll every 30s in case SSE drops
    const fallback = setInterval(() => {
      getJobStatus(jobId).then((data) => { if (active) setJob(data); }).catch(() => {});
    }, 30000);

    return () => { active = false; es.close(); clearInterval(fallback); };
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
  // LoRA config
  const [r, setR] = useState(16);
  const [alpha, setAlpha] = useState(32);
  const [loraDropout, setLoraDropout] = useState(0.05);
  const [targetModules, setTargetModules] = useState("all-linear");
  // Training config
  const [numEpochs, setNumEpochs] = useState(-1);
  const [maxSteps, setMaxSteps] = useState(500);

  const handleEpochsChange = (val) => {
    setNumEpochs(val);
    if (val > 0) setMaxSteps(-1);
  };
  const handleMaxStepsChange = (val) => {
    setMaxSteps(val);
    if (val > 0) setNumEpochs(-1);
  };

  const [learningRate, setLearningRate] = useState(2e-4);
  const [batchSize, setBatchSize] = useState(2);
  const [gradAccum, setGradAccum] = useState(4);
  const [lrScheduler, setLrScheduler] = useState("cosine");
  const [warmupSteps, setWarmupSteps] = useState(10);
  const [maxGradNorm, setMaxGradNorm] = useState(0.3);
  const [seed, setSeed] = useState(42);
  const [maxSeqLength, setMaxSeqLength] = useState(512);
  // Quantization config
  const [quantType, setQuantType] = useState("awq");
  const [wBit, setWBit] = useState(4);
  const [qGroupSize, setQGroupSize] = useState(128);
  // Evaluation config
  const [evalMode, setEvalMode] = useState("quick");
  const [maxNewTokens, setMaxNewTokens] = useState(256);
  // UI state
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(false);
  const [openSections, setOpenSections] = useState({});
  const [activeJobs, setActiveJobs] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem(STORAGE_KEY)) || [];
    } catch { return []; }
  });

  const toggleSection = useCallback((key) => {
    setOpenSections((prev) => ({ ...prev, [key]: !prev[key] }));
  }, []);

  // Persist to localStorage whenever activeJobs changes
  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(activeJobs));
  }, [activeJobs]);

  // Remove progress bar when experiment is deleted from Results tab
  useEffect(() => {
    const es = subscribeEvents((event) => {
      if (event.type === "job_deleted" && event.job_id) {
        setActiveJobs((prev) => prev.filter((id) => id !== event.job_id));
      }
    });
    return () => es.close();
  }, []);

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
        params: {
          r, alpha, lora_dropout: loraDropout, target_modules: targetModules,
          num_train_epochs: numEpochs,
          max_steps: maxSteps, learning_rate: learningRate,
          per_device_train_batch_size: batchSize,
          gradient_accumulation_steps: gradAccum,
          lr_scheduler_type: lrScheduler, warmup_steps: warmupSteps,
          max_grad_norm: maxGradNorm, seed, max_seq_length: maxSeqLength,
          quant_type: quantType, w_bit: wBit, q_group_size: qGroupSize,
          eval_mode: evalMode, max_new_tokens: maxNewTokens,
        },
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

      {/* ---- LoRA Config ---- */}
      <fieldset className="rounded border border-gray-700 overflow-hidden">
        <button type="button" onClick={() => toggleSection("lora")}
          className="w-full flex items-center justify-between px-4 py-2.5 bg-gray-800/80 text-sm font-medium text-gray-200 hover:bg-gray-700/80">
          <span>LoRA Configuration</span>
          <span className="text-gray-500 text-xs">{openSections.lora ? "▲" : "▼"}</span>
        </button>
        {openSections.lora && (
          <div className="px-4 py-3 space-y-3 bg-gray-800/30">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-xs text-gray-400 mb-1">Rank (r)</label>
                <input type="number" min={1} value={r} onChange={(e) => setR(Number(e.target.value))}
                  className="w-full rounded bg-gray-800 border border-gray-600 px-3 py-1.5 text-sm text-white" />
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">Alpha</label>
                <input type="number" min={1} value={alpha} onChange={(e) => setAlpha(Number(e.target.value))}
                  className="w-full rounded bg-gray-800 border border-gray-600 px-3 py-1.5 text-sm text-white" />
              </div>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-xs text-gray-400 mb-1">Dropout</label>
                <input type="number" min={0} max={0.5} step={0.01} value={loraDropout}
                  onChange={(e) => setLoraDropout(Number(e.target.value))}
                  className="w-full rounded bg-gray-800 border border-gray-600 px-3 py-1.5 text-sm text-white" />
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">Target Modules</label>
                <select value={targetModules} onChange={(e) => setTargetModules(e.target.value)}
                  className="w-full rounded bg-gray-800 border border-gray-600 px-3 py-1.5 text-sm text-white">
                  <option value="all-linear">all-linear</option>
                  <option value="q_proj,v_proj">q_proj, v_proj</option>
                </select>
              </div>
            </div>
          </div>
        )}
      </fieldset>

      {/* ---- Training Config ---- */}
      <fieldset className="rounded border border-gray-700 overflow-hidden">
        <button type="button" onClick={() => toggleSection("training")}
          className="w-full flex items-center justify-between px-4 py-2.5 bg-gray-800/80 text-sm font-medium text-gray-200 hover:bg-gray-700/80">
          <span>Training Configuration</span>
          <span className="text-gray-500 text-xs">{openSections.training ? "▲" : "▼"}</span>
        </button>
        {openSections.training && (
          <div className="px-4 py-3 space-y-3 bg-gray-800/30">
            <div className="rounded bg-gray-700/40 px-3 py-2 text-[11px] text-gray-400">
              Set <strong>Epochs</strong> to a positive value to train by epochs (Max Steps will be ignored).
              Leave Epochs at -1 to train by Max Steps instead.
            </div>
            <div className="grid grid-cols-3 gap-4">
              <div>
                <label className="block text-xs text-gray-400 mb-1">Epochs</label>
                <input type="number" min={-1} value={numEpochs} onChange={(e) => handleEpochsChange(Number(e.target.value))}
                  className={`w-full rounded border px-3 py-1.5 text-sm text-white ${numEpochs > 0 ? "bg-gray-800 border-indigo-500" : "bg-gray-800 border-gray-600 opacity-60"}`} />
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">Max Steps</label>
                <input type="number" min={-1} value={maxSteps} onChange={(e) => handleMaxStepsChange(Number(e.target.value))}
                  className={`w-full rounded border px-3 py-1.5 text-sm text-white ${maxSteps > 0 ? "bg-gray-800 border-indigo-500" : "bg-gray-800 border-gray-600 opacity-60"}`} />
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">Learning Rate</label>
                <input type="number" min={0} step={0.0001} value={learningRate}
                  onChange={(e) => setLearningRate(Number(e.target.value))}
                  className="w-full rounded bg-gray-800 border border-gray-600 px-3 py-1.5 text-sm text-white" />
              </div>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-xs text-gray-400 mb-1">Batch Size per GPU</label>
                <select value={batchSize} onChange={(e) => setBatchSize(Number(e.target.value))}
                  className="w-full rounded bg-gray-800 border border-gray-600 px-3 py-1.5 text-sm text-white">
                  <option value={1}>1</option>
                  <option value={2}>2</option>
                  <option value={4}>4</option>
                </select>
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">Gradient Accumulation</label>
                <select value={gradAccum} onChange={(e) => setGradAccum(Number(e.target.value))}
                  className="w-full rounded bg-gray-800 border border-gray-600 px-3 py-1.5 text-sm text-white">
                  <option value={1}>1</option>
                  <option value={2}>2</option>
                  <option value={4}>4</option>
                  <option value={8}>8</option>
                </select>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-xs text-gray-400 mb-1">LR Scheduler</label>
                <select value={lrScheduler} onChange={(e) => setLrScheduler(e.target.value)}
                  className="w-full rounded bg-gray-800 border border-gray-600 px-3 py-1.5 text-sm text-white">
                  <option value="cosine">Cosine</option>
                  <option value="linear">Linear</option>
                  <option value="constant">Constant</option>
                </select>
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">Warmup Steps</label>
                <input type="number" min={0} value={warmupSteps} onChange={(e) => setWarmupSteps(Number(e.target.value))}
                  className="w-full rounded bg-gray-800 border border-gray-600 px-3 py-1.5 text-sm text-white" />
              </div>
            </div>
            <div className="grid grid-cols-3 gap-4">
              <div>
                <label className="block text-xs text-gray-400 mb-1">Max Grad Norm</label>
                <input type="number" min={0} step={0.1} value={maxGradNorm}
                  onChange={(e) => setMaxGradNorm(Number(e.target.value))}
                  className="w-full rounded bg-gray-800 border border-gray-600 px-3 py-1.5 text-sm text-white" />
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">Seed</label>
                <input type="number" min={0} value={seed} onChange={(e) => setSeed(Number(e.target.value))}
                  className="w-full rounded bg-gray-800 border border-gray-600 px-3 py-1.5 text-sm text-white" />
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">Max Seq Length</label>
                <select value={maxSeqLength} onChange={(e) => setMaxSeqLength(Number(e.target.value))}
                  className="w-full rounded bg-gray-800 border border-gray-600 px-3 py-1.5 text-sm text-white">
                  <option value={256}>256</option>
                  <option value={512}>512</option>
                  <option value={1024}>1024</option>
                  <option value={2048}>2048</option>
                </select>
              </div>
            </div>
          </div>
        )}
      </fieldset>

      {/* ---- Quantization Config ---- */}
      <fieldset className="rounded border border-gray-700 overflow-hidden">
        <button type="button" onClick={() => toggleSection("quant")}
          className="w-full flex items-center justify-between px-4 py-2.5 bg-gray-800/80 text-sm font-medium text-gray-200 hover:bg-gray-700/80">
          <span>Quantization Configuration</span>
          <span className="text-gray-500 text-xs">{openSections.quant ? "▲" : "▼"}</span>
        </button>
        {openSections.quant && (
          <div className="px-4 py-3 space-y-3 bg-gray-800/30">
            <div>
              <label className="block text-xs text-gray-400 mb-1">Quantization Type</label>
              <select value={quantType} onChange={(e) => setQuantType(e.target.value)}
                className="w-full rounded bg-gray-800 border border-gray-600 px-3 py-1.5 text-sm text-white">
                <option value="awq">AWQ</option>
                <option value="none">None (FP16)</option>
              </select>
            </div>
            {quantType === "awq" && (
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-xs text-gray-400 mb-1">Weight Bits</label>
                  <select value={wBit} onChange={(e) => setWBit(Number(e.target.value))}
                    className="w-full rounded bg-gray-800 border border-gray-600 px-3 py-1.5 text-sm text-white">
                    <option value={4}>4-bit</option>
                    <option value={8}>8-bit</option>
                  </select>
                </div>
                <div>
                  <label className="block text-xs text-gray-400 mb-1">Group Size</label>
                  <select value={qGroupSize} onChange={(e) => setQGroupSize(Number(e.target.value))}
                    className="w-full rounded bg-gray-800 border border-gray-600 px-3 py-1.5 text-sm text-white">
                    <option value={32}>32</option>
                    <option value={64}>64</option>
                    <option value={128}>128</option>
                  </select>
                </div>
              </div>
            )}
          </div>
        )}
      </fieldset>

      {/* ---- Evaluation Config ---- */}
      <fieldset className="rounded border border-gray-700 overflow-hidden">
        <button type="button" onClick={() => toggleSection("eval")}
          className="w-full flex items-center justify-between px-4 py-2.5 bg-gray-800/80 text-sm font-medium text-gray-200 hover:bg-gray-700/80">
          <span>Evaluation Configuration</span>
          <span className="text-gray-500 text-xs">{openSections.eval ? "▲" : "▼"}</span>
        </button>
        {openSections.eval && (
          <div className="px-4 py-3 space-y-3 bg-gray-800/30">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-xs text-gray-400 mb-1">Evaluation Mode</label>
                <select value={evalMode} onChange={(e) => setEvalMode(e.target.value)}
                  className="w-full rounded bg-gray-800 border border-gray-600 px-3 py-1.5 text-sm text-white">
                  <option value="quick">Quick (ROUGE-L)</option>
                  <option value="full">Full (ROUGE-L + MMLU)</option>
                </select>
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">Max New Tokens</label>
                <select value={maxNewTokens} onChange={(e) => setMaxNewTokens(Number(e.target.value))}
                  className="w-full rounded bg-gray-800 border border-gray-600 px-3 py-1.5 text-sm text-white">
                  <option value={128}>128</option>
                  <option value={256}>256</option>
                  <option value={512}>512</option>
                </select>
              </div>
            </div>
          </div>
        )}
      </fieldset>

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
