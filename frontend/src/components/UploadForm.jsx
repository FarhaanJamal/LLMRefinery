import { useState } from "react";
import { uploadDataset, startExperiment } from "../api/client";

const MODELS = [
  "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "meta-llama/Meta-Llama-3-8B",
  "mistralai/Mistral-7B-v0.1",
  "google/gemma-7b",
];

export default function UploadForm() {
  const [file, setFile] = useState(null);
  const [model, setModel] = useState(MODELS[0]);
  const [r, setR] = useState(16);
  const [alpha, setAlpha] = useState(32);
  const [quantType, setQuantType] = useState("awq");
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(false);

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
        params: { r, alpha, quant_type: quantType },
        dataset_path: uploadRes.s3_path,
      });
      setStatus({
        type: "success",
        msg: `Job queued! ID: ${experimentRes.job_id} (${uploadRes.row_count} rows uploaded)`,
      });
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
          <option value="gptq">GPTQ (4-bit)</option>
          <option value="none">None (FP16)</option>
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
    </form>
  );
}
