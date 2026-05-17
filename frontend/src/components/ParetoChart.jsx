import { useState, useEffect } from "react";
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import { getResults, getServingStatus, deployModel, undeployModel, deleteExperiment, subscribeEvents } from "../api/client";

/* ---- Axis options ---- */
const AXIS_OPTIONS = [
  { key: "accuracy",           label: "Accuracy",       unit: "",       fmt: (v) => v?.toFixed(3) },
  { key: "latency",            label: "Latency",        unit: " tok/s", fmt: (v) => v?.toFixed(1) },
  { key: "compression_ratio",  label: "Compression",    unit: "x",      fmt: (v) => v?.toFixed(2) },
  { key: "vram_max_allocated", label: "VRAM Peak",      unit: " GB",    fmt: (v) => v?.toFixed(3) },
];

function axisLabel(opt) {
  return `${opt.label}${opt.unit ? ` (${opt.unit.trim()})` : ""}`;
}

/* ---- Tooltip ---- */
function CustomTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="rounded bg-gray-800 border border-gray-600 px-3 py-2 text-xs text-gray-200 space-y-1">
      <p className="font-semibold text-white">{d.model?.split("/").pop()}</p>
      <p>Accuracy: {d.accuracy?.toFixed(3)}</p>
      <p>Latency: {d.latency?.toFixed(1)} tok/s</p>
      <p>Quant: {d.quantization_type || "none"}{d.w_bit ? ` (${d.w_bit}-bit)` : ""}</p>
      <p>LoRA r={d.lora_rank} α={d.lora_alpha}</p>
      <p>Steps: {d.max_steps} | LR: {d.learning_rate}</p>
      <p>Compression: {d.compression_ratio?.toFixed(2)}x</p>
      <p>VRAM: {d.vram_max_allocated?.toFixed(3)} GB</p>
    </div>
  );
}

/* ---- Comparison table ---- */
const COMPARE_FIELDS = [
  { key: "model",              label: "Model",        fmt: (v) => v?.split("/").pop() },
  { key: "quantization_type",  label: "Quantization", fmt: (v) => v || "none" },
  { key: "accuracy",           label: "Accuracy",     fmt: (v) => v?.toFixed(3) },
  { key: "latency",            label: "Latency",      fmt: (v) => `${v?.toFixed(1)} tok/s` },
  { key: "compression_ratio",  label: "Compression",  fmt: (v) => `${v?.toFixed(2)}x` },
  { key: "lora_rank",          label: "LoRA rank",    fmt: (v) => v },
  { key: "lora_alpha",         label: "LoRA alpha",   fmt: (v) => v },
  { key: "lora_dropout",       label: "LoRA dropout", fmt: (v) => v },
  { key: "num_train_epochs",   label: "Epochs",       fmt: (v) => (v && Number(v) > 0) ? v : "—" },
  { key: "max_steps",          label: "Max steps",    fmt: (v) => v },
  { key: "learning_rate",      label: "Learning rate",fmt: (v) => v },
  { key: "batch_size",         label: "Batch size",   fmt: (v) => v },
  { key: "lr_scheduler_type",  label: "LR scheduler", fmt: (v) => v },
  { key: "max_seq_length",     label: "Max seq len",  fmt: (v) => v },
  { key: "w_bit",              label: "Quant bits",   fmt: (v) => v ? `${v}-bit` : "—" },
  { key: "q_group_size",       label: "Group size",   fmt: (v) => v || "—" },
  { key: "time_to_train",      label: "Train time",   fmt: (v) => `${v?.toFixed(0)}s` },
  { key: "vram_max_allocated", label: "VRAM peak",    fmt: (v) => `${v?.toFixed(3)} GB` },
  { key: "eval_mode",          label: "Eval mode",    fmt: (v) => v },
];

function bestIdx(items, key, higher) {
  if (items.length === 0) return -1;
  let best = 0;
  for (let i = 1; i < items.length; i++) {
    const a = items[i][key], b = items[best][key];
    if (typeof a === "number" && typeof b === "number") {
      if (higher ? a > b : a < b) best = i;
    }
  }
  return best;
}

function CompareTable({ items, onRemove }) {
  if (items.length === 0) return null;
  return (
    <div className="mt-4 overflow-x-auto">
      <table className="w-full text-xs text-left border-collapse">
        <thead>
          <tr className="border-b border-gray-600">
            <th className="py-2 px-2 text-gray-400 font-medium">Metric</th>
            {items.map((item, i) => (
              <th key={i} className="py-2 px-2 text-gray-300 font-medium">
                <div className="flex items-center gap-1">
                  <span className="inline-block w-2.5 h-2.5 rounded-full" style={{ backgroundColor: item._color }} />
                  {item.model?.split("/").pop()}
                  <button onClick={() => onRemove(item.run_id)} className="ml-1 text-gray-500 hover:text-red-400">✕</button>
                </div>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {COMPARE_FIELDS.map((field) => {
            const higherBetter = ["accuracy", "latency", "compression_ratio"].includes(field.key);
            const lowerBetter = ["time_to_train", "vram_max_allocated"].includes(field.key);
            const bIdx = (higherBetter || lowerBetter) ? bestIdx(items, field.key, higherBetter) : -1;
            return (
              <tr key={field.key} className="border-b border-gray-700/50">
                <td className="py-1.5 px-2 text-gray-400">{field.label}</td>
                {items.map((item, i) => (
                  <td key={i} className={`py-1.5 px-2 ${i === bIdx ? "text-green-400 font-semibold" : "text-white"}`}>
                    {field.fmt(item[field.key])}
                  </td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

/* ---- Main component ---- */
export default function ParetoChart({ onSelectExperiment }) {
  const [data, setData] = useState([]);
  const [compareIds, setCompareIds] = useState(new Set());
  const [serving, setServing] = useState(null);
  const [deploying, setDeploying] = useState(false);
  const [error, setError] = useState(null);
  const [xAxis, setXAxis] = useState("latency");
  const [yAxis, setYAxis] = useState("accuracy");

  useEffect(() => {
    let active = true;
    async function fetchAll() {
      try {
        const [results, status] = await Promise.all([getResults(), getServingStatus()]);
        if (active) { setData(results); setServing(status); setError(null); }
      } catch {
        if (active) setError("Failed to fetch results.");
      }
    }
    fetchAll();

    // SSE-driven updates
    const es = subscribeEvents((event) => {
      if (!active) return;
      if (event.type === "job_status" && event.status === "completed") {
        fetchAll();
      } else if (event.type === "deploy_status") {
        getServingStatus().then((s) => { if (active) setServing(s); }).catch(() => {});
      }
    });

    // Fallback poll every 30s
    const fallback = setInterval(fetchAll, 30000);
    return () => { active = false; es.close(); clearInterval(fallback); };
  }, []);

  async function handleDeploy(runId) {
    try {
      setDeploying(true);
      await deployModel(runId);
      setServing({ status: "deploying", run_id: runId });
    } catch (err) {
      setError(err.message || "Deploy failed.");
    } finally {
      setDeploying(false);
    }
  }

  async function handleUndeploy() {
    try {
      setDeploying(true);
      await undeployModel();
      setServing({ status: "stopping" });
    } catch (err) {
      setError(err.message || "Undeploy failed.");
    } finally {
      setDeploying(false);
    }
  }

  function toggleCompare(runId) {
    setCompareIds((prev) => {
      const next = new Set(prev);
      next.has(runId) ? next.delete(runId) : next.add(runId);
      return next;
    });
  }

  async function handleDelete(runId) {
    if (!confirm("Delete this experiment? This cannot be undone.")) return;
    try {
      await deleteExperiment(runId);
      setData((prev) => prev.filter((d) => d.run_id !== runId));
      setCompareIds((prev) => {
        const next = new Set(prev);
        next.delete(runId);
        return next;
      });
    } catch (err) {
      setError(err.message || "Delete failed.");
    }
  }

  // Prevent same axis on both dropdowns
  const xOpts = AXIS_OPTIONS.filter((o) => o.key !== yAxis);
  const yOpts = AXIS_OPTIONS.filter((o) => o.key !== xAxis);
  const xMeta = AXIS_OPTIONS.find((o) => o.key === xAxis);
  const yMeta = AXIS_OPTIONS.find((o) => o.key === yAxis);

  const COMPARE_COLORS = ["#818cf8", "#f472b6", "#facc15", "#34d399", "#fb923c", "#a78bfa"];
  const compareItems = data
    .filter((e) => compareIds.has(e.run_id))
    .map((e, i) => ({ ...e, _color: COMPARE_COLORS[i % COMPARE_COLORS.length] }));

  return (
    <div className="max-w-4xl mx-auto">
      {error && <p className="text-red-400 text-sm mb-2">{error}</p>}

      {data.length === 0 ? (
        <p className="text-gray-400 text-sm">
          No experiment results yet. Submit a job from the Upload tab.
        </p>
      ) : (
        <>
          {/* Axis selectors */}
          <div className="flex items-center gap-4 mb-4">
            <h2 className="text-lg font-semibold text-white">Results</h2>
            <div className="flex items-center gap-2 ml-auto text-xs">
              <label className="text-gray-400">X:</label>
              <select
                value={xAxis}
                onChange={(e) => setXAxis(e.target.value)}
                className="rounded bg-gray-800 border border-gray-600 px-2 py-1 text-white text-xs"
              >
                {xOpts.map((o) => <option key={o.key} value={o.key}>{o.label}</option>)}
              </select>
              <label className="text-gray-400 ml-2">Y:</label>
              <select
                value={yAxis}
                onChange={(e) => setYAxis(e.target.value)}
                className="rounded bg-gray-800 border border-gray-600 px-2 py-1 text-white text-xs"
              >
                {yOpts.map((o) => <option key={o.key} value={o.key}>{o.label}</option>)}
              </select>
            </div>
          </div>

          {/* Chart */}
          <ResponsiveContainer width="100%" height={380}>
            <ScatterChart margin={{ top: 10, right: 20, bottom: 20, left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis
                dataKey={xAxis}
                name={xMeta.label}
                unit={xMeta.unit}
                type="number"
                stroke="#9ca3af"
                label={{ value: axisLabel(xMeta), position: "bottom", fill: "#9ca3af" }}
              />
              <YAxis
                dataKey={yAxis}
                name={yMeta.label}
                unit={yMeta.unit}
                type="number"
                stroke="#9ca3af"
                label={{ value: axisLabel(yMeta), angle: -90, position: "insideLeft", fill: "#9ca3af" }}
              />
              <Tooltip content={<CustomTooltip />} />
              <Scatter data={data} cursor="pointer" onClick={(point) => onSelectExperiment?.(point)}>
                {data.map((entry, i) => {
                  const isDeployed = serving?.status === "running" && serving?.run_id === entry.run_id;
                  const isCompared = compareIds.has(entry.run_id);
                  let fill = "#6b7280";
                  if (isCompared) {
                    const ci = compareItems.findIndex((c) => c.run_id === entry.run_id);
                    if (ci >= 0) fill = compareItems[ci]._color;
                  }
                  if (isDeployed) fill = "#34d399";
                  return (
                    <Cell
                      key={i}
                      fill={fill}
                      r={isDeployed ? 8 : isCompared ? 7 : 5}
                      stroke={isCompared ? "#fff" : "none"}
                      strokeWidth={isCompared ? 2 : 0}
                    />
                  );
                })}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>

          {/* Model list with compare + deploy */}
          <div className="mt-4 rounded border border-gray-700 bg-gray-800/50 p-3">
            <p className="text-xs text-gray-400 mb-2">Select models to compare:</p>
            <div className="space-y-1 max-h-48 overflow-y-auto">
              {data.map((entry) => {
                const checked = compareIds.has(entry.run_id);
                const isDeployed = serving?.status === "running" && serving?.run_id === entry.run_id;
                const isDeploying = serving?.status === "deploying" && serving?.run_id === entry.run_id;
                const anotherDeployed = serving?.status === "running" && serving?.run_id !== entry.run_id;
                const anotherDeploying = serving?.status === "deploying" && serving?.run_id !== entry.run_id;
                return (
                  <div key={entry.run_id} className="flex items-center gap-2 text-xs hover:bg-gray-700/50 rounded px-2 py-1.5">
                    <input
                      type="checkbox"
                      checked={checked}
                      onChange={() => toggleCompare(entry.run_id)}
                      className="accent-indigo-500"
                    />
                    <span className="text-white flex-1 min-w-0 truncate">
                      {entry.model?.split("/").pop()}
                      <span className="text-gray-500 ml-1">— {entry.quantization_type || "none"}, r={entry.lora_rank}</span>
                    </span>
                    {isDeployed ? (
                      <>
                        <span className="text-green-400 flex items-center gap-1 text-[10px]">
                          <span className="inline-block w-2 h-2 rounded-full bg-green-400" /> Serving
                        </span>
                        <button
                          onClick={() => handleUndeploy()}
                          disabled={deploying}
                          className="px-2 py-0.5 rounded bg-red-700 hover:bg-red-600 text-white text-[10px] font-medium disabled:opacity-50"
                        >
                          Undeploy
                        </button>
                      </>
                    ) : (
                      <button
                        onClick={() => handleDeploy(entry.run_id)}
                        disabled={deploying || isDeploying || anotherDeployed || anotherDeploying}
                        className="px-2 py-0.5 rounded bg-indigo-600 hover:bg-indigo-500 text-white text-[10px] font-medium disabled:opacity-50"
                        title={anotherDeployed || anotherDeploying ? "Undeploy the current model first" : ""}
                      >
                        {isDeploying ? "Deploying…" : "Deploy"}
                      </button>
                    )}
                    <button
                      onClick={() => handleDelete(entry.run_id)}
                      disabled={isDeployed || isDeploying}
                      className="px-2 py-0.5 rounded bg-gray-700 hover:bg-red-700 text-gray-400 hover:text-white text-[10px] font-medium disabled:opacity-30 disabled:cursor-not-allowed disabled:hover:bg-gray-700 disabled:hover:text-gray-400"
                      title={isDeployed || isDeploying ? "Undeploy the model before deleting" : "Delete experiment"}
                    >
                      Delete
                    </button>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Compare table */}
          {compareItems.length > 0 && (
            <div className="mt-4 rounded border border-gray-600 bg-gray-800 p-4">
              <div className="flex items-center justify-between mb-2">
                <h3 className="text-sm font-semibold text-white">Comparison ({compareItems.length} models)</h3>
                <button
                  onClick={() => setCompareIds(new Set())}
                  className="text-xs text-gray-400 hover:text-white underline"
                >
                  Clear all
                </button>
              </div>
              <CompareTable items={compareItems} onRemove={(id) => toggleCompare(id)} />
            </div>
          )}
        </>
      )}
    </div>
  );
}
