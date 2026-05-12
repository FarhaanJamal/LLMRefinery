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
import { getResults, getServingStatus, deployModel, undeployModel } from "../api/client";

/* ---- Pareto frontier calculation ---- */
function computePareto(points) {
  // Pareto-optimal = not dominated on BOTH accuracy (higher=better) and latency (higher=better)
  const dominated = new Set();
  for (let i = 0; i < points.length; i++) {
    for (let j = 0; j < points.length; j++) {
      if (i === j) continue;
      if (
        points[j].accuracy >= points[i].accuracy &&
        points[j].latency >= points[i].latency &&
        (points[j].accuracy > points[i].accuracy || points[j].latency > points[i].latency)
      ) {
        dominated.add(i);
        break;
      }
    }
  }
  return new Set(
    points.map((_, i) => i).filter((i) => !dominated.has(i))
  );
}

function CustomTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="rounded bg-gray-800 border border-gray-600 px-3 py-2 text-xs text-gray-200 space-y-1">
      <p className="font-semibold text-white">{d.model}</p>
      <p>Accuracy: {d.accuracy.toFixed(3)}</p>
      <p>Latency: {d.latency.toFixed(1)} tok/s</p>
      <p>Quant: {d.quantization_type || "none"}</p>
      <p>LoRA r={d.lora_rank} α={d.lora_alpha}</p>
      <p>Compression: {d.compression_ratio.toFixed(2)}x</p>
      {d.isPareto && <p className="text-green-400 font-semibold">★ Pareto-optimal</p>}
    </div>
  );
}

/* ---- Detail card for selected point ---- */
function DetailCard({ point, serving, onDeploy, onUndeploy, deploying }) {
  if (!point) return null;

  const isDeployed = serving?.status === "running" && serving?.run_id === point.run_id;
  const isDeploying = serving?.status === "deploying" && serving?.run_id === point.run_id;

  return (
    <div className="mt-4 rounded border border-gray-600 bg-gray-800 p-4 text-sm text-gray-200 space-y-2">
      <div className="flex items-center justify-between">
        <h3 className="font-semibold text-white text-base">{point.model}</h3>
        {point.isPareto && (
          <span className="text-xs bg-green-900 text-green-300 px-2 py-0.5 rounded">Pareto-optimal</span>
        )}
      </div>

      <div className="grid grid-cols-2 gap-x-6 gap-y-1 text-xs">
        <p>Accuracy: <span className="text-white">{point.accuracy.toFixed(3)}</span></p>
        <p>Latency: <span className="text-white">{point.latency.toFixed(1)} tok/s</span></p>
        <p>Quantization: <span className="text-white">{point.quantization_type || "none"}</span></p>
        <p>Compression: <span className="text-white">{point.compression_ratio.toFixed(2)}x</span></p>
        <p>LoRA rank: <span className="text-white">{point.lora_rank}</span></p>
        <p>LoRA alpha: <span className="text-white">{point.lora_alpha}</span></p>
        <p>Train time: <span className="text-white">{point.time_to_train.toFixed(0)}s</span></p>
        <p>VRAM peak: <span className="text-white">{point.vram_max_allocated.toFixed(1)} GB</span></p>
      </div>

      <div className="pt-2 flex gap-2">
        {isDeployed ? (
          <button
            onClick={onUndeploy}
            disabled={deploying}
            className="px-4 py-1.5 rounded bg-red-700 hover:bg-red-600 text-white text-xs font-medium disabled:opacity-50"
          >
            Undeploy
          </button>
        ) : (
          <button
            onClick={() => onDeploy(point.run_id)}
            disabled={deploying || isDeploying}
            className="px-4 py-1.5 rounded bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-medium disabled:opacity-50"
          >
            {isDeploying ? "Deploying…" : "Deploy Model"}
          </button>
        )}
        {isDeployed && (
          <span className="text-green-400 text-xs flex items-center gap-1">
            <span className="inline-block w-2 h-2 rounded-full bg-green-400" /> Serving
          </span>
        )}
      </div>
    </div>
  );
}

export default function ParetoChart({ onSelectExperiment }) {
  const [data, setData] = useState([]);
  const [selected, setSelected] = useState(null);
  const [serving, setServing] = useState(null);
  const [deploying, setDeploying] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    let active = true;

    async function poll() {
      try {
        const [results, status] = await Promise.all([getResults(), getServingStatus()]);
        if (active) {
          setData(results);
          setServing(status);
          setError(null);
        }
      } catch {
        if (active) setError("Failed to fetch results.");
      }
    }

    poll();
    const id = setInterval(poll, 10000);
    return () => { active = false; clearInterval(id); };
  }, []);

  // Compute Pareto set
  const paretoSet = computePareto(data);
  const enriched = data.map((d, i) => ({ ...d, isPareto: paretoSet.has(i) }));

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

  function handleClick(point) {
    setSelected(point);
    onSelectExperiment?.(point);
  }

  return (
    <div className="max-w-3xl mx-auto">
      <h2 className="text-lg font-semibold text-white mb-4">
        Pareto Frontier — Accuracy vs. Latency
      </h2>

      {error && (
        <p className="text-red-400 text-sm mb-2">{error}</p>
      )}

      {data.length === 0 ? (
        <p className="text-gray-400 text-sm">
          No experiment results yet. Submit a job from the Upload tab.
        </p>
      ) : (
        <>
          <ResponsiveContainer width="100%" height={400}>
            <ScatterChart margin={{ top: 10, right: 20, bottom: 20, left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis
                dataKey="latency"
                name="Latency"
                unit=" tok/s"
                type="number"
                stroke="#9ca3af"
                label={{ value: "Inference Latency (tok/s)", position: "bottom", fill: "#9ca3af" }}
              />
              <YAxis
                dataKey="accuracy"
                name="Accuracy"
                type="number"
                stroke="#9ca3af"
                label={{ value: "Accuracy", angle: -90, position: "insideLeft", fill: "#9ca3af" }}
              />
              <Tooltip content={<CustomTooltip />} />
              <Scatter
                data={enriched}
                cursor="pointer"
                onClick={(point) => handleClick(point)}
              >
                {enriched.map((entry, i) => {
                  const isDeployed = serving?.status === "running" && serving?.run_id === entry.run_id;
                  let fill = "#6b7280"; // gray — dominated
                  if (entry.isPareto) fill = "#818cf8"; // indigo — Pareto
                  if (isDeployed) fill = "#34d399"; // green — deployed
                  return <Cell key={i} fill={fill} r={isDeployed ? 8 : 6} />;
                })}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>

          <div className="flex gap-4 text-xs text-gray-400 mt-1 justify-center">
            <span className="flex items-center gap-1"><span className="inline-block w-3 h-3 rounded-full bg-indigo-400" /> Pareto-optimal</span>
            <span className="flex items-center gap-1"><span className="inline-block w-3 h-3 rounded-full bg-gray-500" /> Dominated</span>
            <span className="flex items-center gap-1"><span className="inline-block w-3 h-3 rounded-full bg-green-400" /> Deployed</span>
          </div>

          <DetailCard
            point={selected}
            serving={serving}
            onDeploy={handleDeploy}
            onUndeploy={handleUndeploy}
            deploying={deploying}
          />
        </>
      )}
    </div>
  );
}
