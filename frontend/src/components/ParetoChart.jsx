import { useState, useEffect } from "react";
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { getResults } from "../api/client";

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
    </div>
  );
}

export default function ParetoChart({ onSelectExperiment }) {
  const [data, setData] = useState([]);
  const [error, setError] = useState(null);

  useEffect(() => {
    let active = true;

    async function poll() {
      try {
        const results = await getResults();
        if (active) {
          setData(results);
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
              data={data}
              fill="#818cf8"
              cursor="pointer"
              onClick={(point) => onSelectExperiment?.(point)}
            />
          </ScatterChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}
