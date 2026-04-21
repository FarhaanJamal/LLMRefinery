export default function HowToUse() {
  return (
    <div className="max-w-2xl mx-auto space-y-6 text-sm text-gray-300">
      <h2 className="text-lg font-semibold text-white">How to Use LLM Refinery</h2>

      <div className="space-y-4">
        <section className="rounded bg-gray-800/50 border border-gray-700 p-4 space-y-2">
          <h3 className="text-white font-medium">1. Upload a Dataset</h3>
          <p>
            Go to the <span className="text-indigo-400 font-medium">Upload</span> tab. Select a{" "}
            <code className="bg-gray-700 px-1 rounded text-xs">.jsonl</code> file where each
            line is a JSON object (e.g. with <code className="bg-gray-700 px-1 rounded text-xs">instruction</code> and{" "}
            <code className="bg-gray-700 px-1 rounded text-xs">output</code> fields).
          </p>
        </section>

        <section className="rounded bg-gray-800/50 border border-gray-700 p-4 space-y-2">
          <h3 className="text-white font-medium">2. Configure Your Experiment</h3>
          <ul className="list-disc list-inside space-y-1">
            <li><span className="text-white">Base Model</span> — choose an open-source LLM (e.g. Llama 3 8B)</li>
            <li><span className="text-white">LoRA Rank (r)</span> — adapter size (higher = more capacity, more VRAM)</li>
            <li><span className="text-white">LoRA Alpha</span> — scaling factor (typically 2x rank)</li>
            <li><span className="text-white">Quantization</span> — AWQ or GPTQ for 4-bit compression, or None for full FP16</li>
          </ul>
        </section>

        <section className="rounded bg-gray-800/50 border border-gray-700 p-4 space-y-2">
          <h3 className="text-white font-medium">3. Submit & Wait</h3>
          <p>
            Click <span className="text-indigo-400 font-medium">Upload & Start Experiment</span>.
            The job is queued and processed on the remote GPU pod. The pipeline runs:{" "}
            <span className="text-white">Fine-tune → Quantize → Evaluate</span> — all sequentially
            to stay within 24GB VRAM.
          </p>
        </section>

        <section className="rounded bg-gray-800/50 border border-gray-700 p-4 space-y-2">
          <h3 className="text-white font-medium">4. Analyze Results</h3>
          <p>
            Go to the <span className="text-indigo-400 font-medium">Results</span> tab.
            Each completed experiment appears as a dot on the Pareto chart — X-axis is inference
            speed (tokens/sec), Y-axis is accuracy. Hover for details. Click a dot to select it.
          </p>
        </section>

        <section className="rounded bg-gray-800/50 border border-gray-700 p-4 space-y-2">
          <h3 className="text-white font-medium">5. Deploy & Chat</h3>
          <p>
            Select the best experiment from the chart, then go to the{" "}
            <span className="text-indigo-400 font-medium">Chat</span> tab. The selected model
            will be deployed via vLLM on the GPU pod, and you can interact with it directly.
          </p>
        </section>
      </div>
    </div>
  );
}
