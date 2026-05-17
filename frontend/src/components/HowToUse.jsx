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
            <li><span className="text-white">Base Model</span> — choose an open-source LLM (e.g. TinyLlama 1.1B, Llama 3 8B, Mistral 7B, Gemma 7B)</li>
            <li><span className="text-white">LoRA Rank (r)</span> — adapter size (higher = more capacity, more VRAM)</li>
            <li><span className="text-white">LoRA Alpha</span> — scaling factor (typically 2x rank)</li>
            <li><span className="text-white">Quantization</span> — AWQ for 4-bit compression, or None to skip quantization</li>
            <li><span className="text-white">Eval Mode</span> — Quick (accuracy on test split) or Full (adds MMLU benchmarks)</li>
          </ul>
        </section>

        <section className="rounded bg-gray-800/50 border border-gray-700 p-4 space-y-2">
          <h3 className="text-white font-medium">3. Submit & Track Progress</h3>
          <p>
            Click <span className="text-indigo-400 font-medium">Upload & Start Experiment</span>.
            A progress bar appears showing each pipeline stage in real time:{" "}
            <span className="text-white">Queued → Training → Quantizing → Evaluating → Done</span>.
            You can submit multiple jobs — each gets its own tracker. Dismiss any tracker with the ✕ button.
          </p>
        </section>

        <section className="rounded bg-gray-800/50 border border-gray-700 p-4 space-y-2">
          <h3 className="text-white font-medium">4. Analyze & Compare Results</h3>
          <p>
            Go to the <span className="text-indigo-400 font-medium">Results</span> tab.
            Each completed experiment appears as a dot on the scatter chart. Use the X/Y axis
            dropdowns to plot any combination of accuracy, latency, compression ratio, and VRAM usage.
          </p>
          <p>
            Check the boxes next to models to open a side-by-side{" "}
            <span className="text-white">comparison table</span> — best values are highlighted in green.
          </p>
        </section>

        <section className="rounded bg-gray-800/50 border border-gray-700 p-4 space-y-2">
          <h3 className="text-white font-medium">5. Deploy & Chat</h3>
          <p>
            Click <span className="text-indigo-400 font-medium">Deploy</span> next to any model
            in the Results tab. Only one model can be served at a time — undeploy the current one
            to switch. Once deployed, go to the{" "}
            <span className="text-indigo-400 font-medium">Chat</span> tab to interact with
            your fine-tuned model via streaming chat (powered by vLLM on the GPU pod).
          </p>
        </section>

        <section className="rounded bg-gray-800/50 border border-gray-700 p-4 space-y-2">
          <h3 className="text-white font-medium">6. Clean Up</h3>
          <p>
            Delete experiments you no longer need using the{" "}
            <span className="text-white">Delete</span> button in the Results tab.
            This removes the MLflow run, MinIO artifacts, and all MongoDB records.
            Deployed models must be undeployed before they can be deleted.
          </p>
        </section>
      </div>
    </div>
  );
}
