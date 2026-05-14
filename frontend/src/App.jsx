import { useState } from "react";
import UploadForm from "./components/UploadForm";
import ParetoChart from "./components/ParetoChart";
import ChatInterface from "./components/ChatInterface";
import HowToUse from "./components/HowToUse";

const TABS = ["Upload", "Results", "Chat", "How to Use"];

function App() {
  const [activeTab, setActiveTab] = useState("Upload");
  const [selectedExperiment, setSelectedExperiment] = useState(null);

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Header */}
      <header className="border-b border-gray-700 px-6 py-4">
        <h1 className="text-xl font-bold tracking-tight">LLM Refinery</h1>
        <p className="text-sm text-gray-400">Fine-tune, quantize, evaluate, deploy</p>
      </header>

      {/* Tabs */}
      <nav className="flex gap-1 px-6 pt-4">
        {TABS.map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`px-4 py-2 text-sm font-medium rounded-t transition-colors ${
              activeTab === tab
                ? "bg-gray-800 text-white border-b-2 border-indigo-500"
                : "text-gray-400 hover:text-gray-200"
            }`}
          >
            {tab}
          </button>
        ))}
      </nav>

      {/* Content — all tabs stay mounted so SSE listeners remain active */}
      <main className="px-6 py-8">
        <div style={{ display: activeTab === "Upload" ? undefined : "none" }}>
          <UploadForm />
        </div>
        <div style={{ display: activeTab === "Results" ? undefined : "none" }}>
          <ParetoChart onSelectExperiment={setSelectedExperiment} />
        </div>
        <div style={{ display: activeTab === "Chat" ? undefined : "none" }}>
          <ChatInterface />
        </div>
        {activeTab === "How to Use" && <HowToUse />}
      </main>
    </div>
  );
}

export default App;
