import { useState } from "react";

export default function ChatInterface({ experiment }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  // Placeholder: will be connected to vLLM endpoint in Phase 4
  const endpoint = experiment?.model_artifact_path
    ? `http://<tailscale-ip>:8000/v1/chat/completions`
    : null;

  async function handleSend(e) {
    e.preventDefault();
    if (!input.trim() || !endpoint) return;

    const userMsg = { role: "user", content: input };
    const updated = [...messages, userMsg];
    setMessages(updated);
    setInput("");
    setLoading(true);

    try {
      // Phase 4: replace with actual fetch to vLLM endpoint
      const assistantMsg = {
        role: "assistant",
        content: "[Model not deployed yet — deploy from the Results tab first]",
      };
      setMessages([...updated, assistantMsg]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="max-w-2xl mx-auto flex flex-col h-[500px]">
      <h2 className="text-lg font-semibold text-white mb-4">Chat Interface</h2>

      {!experiment ? (
        <p className="text-gray-400 text-sm">
          Select an experiment from the Results tab and click "Deploy" to start chatting.
        </p>
      ) : (
        <>
          <div className="flex-1 overflow-y-auto space-y-3 mb-4 rounded bg-gray-800/50 border border-gray-700 p-4">
            {messages.length === 0 && (
              <p className="text-gray-500 text-sm">Start a conversation...</p>
            )}
            {messages.map((msg, i) => (
              <div
                key={i}
                className={`text-sm rounded px-3 py-2 max-w-[80%] ${
                  msg.role === "user"
                    ? "ml-auto bg-indigo-600 text-white"
                    : "mr-auto bg-gray-700 text-gray-200"
                }`}
              >
                {msg.content}
              </div>
            ))}
            {loading && (
              <div className="mr-auto bg-gray-700 text-gray-400 text-sm rounded px-3 py-2">
                Thinking...
              </div>
            )}
          </div>

          <form onSubmit={handleSend} className="flex gap-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Type a message..."
              className="flex-1 rounded bg-gray-800 border border-gray-600 px-3 py-2 text-sm text-white placeholder-gray-500"
            />
            <button
              type="submit"
              disabled={loading || !input.trim()}
              className="rounded bg-indigo-600 px-4 py-2 text-sm font-semibold text-white
                         hover:bg-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Send
            </button>
          </form>
        </>
      )}
    </div>
  );
}
