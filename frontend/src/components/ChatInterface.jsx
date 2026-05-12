import { useState, useEffect, useRef } from "react";
import { getServingStatus, chatCompletions } from "../api/client";

export default function ChatInterface() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [serving, setServing] = useState(null);
  const scrollRef = useRef(null);

  // Poll serving status
  useEffect(() => {
    let active = true;
    async function poll() {
      try {
        const s = await getServingStatus();
        if (active) setServing(s);
      } catch { /* ignore */ }
    }
    poll();
    const id = setInterval(poll, 5000);
    return () => { active = false; clearInterval(id); };
  }, []);

  // Auto-scroll to bottom
  useEffect(() => {
    scrollRef.current?.scrollTo(0, scrollRef.current.scrollHeight);
  }, [messages, loading]);

  const isReady = serving?.status === "running";

  async function handleSend(e) {
    e.preventDefault();
    if (!input.trim() || !isReady) return;

    const userMsg = { role: "user", content: input };
    const history = [...messages, userMsg];
    setMessages(history);
    setInput("");
    setLoading(true);

    try {
      // Stream response from vLLM via backend proxy
      const body = await chatCompletions(history, { stream: true });
      const reader = body.getReader();
      const decoder = new TextDecoder();

      let assistantContent = "";
      setMessages([...history, { role: "assistant", content: "" }]);

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const text = decoder.decode(value, { stream: true });
        // Parse SSE lines: "data: {...}\n\n"
        const lines = text.split("\n").filter((l) => l.startsWith("data: "));
        for (const line of lines) {
          const payload = line.slice(6).trim();
          if (payload === "[DONE]") break;
          try {
            const parsed = JSON.parse(payload);
            const delta = parsed.choices?.[0]?.delta?.content || "";
            assistantContent += delta;
            setMessages((prev) => {
              const updated = [...prev];
              updated[updated.length - 1] = { role: "assistant", content: assistantContent };
              return updated;
            });
          } catch { /* skip malformed chunk */ }
        }
      }
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: `Error: ${err.message || "Failed to get response."}` },
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="max-w-2xl mx-auto flex flex-col h-[500px]">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-white">Chat Interface</h2>
        {serving?.status === "running" && (
          <div className="flex items-center gap-2 text-xs text-gray-300 bg-gray-800 border border-gray-600 rounded px-3 py-1.5">
            <span className="inline-block w-2 h-2 rounded-full bg-green-400 animate-pulse" />
            <span className="text-white font-medium">{serving.model?.split("/").pop()}</span>
            <span className="text-gray-500">|</span>
            <span>{serving.quantization_type || "none"}</span>
          </div>
        )}
      </div>

      {!isReady ? (
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center space-y-2">
            <p className="text-gray-400 text-sm">
              {serving?.status === "deploying"
                ? "Model is deploying… this may take a minute."
                : "No model deployed. Deploy one from the Results tab."}
            </p>
            {serving?.status === "deploying" && (
              <div className="mx-auto w-6 h-6 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin" />
            )}
          </div>
        </div>
      ) : (
        <>
          <div
            ref={scrollRef}
            className="flex-1 overflow-y-auto space-y-3 mb-4 rounded bg-gray-800/50 border border-gray-700 p-4"
          >
            {messages.length === 0 && (
              <p className="text-gray-500 text-sm">Start a conversation...</p>
            )}
            {messages.map((msg, i) => (
              <div
                key={i}
                className={`text-sm rounded px-3 py-2 max-w-[80%] whitespace-pre-wrap ${
                  msg.role === "user"
                    ? "ml-auto bg-indigo-600 text-white"
                    : "mr-auto bg-gray-700 text-gray-200"
                }`}
              >
                {msg.content}
              </div>
            ))}
            {loading && messages[messages.length - 1]?.role !== "assistant" && (
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
