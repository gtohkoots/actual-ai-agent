import { useEffect, useMemo, useRef, useState } from "react";

import { Bot, ChevronRight, LoaderCircle, SendHorizontal, Sparkles, WandSparkles } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { sendChatMessage } from "../chat/api";
import { createWelcomeMessage, summarizeContext } from "../chat/mockResponder";

function ChatPanel({ card }) {
  const [messages, setMessages] = useState(() => [createWelcomeMessage(card)]);
  const [draft, setDraft] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [conversationId, setConversationId] = useState(null);
  const [errorMessage, setErrorMessage] = useState("");
  const feedRef = useRef(null);

  const contextLines = useMemo(() => summarizeContext(card), [card]);
  const chatContext = useMemo(
    () => ({
      selected_tab: "AI Assistant",
      account_name: card.context.card,
      card_label: card.name,
      start_date: card.context.windowStart,
      end_date: card.context.windowEnd,
      focus_category: card.context.focus,
      focus_payee: card.summary.topMerchant,
    }),
    [card]
  );

  useEffect(() => {
    setMessages([createWelcomeMessage(card)]);
    setDraft("");
    setIsSending(false);
    setConversationId(null);
    setErrorMessage("");
  }, [card]);

  useEffect(() => {
    const el = feedRef.current;
    if (el) {
      el.scrollTop = el.scrollHeight;
    }
  }, [messages, isSending]);

  async function handleSubmit(event) {
    event.preventDefault();
    const text = draft.trim();
    if (!text || isSending) return;
    await submitMessage(text);
  }

  function handleQuickPrompt(prompt) {
    void submitMessage(prompt);
  }

  async function submitMessage(text) {
    const userMessage = {
      id: `user-${Date.now()}`,
      role: "user",
      content: text,
    };

    setMessages((current) => [...current, userMessage]);
    setDraft("");
    setIsSending(true);

    const assistantMessage = {
      id: `assistant-${Date.now()}`,
      role: "assistant",
      status: "thinking",
      content: "Retrieving relevant card context and historical signals...",
      sources: [],
      actions: [],
    };

    setMessages((current) => [...current, assistantMessage]);

    try {
      const response = await sendChatMessage({
        message: text,
        conversationId,
        history: messages,
        context: chatContext,
      });

      setConversationId(response.conversation_id || null);
      setMessages((current) =>
        current.map((message) =>
          message.id === assistantMessage.id
            ? {
                ...message,
                status: "ready",
                content: response.content,
                sources: response.sources,
                actions: response.actions,
                facts: response.facts,
                retrievalStrategy: response.retrieval_strategy,
              }
            : message
        )
      );
      setErrorMessage("");
    } catch (error) {
      setMessages((current) =>
        current.map((message) =>
          message.id === assistantMessage.id
            ? {
                ...message,
                status: "ready",
                content: "I couldn’t reach the backend chat endpoint. Make sure the FastAPI server is running on `http://127.0.0.1:8000`.",
                sources: [],
                actions: [],
              }
            : message
        )
      );
      setErrorMessage(error instanceof Error ? error.message : "Chat request failed");
    } finally {
      setIsSending(false);
    }
  }

  function handleActionChip(action) {
    if (!isSending) {
      void submitMessage(action);
    }
  }

  return (
    <aside className="panel chat-panel">
      <div className="panel-header">
        <div>
          <p className="section-label">AI Copilot</p>
          <h3>Context-aware finance chat</h3>
        </div>
        <span className="panel-note">
          <Sparkles size={14} /> Backend connected
        </span>
      </div>

      {errorMessage ? (
        <div className="chat-error" role="alert">
          {errorMessage}
        </div>
      ) : null}

      <div className="chat-context">
        {contextLines.map((line) => (
          <span key={line} className="chat-context-chip">
            {line}
          </span>
        ))}
      </div>

      <div className="chat-toolbar">
        {card.quickPrompts.map((prompt) => (
          <button key={prompt} className="suggestion-chip" type="button" onClick={() => handleQuickPrompt(prompt)}>
            <WandSparkles size={14} />
            {prompt}
          </button>
        ))}
      </div>

      <div className="chat-feed" ref={feedRef}>
        {messages.map((message) => (
          <article key={message.id} className={`chat-message ${message.role}`}>
            <div className="chat-message-head">
              <div className="chat-role">
                <span className="chat-avatar">{message.role === "assistant" ? <Bot size={14} /> : "You"}</span>
                <strong>{message.role === "assistant" ? "Finance Copilot" : "You"}</strong>
              </div>
              {message.status === "thinking" ? (
                <span className="chat-status">
                  <LoaderCircle size={14} className="spin" />
                  Thinking
                </span>
              ) : null}
            </div>

            <div className="chat-content">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
            </div>

            {message.sources?.length ? (
              <div className="chat-sources">
                <div className="chat-subtitle">Sources</div>
                <div className="source-grid">
                  {message.sources.map((source) => (
                    <div key={`${message.id}-${source.label}`} className="source-card">
                      <strong>{source.label}</strong>
                      <span>{source.detail}</span>
                    </div>
                  ))}
                </div>
              </div>
            ) : null}

            {message.actions?.length ? (
              <div className="chat-actions">
                {message.actions.map((action) => (
                  <button
                    key={`${message.id}-${action}`}
                    className="action-chip"
                    type="button"
                    onClick={() => handleActionChip(action)}
                  >
                    {action}
                    <ChevronRight size={14} />
                  </button>
                ))}
              </div>
            ) : null}
          </article>
        ))}
      </div>

      <form className="chat-form" onSubmit={handleSubmit}>
        <label className="sr-only" htmlFor="chatInput">
          Chat input
        </label>
        <textarea
          id="chatInput"
          rows="3"
          placeholder="Ask about the selected card, a merchant, a spike, or a category trend..."
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
        />
        <div className="chat-form-footer">
          <span className="panel-note">Markdown responses and citations will render here.</span>
          <button className="primary-button" type="submit" disabled={isSending}>
            <SendHorizontal size={16} />
            Send
          </button>
        </div>
      </form>
    </aside>
  );
}

export default ChatPanel;
