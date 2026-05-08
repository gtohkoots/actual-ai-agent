import { useEffect, useMemo, useRef, useState } from "react";

import { Bot, ChevronRight, LoaderCircle, SendHorizontal, Sparkles, WandSparkles } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { deleteChatConversation, fetchChatConversation, fetchChatConversations, sendChatMessage } from "../chat/api";
import { createWelcomeMessage } from "../chat/mockResponder";
import {
  deletePlannerConversation,
  fetchPlannerConversation,
  fetchPlannerConversations,
  sendPlannerMessage,
} from "../planner/api";

const CONTEXT_TABS = [
  { id: "card", label: "Card" },
  { id: "window", label: "Window" },
  { id: "focus", label: "Focus" },
];

function createPlannerWelcomeMessage(card, analysisWindow) {
  return {
    id: `planner-welcome-${card?.id || "default"}`,
    role: "assistant",
    status: "ready",
    content:
      `You're in the **planner workspace**. I can help review budgets, analyze recent spending, draft a savings-aware budget, revise that draft, and save it once you approve.\n\n` +
      `The current window is **${analysisWindow?.label || card?.context?.dateRange || "the active budgeting window"}**. Ask me to create a budget, review spending, revise a draft, or approve a plan.`,
    sources: [],
    actions: [
      "Create a budget starting today for a month and save $500",
      "Review spending for last month",
      "Review my current budget",
    ],
  };
}

function mapThreadMessages(thread, fallbackMessage) {
  return (thread.messages || []).length
    ? thread.messages.map((message) => ({
        id: `${thread.conversation_id}-${message.created_at || message.role}-${message.role}`,
        role: message.role,
        content: message.content,
        createdAt: message.created_at,
      }))
    : [fallbackMessage];
}

function ChatPanel({
  card,
  analysisWindow,
  mode = "legacy",
  seedMessage = "",
  seedMessageId = "",
  onSeedConsumed = () => {},
}) {
  const isPlannerMode = mode === "planner";
  const initialWelcomeMessage = useMemo(
    () => (isPlannerMode ? createPlannerWelcomeMessage(card, analysisWindow) : createWelcomeMessage(card)),
    [analysisWindow, card, isPlannerMode]
  );
  const [messages, setMessages] = useState(() => [initialWelcomeMessage]);
  const [draft, setDraft] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [conversationId, setConversationId] = useState(null);
  const [errorMessage, setErrorMessage] = useState("");
  const [isThreadReady, setIsThreadReady] = useState(false);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [activeContextTab, setActiveContextTab] = useState("card");
  const [recentThreads, setRecentThreads] = useState([]);
  const [isLoadingThreads, setIsLoadingThreads] = useState(false);
  const [historyRefreshToken, setHistoryRefreshToken] = useState(0);
  const feedRef = useRef(null);
  const conversationIdRef = useRef(null);
  const consumedSeedRef = useRef("");
  const storageKey = useMemo(
    () => `finance-agent:${isPlannerMode ? "planner" : "chat"}:conversation:${card.context.accountPid}`,
    [card.context.accountPid, isPlannerMode]
  );

  const chatApi = useMemo(
    () => ({
      sendMessage: isPlannerMode ? sendPlannerMessage : sendChatMessage,
      fetchConversation: isPlannerMode ? fetchPlannerConversation : fetchChatConversation,
      fetchConversations: isPlannerMode ? fetchPlannerConversations : fetchChatConversations,
      deleteConversation: isPlannerMode ? deletePlannerConversation : deleteChatConversation,
    }),
    [isPlannerMode]
  );

  const chatContext = useMemo(
    () =>
      isPlannerMode
        ? {
            selected_tab: "budget",
            account_pid: card.context.accountPid,
            account_name: card.context.accountName || card.context.card,
            card_label: card.name,
            start_date: analysisWindow?.start || card.context.windowStart,
            end_date: analysisWindow?.end || card.context.windowEnd,
          }
        : {
            selected_tab: activeContextTab,
            account_pid: card.context.accountPid,
            account_name: card.context.accountName || card.context.card,
            card_label: card.name,
            start_date: analysisWindow?.start || card.context.windowStart,
            end_date: analysisWindow?.end || card.context.windowEnd,
            focus_category: card.context.focus,
            focus_payee: card.summary.topMerchant,
          },
    [activeContextTab, analysisWindow?.end, analysisWindow?.start, card, isPlannerMode]
  );

  const activeContextDetails = useMemo(() => {
    if (isPlannerMode) {
      return {
        title: "Planner workspace",
        description: "Build, revise, and approve budgets directly from chat using the planner agent.",
        prompts: [
          "Create a budget starting today for a month and save $500",
          "Review spending for last month",
          "Review my current budget",
        ],
      };
    }

    if (activeContextTab === "window") {
      return {
        title: analysisWindow?.label || card.context.dateRange,
        description: "Adjust the calendar to change the analysis window for the dashboard and the assistant.",
        prompts: [
          `Summarize this window for ${card.name}`,
          "Compare this window to the prior period",
          "What changed most in this window?",
        ],
      };
    }

    if (activeContextTab === "focus") {
      return {
        title: `${card.summary.topCategory} · ${card.summary.topMerchant}`,
        description: "Focus on the strongest spending signal or the merchant driving the current card activity.",
        prompts: [
          `Explain ${card.summary.topCategory} spend`,
          `Why is ${card.summary.topMerchant} so prominent?`,
          "Look for unusual or recurring charges",
        ],
      };
    }

    return {
      title: card.name,
      description: `Current balance ${card.summary.totalSpend} and cycle spend anchored to the selected card.`,
      prompts: card.quickPrompts.slice(0, 3),
    };
  }, [activeContextTab, analysisWindow?.label, card, isPlannerMode]);

  useEffect(() => {
    let cancelled = false;

    async function restoreConversation() {
      setMessages([initialWelcomeMessage]);
      setDraft("");
      setIsSending(false);
      setErrorMessage("");
      setIsThreadReady(false);
      setHistoryOpen(false);
      setActiveContextTab("card");

      const savedConversationId = window.localStorage.getItem(storageKey);
      if (!savedConversationId) {
        setConversationId(null);
        setIsThreadReady(true);
        return;
      }

      try {
        const thread = await chatApi.fetchConversation(savedConversationId);
        if (cancelled) return;
        if (!thread || thread.account_pid !== card.context.accountPid) {
          window.localStorage.removeItem(storageKey);
          setConversationId(null);
          setIsThreadReady(true);
          return;
        }

        setConversationId(thread.conversation_id);
        setMessages(mapThreadMessages(thread, initialWelcomeMessage));
      } catch {
        if (cancelled) return;
        window.localStorage.removeItem(storageKey);
        setConversationId(null);
        setMessages([initialWelcomeMessage]);
      } finally {
        if (!cancelled) {
          setIsThreadReady(true);
        }
      }
    }

    void restoreConversation();
    return () => {
      cancelled = true;
    };
  }, [card, chatApi, initialWelcomeMessage, storageKey]);

  useEffect(() => {
    let cancelled = false;

    async function loadThreads() {
      setIsLoadingThreads(true);
      try {
        const threads = await chatApi.fetchConversations(card.context.accountPid, 6);
        if (!cancelled) {
          setRecentThreads(threads);
        }
      } catch {
        if (!cancelled) {
          setRecentThreads([]);
        }
      } finally {
        if (!cancelled) {
          setIsLoadingThreads(false);
        }
      }
    }

    void loadThreads();
    return () => {
      cancelled = true;
    };
  }, [card.context.accountPid, chatApi, conversationId, historyRefreshToken]);

  useEffect(() => {
    conversationIdRef.current = conversationId;
    if (conversationId) {
      window.localStorage.setItem(storageKey, conversationId);
    } else {
      window.localStorage.removeItem(storageKey);
    }
  }, [conversationId, storageKey]);

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

  async function handleComposerKeyDown(event) {
    if (event.key !== "Enter" || event.shiftKey || event.nativeEvent.isComposing) {
      return;
    }

    event.preventDefault();
    const text = draft.trim();
    if (!text || isSending) return;
    await submitMessage(text);
  }

  function handleQuickPrompt(prompt) {
    void submitMessage(prompt);
  }

  async function loadConversationThread(threadId) {
    try {
      const thread = await chatApi.fetchConversation(threadId);
      setConversationId(thread.conversation_id || null);
      setMessages(mapThreadMessages(thread, initialWelcomeMessage));
      setHistoryOpen(false);
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Failed to load conversation history");
    }
  }

  async function handleDeleteConversation(threadId) {
    try {
      await chatApi.deleteConversation(threadId);
      if (threadId === conversationIdRef.current) {
        window.localStorage.removeItem(storageKey);
        setConversationId(null);
        setMessages([initialWelcomeMessage]);
        setDraft("");
      }
      setHistoryRefreshToken((current) => current + 1);
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Failed to delete conversation");
    }
  }

  async function submitMessage(text) {
    const userMessage = {
      id: `user-${Date.now()}`,
      role: "user",
      content: text,
    };
    const nextHistory = [...messages, userMessage];

    setMessages((current) => [...current, userMessage]);
    setDraft("");
    setIsSending(true);

    const assistantMessage = {
      id: `assistant-${Date.now()}`,
      role: "assistant",
      status: "thinking",
      content: isPlannerMode
        ? "Reviewing planner context, budget state, and any required tool calls..."
        : "Retrieving relevant card context and historical signals...",
      sources: [],
      actions: [],
    };

    setMessages((current) => [...current, assistantMessage]);

    try {
      const response = await chatApi.sendMessage({
        message: text,
        conversationId: conversationIdRef.current,
        history: nextHistory,
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
                sources: response.sources || [],
                actions: response.actions || [],
                facts: response.facts,
                retrievalStrategy: response.retrieval_strategy,
                plannerState: response.planner_state,
                turnIntent: response.turn_intent,
                summary: response.summary,
                highlights: response.highlights,
                nextAction: response.next_action,
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
                content: isPlannerMode
                  ? "I couldn’t reach the planner chat endpoint. Make sure the FastAPI server is running on `http://127.0.0.1:8000`."
                  : "I couldn’t reach the backend chat endpoint. Make sure the FastAPI server is running on `http://127.0.0.1:8000`.",
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

  useEffect(() => {
    if (!seedMessage || !seedMessageId || !isThreadReady || seedMessageId === consumedSeedRef.current) {
      return;
    }

    consumedSeedRef.current = seedMessageId;
    onSeedConsumed();
    void submitMessage(seedMessage);
  }, [seedMessage, seedMessageId, isThreadReady, onSeedConsumed]);

  function handleActionChip(action) {
    if (!isSending) {
      void submitMessage(action);
    }
  }

  return (
    <aside className="panel chat-panel">
      <div className="panel-header">
        <div>
          <p className="section-label">{isPlannerMode ? "Planner Agent" : "AI Copilot"}</p>
          <h3>{isPlannerMode ? "Planner chat" : "Finance chat"}</h3>
        </div>
        <div className="chat-header-actions">
          <button className="ghost-button chat-mini-button" type="button" onClick={() => setHistoryOpen((current) => !current)}>
            History
          </button>
          <span className="panel-note">
            <Sparkles size={14} /> Live
          </span>
        </div>
      </div>

      {!isPlannerMode ? (
        <div className="chat-context compact">
          {CONTEXT_TABS.map((tab) => (
            <button
              key={tab.id}
              className={`chat-context-chip chat-context-chip--button ${activeContextTab === tab.id ? "active" : ""}`}
              type="button"
              onClick={() => setActiveContextTab(tab.id)}
            >
              {tab.label}
            </button>
          ))}
        </div>
      ) : null}

      <div className="chat-context-detail">
        <strong>{activeContextDetails.title}</strong>
        <span>{activeContextDetails.description}</span>
      </div>

      <div className="chat-slim-prompts">
        {activeContextDetails.prompts.slice(0, 3).map((prompt) => (
          <button key={prompt} className="suggestion-chip suggestion-chip--slim" type="button" onClick={() => handleQuickPrompt(prompt)}>
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
                <strong>{message.role === "assistant" ? (isPlannerMode ? "Planner Agent" : "Finance Copilot") : "You"}</strong>
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

            {message.actions?.length ? (
              <div className="chat-actions compact">
                {message.actions.slice(0, 3).map((action) => (
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

      <form className="chat-form chat-form--sticky" onSubmit={handleSubmit}>
        <label className="sr-only" htmlFor="chatInput">
          Chat input
        </label>
        <textarea
          id="chatInput"
          rows="2"
          placeholder={isPlannerMode ? "Ask to create, revise, review, or approve a budget..." : "Ask about this card or a specific transaction..."}
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={handleComposerKeyDown}
        />
        <div className="chat-form-footer">
          <span className="panel-note">
            {isPlannerMode ? "Planner turns stay tied to this budgeting conversation." : "Responses stay tied to this card."}
          </span>
          <button className="primary-button" type="submit" disabled={isSending}>
            <SendHorizontal size={16} />
            Send
          </button>
        </div>
      </form>

      {historyOpen ? (
        <div className="chat-history-drawer">
          <div className="chat-history-header">
            <strong>Recent conversations</strong>
            <button className="ghost-button chat-mini-button" type="button" onClick={() => setHistoryOpen(false)}>
              Close
            </button>
          </div>
          <div className="chat-history-list">
            {isLoadingThreads ? (
              <p className="panel-note">Loading history...</p>
            ) : recentThreads.length ? (
              recentThreads.map((thread) => (
                <div key={thread.conversation_id} className="chat-history-item">
                  <button
                    className="chat-history-item-open"
                    type="button"
                    onClick={() => loadConversationThread(thread.conversation_id)}
                  >
                    <div className="chat-history-item-top">
                      <strong>{thread.card_label || thread.account_name || "Conversation"}</strong>
                      <span>{thread.message_count} msgs</span>
                    </div>
                    <p>{thread.preview || "No preview available"}</p>
                  </button>
                  <button
                    className="chat-history-item-delete"
                    type="button"
                    onClick={() => handleDeleteConversation(thread.conversation_id)}
                    aria-label={`Delete conversation ${thread.card_label || thread.account_name || thread.conversation_id}`}
                  >
                    Delete
                  </button>
                </div>
              ))
            ) : (
              <p className="panel-note">
                {isPlannerMode ? "No saved planner conversations for this account yet." : "No saved conversations for this card yet."}
              </p>
            )}
          </div>
        </div>
      ) : null}

      {errorMessage ? <div className="chat-error chat-error--compact" role="alert">{errorMessage}</div> : null}
    </aside>
  );
}

export default ChatPanel;
