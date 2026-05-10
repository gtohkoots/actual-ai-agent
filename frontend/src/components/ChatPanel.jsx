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

function getPlannerStateFromThread(thread) {
  return thread?.context?.planner_state || null;
}

function currency(value) {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 2,
  }).format(value || 0);
}

function buildPlannerStatusView(plannerState) {
  if (!plannerState) return null;

  const pendingRecommendation = plannerState.pending_recommendation || null;
  const savedPlan = plannerState.latest_saved_plan || plannerState.last_create_payload || null;

  if (plannerState.awaiting_approval && pendingRecommendation) {
    return {
      eyebrow: "Draft Ready",
      title: "Budget waiting for approval",
      description: "You can revise the draft with natural language or approve it to save the plan.",
      period:
        pendingRecommendation.period_start && pendingRecommendation.period_end
          ? `${pendingRecommendation.period_start} to ${pendingRecommendation.period_end}`
          : "",
      savings:
        typeof pendingRecommendation.planned_savings === "number"
          ? currency(pendingRecommendation.planned_savings)
          : "",
      targetCount: Array.isArray(pendingRecommendation.category_targets)
        ? pendingRecommendation.category_targets.length
        : 0,
      targetRows: Array.isArray(pendingRecommendation.category_targets)
        ? pendingRecommendation.category_targets
            .filter((item) => typeof item?.recommended_target === "number")
            .slice(0, 4)
            .map((item) => ({
              category: item.category_name || "Uncategorized",
              amount: currency(item.recommended_target),
            }))
        : [],
      actions: [
        { id: "approve", label: "Approve budget", type: "submit", prompt: "Approve this budget" },
        { id: "revise", label: "Revise draft", type: "compose", prompt: "Keep savings at " },
      ],
    };
  }

  if (savedPlan) {
    return {
      eyebrow: "Saved Plan",
      title: "Budget saved successfully",
      description: "The latest approved plan is active. You can review it or ask for a new draft when things change.",
      period:
        savedPlan.period_start && savedPlan.period_end
          ? `${savedPlan.period_start} to ${savedPlan.period_end}`
          : "",
      savings:
        Array.isArray(savedPlan.targets)
          ? (() => {
              const savingsTarget = savedPlan.targets.find((item) => item?.category_name === "Savings");
              return typeof savingsTarget?.target_amount === "number"
                ? currency(savingsTarget.target_amount)
                : "";
            })()
          : "",
      targetCount: Array.isArray(savedPlan.targets) ? savedPlan.targets.length : 0,
      targetRows: Array.isArray(savedPlan.targets)
        ? savedPlan.targets
            .filter((item) => typeof item?.target_amount === "number")
            .slice(0, 4)
            .map((item) => ({
              category: item.category_name || "Uncategorized",
              amount: currency(item.target_amount),
            }))
        : [],
      actions: [
        { id: "review", label: "Review budget", type: "submit", prompt: "Review my current budget" },
        { id: "new-draft", label: "Create new draft", type: "submit", prompt: "Create a budget starting today for a month and save $500" },
      ],
    };
  }

  return {
    eyebrow: "Planner State",
    title: "No pending draft",
    description: "Ask for a new budget draft to start the recommend, revise, and approve workflow.",
    period: "",
    savings: "",
    targetCount: 0,
    targetRows: [],
    actions: [
      { id: "create", label: "Create a draft", type: "submit", prompt: "Create a budget starting today for a month and save $500" },
    ],
  };
}

function ChatPanel({
  card,
  analysisWindow,
  mode = "legacy",
  seedMessage = "",
  seedMessageId = "",
  onSeedConsumed = () => {},
  onPlannerStateChange = () => {},
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
  const [plannerState, setPlannerState] = useState(null);
  const [recentThreads, setRecentThreads] = useState([]);
  const [isLoadingThreads, setIsLoadingThreads] = useState(false);
  const [historyRefreshToken, setHistoryRefreshToken] = useState(0);
  const feedRef = useRef(null);
  const textareaRef = useRef(null);
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

  const plannerStatusView = useMemo(
    () => (isPlannerMode ? buildPlannerStatusView(plannerState) : null),
    [isPlannerMode, plannerState]
  );

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
      setPlannerState(null);

      const savedConversationId = window.localStorage.getItem(storageKey);
      if (!savedConversationId) {
        setConversationId(null);
        onPlannerStateChange(null);
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
        const restoredPlannerState = getPlannerStateFromThread(thread);
        setPlannerState(restoredPlannerState);
        onPlannerStateChange(restoredPlannerState);
      } catch {
        if (cancelled) return;
        window.localStorage.removeItem(storageKey);
        setConversationId(null);
        setMessages([initialWelcomeMessage]);
        setPlannerState(null);
        onPlannerStateChange(null);
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
  }, [card, chatApi, initialWelcomeMessage, onPlannerStateChange, storageKey]);

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
      const restoredPlannerState = getPlannerStateFromThread(thread);
      setPlannerState(restoredPlannerState);
      onPlannerStateChange(restoredPlannerState);
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
        setPlannerState(null);
        onPlannerStateChange(null);
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
      const nextPlannerState = response.planner_state || null;
      setPlannerState(nextPlannerState);
      onPlannerStateChange(nextPlannerState);
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

  function handlePlannerQuickAction(action) {
    if (!action || isSending) return;
    if (action.type === "compose") {
      setDraft((current) => current || action.prompt);
      textareaRef.current?.focus();
      return;
    }
    void submitMessage(action.prompt);
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

      {isPlannerMode && plannerStatusView ? (
        <div className="planner-state-card">
          <div className="planner-state-header">
            <div>
              <p className="section-label">{plannerStatusView.eyebrow}</p>
              <strong>{plannerStatusView.title}</strong>
            </div>
            {plannerStatusView.period ? (
              <span className="planner-state-badge">{plannerStatusView.period}</span>
            ) : null}
          </div>
          <p className="panel-note">{plannerStatusView.description}</p>

          {plannerStatusView.savings || plannerStatusView.targetCount ? (
            <div className="planner-state-metrics">
              {plannerStatusView.savings ? (
                <div className="planner-state-metric">
                  <span>Savings target</span>
                  <strong>{plannerStatusView.savings}</strong>
                </div>
              ) : null}
              {plannerStatusView.targetCount ? (
                <div className="planner-state-metric">
                  <span>Budget targets</span>
                  <strong>{plannerStatusView.targetCount}</strong>
                </div>
              ) : null}
            </div>
          ) : null}

          {plannerStatusView.targetRows.length ? (
            <div className="planner-state-targets">
              {plannerStatusView.targetRows.map((item) => (
                <div key={`${item.category}-${item.amount}`} className="planner-state-target-row">
                  <span>{item.category}</span>
                  <strong>{item.amount}</strong>
                </div>
              ))}
            </div>
          ) : null}

          <div className="planner-state-actions">
            {plannerStatusView.actions.map((action) => (
              <button
                key={action.id}
                className={`suggestion-chip ${action.id === "approve" ? "planner-state-action--primary" : ""}`}
                type="button"
                onClick={() => handlePlannerQuickAction(action)}
              >
                <WandSparkles size={14} />
                {action.label}
              </button>
            ))}
          </div>
        </div>
      ) : null}

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
          ref={textareaRef}
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
