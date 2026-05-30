import { useEffect, useMemo, useRef, useState } from "react";

import { Bot, ChevronRight, LoaderCircle, SendHorizontal, Sparkles, WandSparkles } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import {
  deleteChatConversation,
  fetchAnalysisAccounts,
  fetchAnalysisCategories,
  fetchAnalysisPayees,
  fetchChatConversation,
  fetchChatConversations,
  sendChatMessage,
} from "../chat/api";
import { createWelcomeMessage } from "../chat/mockResponder";
import {
  deletePlannerConversation,
  fetchPlannerConversation,
  fetchPlannerConversations,
  sendPlannerMessage,
} from "../planner/api";

const NOOP = () => {};

const ANALYSIS_SCOPES = [
  { id: "portfolio", label: "Overall" },
  { id: "account", label: "Account" },
  { id: "category", label: "Category" },
  { id: "payee", label: "Payee" },
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
  layout = "page",
  seedMessage = "",
  seedMessageId = "",
  onSeedConsumed = NOOP,
  onPlannerStateChange = NOOP,
}) {
  const isPlannerMode = mode === "planner";
  const isShellLayout = layout === "shell";
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
  const [helpersOpen, setHelpersOpen] = useState(!isShellLayout);
  const [analysisScopeType, setAnalysisScopeType] = useState("portfolio");
  const [selectedAccountPid, setSelectedAccountPid] = useState("");
  const [selectedCategory, setSelectedCategory] = useState("");
  const [selectedPayee, setSelectedPayee] = useState("");
  const [availableAccounts, setAvailableAccounts] = useState([]);
  const [availableCategories, setAvailableCategories] = useState([]);
  const [availablePayees, setAvailablePayees] = useState([]);
  const [isLoadingAnalysisOptions, setIsLoadingAnalysisOptions] = useState(false);
  const [analysisOptionsError, setAnalysisOptionsError] = useState("");
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

  const selectedAccount = useMemo(
    () => availableAccounts.find((item) => item.account_pid === selectedAccountPid) || null,
    [availableAccounts, selectedAccountPid]
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
            selected_tab: "analysis",
            account_pid: card.context.accountPid,
            account_name: card.context.accountName || card.context.card,
            card_label: card.name,
            start_date: analysisWindow?.start || card.context.windowStart,
            end_date: analysisWindow?.end || card.context.windowEnd,
            scope_type: analysisScopeType,
            selected_account_pid: analysisScopeType === "account" ? selectedAccountPid || null : null,
            selected_account_name: analysisScopeType === "account" ? selectedAccount?.account_name || null : null,
            selected_category: analysisScopeType === "category" ? selectedCategory || null : null,
            selected_payee: analysisScopeType === "payee" ? selectedPayee || null : null,
          },
    [
      analysisScopeType,
      analysisWindow?.end,
      analysisWindow?.start,
      card,
      isPlannerMode,
      selectedAccount,
      selectedAccountPid,
      selectedCategory,
      selectedPayee,
    ]
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

    return {
      title: "Analysis workspace",
      description: "Ask about trends, changes, concentrations, or unusual spending and I’ll pull the right evidence.",
      prompts: [
        `How is spending trending for ${analysisWindow?.label || "this window"}?`,
        "What changed most versus the prior period?",
        "Where is spending concentrating the most?",
      ],
    };
  }, [analysisWindow?.label, isPlannerMode]);

  const plannerStatusView = useMemo(
    () => (isPlannerMode ? buildPlannerStatusView(plannerState) : null),
    [isPlannerMode, plannerState]
  );

  const analysisScopeLabel = useMemo(
    () => ANALYSIS_SCOPES.find((item) => item.id === analysisScopeType)?.label || "Overall",
    [analysisScopeType]
  );

  const analysisContextLine = useMemo(() => {
    if (isPlannerMode) return "";

    const scopeSummary =
      analysisScopeType === "account"
        ? selectedAccount?.account_name || "Account"
        : analysisScopeType === "category"
          ? selectedCategory || "Category"
          : analysisScopeType === "payee"
            ? selectedPayee || "Payee"
            : "All accounts";

    return `${scopeSummary} · ${analysisWindow?.label || card.context.dateRange}`;
  }, [
    analysisScopeType,
    analysisWindow?.label,
    card.context.dateRange,
    isPlannerMode,
    selectedAccount?.account_name,
    selectedCategory,
    selectedPayee,
  ]);

  const shellSummary = useMemo(() => {
    if (isPlannerMode) {
      return plannerStatusView?.period
        ? `${plannerStatusView.title} · ${plannerStatusView.period}`
        : plannerStatusView?.title || "Planner tools ready";
    }
    return analysisContextLine || activeContextDetails.title;
  }, [activeContextDetails.title, analysisContextLine, isPlannerMode, plannerStatusView]);

  useEffect(() => {
    setHelpersOpen(!isShellLayout);
  }, [isShellLayout, mode]);

  useEffect(() => {
    if (isPlannerMode || !["account", "category", "payee"].includes(analysisScopeType)) {
      setAnalysisOptionsError("");
      setIsLoadingAnalysisOptions(false);
      return;
    }

    let cancelled = false;

    async function loadAnalysisOptions() {
      setIsLoadingAnalysisOptions(true);
      setAnalysisOptionsError("");
      try {
        if (analysisScopeType === "account") {
          const items = await fetchAnalysisAccounts();
          if (!cancelled) {
            setAvailableAccounts(items);
            setSelectedAccountPid((current) => {
              if (current && items.some((item) => item.account_pid === current)) {
                return current;
              }
              return items.find((item) => item.account_pid === card.context.accountPid)?.account_pid || items[0]?.account_pid || "";
            });
          }
        } else if (analysisScopeType === "category") {
          const items = await fetchAnalysisCategories({
            accountPid: card.context.accountPid,
            accountName: card.context.accountName || card.context.card,
            startDate: analysisWindow?.start || card.context.windowStart,
            endDate: analysisWindow?.end || card.context.windowEnd,
          });
          if (!cancelled) {
            setAvailableCategories(items);
            setSelectedCategory((current) => (current && items.includes(current) ? current : items[0] || ""));
          }
        } else {
          const items = await fetchAnalysisPayees({
            accountPid: card.context.accountPid,
            accountName: card.context.accountName || card.context.card,
            startDate: analysisWindow?.start || card.context.windowStart,
            endDate: analysisWindow?.end || card.context.windowEnd,
            limit: 50,
          });
          if (!cancelled) {
            setAvailablePayees(items);
            setSelectedPayee((current) => (current && items.includes(current) ? current : items[0] || ""));
          }
        }
      } catch (error) {
        if (!cancelled) {
          setAnalysisOptionsError(error instanceof Error ? error.message : "Failed to load analysis options");
        }
      } finally {
        if (!cancelled) {
          setIsLoadingAnalysisOptions(false);
        }
      }
    }

    void loadAnalysisOptions();
    return () => {
      cancelled = true;
    };
  }, [analysisScopeType, analysisWindow?.end, analysisWindow?.start, card, isPlannerMode]);

  useEffect(() => {
    if (analysisScopeType !== "account") {
      setSelectedAccountPid("");
    }
    if (analysisScopeType !== "category") {
      setSelectedCategory("");
    }
    if (analysisScopeType !== "payee") {
      setSelectedPayee("");
    }
  }, [analysisScopeType]);

  useEffect(() => {
    let cancelled = false;

    async function restoreConversation() {
      setMessages([initialWelcomeMessage]);
      setDraft("");
      setIsSending(false);
      setErrorMessage("");
      setAnalysisOptionsError("");
      setIsThreadReady(false);
      setHistoryOpen(false);
      setAnalysisScopeType("portfolio");
      setSelectedAccountPid("");
      setSelectedCategory("");
      setSelectedPayee("");
      setAvailableAccounts([]);
      setAvailableCategories([]);
      setAvailablePayees([]);
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
    if (!text || isSending || isScopeSelectionIncomplete) return;
    await submitMessage(text);
  }

  async function handleComposerKeyDown(event) {
    if (event.key !== "Enter" || event.shiftKey || event.nativeEvent.isComposing) {
      return;
    }

    event.preventDefault();
    const text = draft.trim();
    if (!text || isSending || isScopeSelectionIncomplete) return;
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
    if (isScopeSelectionIncomplete) {
      return;
    }

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
        : "Planning the best analysis route and collecting the right evidence...",
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
    if (!action || isSending || isScopeSelectionIncomplete) return;
    if (action.type === "compose") {
      setDraft((current) => current || action.prompt);
      textareaRef.current?.focus();
      return;
    }
    void submitMessage(action.prompt);
  }

  const showHelperSection = !isShellLayout || helpersOpen;
  const rootClassName = `panel chat-panel ${isShellLayout ? "chat-panel--shell" : "chat-panel--page"}`;
  const helperButtonLabel = helpersOpen ? "Hide helpers" : "Show helpers";
  const isScopeSelectionIncomplete = !isPlannerMode && ((analysisScopeType === "account" && !selectedAccountPid) || (analysisScopeType === "category" && !selectedCategory) || (analysisScopeType === "payee" && !selectedPayee));

  return (
    <aside className={rootClassName}>
      <div className="panel-header chat-panel__header">
        <div>
          <p className="section-label">{isPlannerMode ? "Planner Agent" : "Analysis Agent"}</p>
          <h3>{isPlannerMode ? "Planner chat" : "Finance chat"}</h3>
        </div>
        <div className="chat-header-actions">
          {isShellLayout ? (
            <button className="ghost-button chat-mini-button" type="button" onClick={() => setHelpersOpen((current) => !current)}>
              {helperButtonLabel}
            </button>
          ) : null}
          <button className="ghost-button chat-mini-button" type="button" onClick={() => setHistoryOpen((current) => !current)}>
            History
          </button>
          <span className="panel-note chat-live-pill">
            <Sparkles size={14} /> Live
          </span>
        </div>
      </div>

      {isShellLayout ? (
        <div className="chat-shell-summary">
          <strong>{shellSummary}</strong>
          <span>{activeContextDetails.description}</span>
        </div>
      ) : null}

      {showHelperSection ? (
        <div className={`chat-helper-stack ${isShellLayout ? "chat-helper-stack--shell" : ""}`}>
          {isPlannerMode ? (
            <div className="chat-context-detail">
              <strong>{activeContextDetails.title}</strong>
              <span>{activeContextDetails.description}</span>
            </div>
          ) : (
            <div className="analysis-current-context">
              <strong>Current context</strong>
              <span>{analysisContextLine}</span>
            </div>
          )}

          {!isPlannerMode ? (
            <div className={`analysis-scope-card ${isShellLayout ? "analysis-scope-card--shell" : ""}`}>
              <div className="analysis-scope-card__header">
                <div>
                  <strong>Analysis scope</strong>
                  <span>Choose whether to analyze overall activity, this account, a category, or a payee.</span>
                </div>
                <span className="analysis-scope-card__badge">{analysisScopeLabel}</span>
              </div>
              <div className="analysis-scope-pills" role="tablist" aria-label="Analysis scope">
                {ANALYSIS_SCOPES.map((scope) => (
                  <button
                    key={scope.id}
                    className={`chat-context-chip chat-context-chip--button ${analysisScopeType === scope.id ? "active" : ""}`}
                    type="button"
                    role="tab"
                    aria-selected={analysisScopeType === scope.id}
                    onClick={() => setAnalysisScopeType(scope.id)}
                  >
                    {scope.label}
                  </button>
                ))}
              </div>
              {analysisScopeType === "account" ? (
                <label className="analysis-scope-field">
                  <span>Account</span>
                  <select value={selectedAccountPid} onChange={(event) => setSelectedAccountPid(event.target.value)} disabled={isLoadingAnalysisOptions}>
                    {availableAccounts.length ? (
                      availableAccounts.map((item) => (
                        <option key={item.account_pid} value={item.account_pid}>
                          {item.account_name}
                        </option>
                      ))
                    ) : (
                      <option value="">{isLoadingAnalysisOptions ? "Loading accounts..." : "No accounts available"}</option>
                    )}
                  </select>
                </label>
              ) : null}
              {analysisScopeType === "category" ? (
                <label className="analysis-scope-field">
                  <span>Category</span>
                  <select value={selectedCategory} onChange={(event) => setSelectedCategory(event.target.value)} disabled={isLoadingAnalysisOptions}>
                    {availableCategories.length ? (
                      availableCategories.map((item) => (
                        <option key={item} value={item}>
                          {item}
                        </option>
                      ))
                    ) : (
                      <option value="">{isLoadingAnalysisOptions ? "Loading categories..." : "No categories available"}</option>
                    )}
                  </select>
                </label>
              ) : null}
              {analysisScopeType === "payee" ? (
                <label className="analysis-scope-field">
                  <span>Payee</span>
                  <select value={selectedPayee} onChange={(event) => setSelectedPayee(event.target.value)} disabled={isLoadingAnalysisOptions}>
                    {availablePayees.length ? (
                      availablePayees.map((item) => (
                        <option key={item} value={item}>
                          {item}
                        </option>
                      ))
                    ) : (
                      <option value="">{isLoadingAnalysisOptions ? "Loading payees..." : "No payees available"}</option>
                    )}
                  </select>
                </label>
              ) : null}
              {analysisOptionsError ? <span className="analysis-scope-card__error">{analysisOptionsError}</span> : null}
            </div>
          ) : null}

          <div className="chat-slim-prompts">
            {activeContextDetails.prompts.slice(0, isShellLayout ? 2 : 3).map((prompt) => (
              <button key={prompt} className="suggestion-chip suggestion-chip--slim" type="button" onClick={() => handleQuickPrompt(prompt)}>
                <WandSparkles size={14} />
                {prompt}
              </button>
            ))}
          </div>

          {isPlannerMode && plannerStatusView ? (
            <div className={`planner-state-card ${isShellLayout ? "planner-state-card--shell" : ""}`}>
              <div className="planner-state-header">
                <div>
                  <p className="section-label">{plannerStatusView.eyebrow}</p>
                  <strong>{plannerStatusView.title}</strong>
                </div>
                {plannerStatusView.period ? <span className="planner-state-badge">{plannerStatusView.period}</span> : null}
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
        </div>
      ) : null}

      <div className="chat-feed" ref={feedRef}>
        {messages.map((message) => {
          const isUserMessage = message.role === "user";
          const rowClassName = `chat-row chat-row--${message.role}`;
          const messageClassName = `chat-message chat-message--${message.role} ${isShellLayout ? "chat-message--shell" : "chat-message--page"}`;

          return (
            <div key={message.id} className={rowClassName}>
              <article className={messageClassName}>
                {isUserMessage ? (
                  <div className="chat-message-meta chat-message-meta--user">
                    <span>You</span>
                  </div>
                ) : (
                  <div className="chat-message-head">
                    <div className="chat-role">
                      <span className="chat-avatar"><Bot size={14} /></span>
                      <div className="chat-role-copy">
                        <strong>{isPlannerMode ? "Planner Agent" : "Finance Copilot"}</strong>
                        <span>{isPlannerMode ? "Budget guidance and approvals" : "Analysis grounded in tool results"}</span>
                      </div>
                    </div>
                    {message.status === "thinking" ? (
                      <span className="chat-status">
                        <LoaderCircle size={14} className="spin" />
                        Thinking
                      </span>
                    ) : null}
                  </div>
                )}

                <div className="chat-content">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
                </div>

                {!isUserMessage && message.actions?.length ? (
                  <div className="chat-actions compact">
                    {message.actions.slice(0, isShellLayout ? 2 : 3).map((action) => (
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
            </div>
          );
        })}
      </div>

      <form className={`chat-form ${isShellLayout ? "chat-form--shell" : "chat-form--sticky"}`} onSubmit={handleSubmit}>
        <label className="sr-only" htmlFor={`chatInput-${layout}-${mode}`}>
          Chat input
        </label>
        <textarea
          ref={textareaRef}
          id={`chatInput-${layout}-${mode}`}
          rows="2"
          placeholder={isPlannerMode ? "Ask to create, revise, review, or approve a budget..." : "Ask about your finances, a category, or a merchant..."}
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={handleComposerKeyDown}
        />
        <div className="chat-form-footer">
          <span className="panel-note">
{isPlannerMode ? "Planner turns stay tied to this budgeting conversation." : isScopeSelectionIncomplete ? "Pick a category or payee before sending this scoped analysis." : "Analysis turns stay tied to this account conversation."}
          </span>
          <button className="primary-button" type="submit" disabled={isSending || isScopeSelectionIncomplete}>
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
                {isPlannerMode ? "No saved planner conversations for this account yet." : "No saved conversations for this account yet."}
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
