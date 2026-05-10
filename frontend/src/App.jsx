import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ChevronLeft, ChevronRight, CreditCard, LayoutDashboard, MessageCircle, PiggyBank, TrendingUp } from "lucide-react";
import { DayPicker } from "react-day-picker";
import { startOfMonth, subDays } from "date-fns";
import "react-day-picker/style.css";
import {
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
} from "recharts";

import ChatPanel from "./components/ChatPanel";
import { fetchAccounts, fetchDashboardOverview } from "./api/dashboard";
import { fetchPlannerOverview } from "./planner/api";

const railNavItems = [
  { label: "Overview", icon: LayoutDashboard, tab: "Overview" },
  { label: "Card Details", icon: CreditCard, tab: "Card Details" },
  { label: "Spending Analysis", icon: TrendingUp, tab: "Spending Analysis" },
  { label: "Budgeting", icon: PiggyBank, tab: "Budgeting Goals" },
];
const WINDOW_PRESETS = [
  { value: "all_time", label: "All time" },
  { value: "month_to_date", label: "Month to date" },
  { value: "last_30_days", label: "Last 30 days" },
  { value: "last_7_days", label: "Last 7 days" },
];

const CARD_TINTS = [
  "linear-gradient(135deg, #235446 0%, #122b24 100%)",
  "linear-gradient(135deg, #51301f 0%, #1f130e 100%)",
  "linear-gradient(135deg, #1b2c55 0%, #0d1425 100%)",
  "linear-gradient(135deg, #5a3d18 0%, #2f1c08 100%)",
];

function currency(value) {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 2,
  }).format(value || 0);
}

function formatLocalDate(date) {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

function pickCardTint(index) {
  return CARD_TINTS[index % CARD_TINTS.length];
}

function toLocalDate(date) {
  if (!date) return "";
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

function fromLocalDate(value) {
  if (!value) return null;
  const [year, month, day] = value.split("-").map(Number);
  return new Date(year, month - 1, day);
}

function getPresetRange(preset) {
  const today = new Date();
  if (preset === "last_7_days") {
    return { from: subDays(today, 6), to: today };
  }
  if (preset === "last_30_days") {
    return { from: subDays(today, 29), to: today };
  }
  return { from: startOfMonth(today), to: today };
}

function formatWindowLabel(range) {
  if (!range?.from || !range?.to) return "Select a window";
  if (toLocalDate(range.from) === toLocalDate(range.to)) {
    return toLocalDate(range.from);
  }
  return `${toLocalDate(range.from)} to ${toLocalDate(range.to)}`;
}

function SimpleListTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;

  return (
    <div className="chart-tooltip">
      {payload.map((entry) => (
        <span key={`${entry.dataKey}-${entry.name || entry.value}`}>
          {entry.name || entry.dataKey}: {currency(entry.value)}
        </span>
      ))}
    </div>
  );
}

function renderPieCategoryLabel({ cx, cy, midAngle, outerRadius, category, source, name }) {
  const radius = outerRadius + 22;
  const x = cx + radius * Math.cos((-midAngle * Math.PI) / 180);
  const y = cy + radius * Math.sin((-midAngle * Math.PI) / 180);
  const anchor = x > cx ? "start" : "end";
  const label = category || source || name || "";

  return (
    <text x={x} y={y} fill="#4f483f" textAnchor={anchor} dominantBaseline="central" fontSize={12} fontWeight={700}>
      {label}
    </text>
  );
}

function buildCardViewModel(account, index, accountMeta = {}) {
  const summary = account.summary || {};
  const mergedSummary = {
    totalSpend: summary.totalSpend || currency(account.cycle_spend),
    totalIncome: summary.totalIncome || currency(0),
    topCategory: summary.topCategory || "n/a",
    topMerchant: summary.topMerchant || "n/a",
    netCashFlow: summary.netCashFlow || currency(0),
    aiSuggestion: summary.aiSuggestion || "Ask for a weekly summary",
  };
  return {
    id: account.account_pid,
    name: account.account_name,
    network: accountMeta.network || "Account",
    last4: accountMeta.last4 || account.account_pid.slice(-4).toUpperCase(),
    tint: pickCardTint(index),
    balanceCurrent: account.balance_current || 0,
    cycleSpend: account.cycle_spend || 0,
    deltaText: account.delta_text || "+0.0% vs previous window",
    utilizationText: account.utilization_text || "Active account",
    creditLimit: accountMeta.creditLimit || "Live account",
    transactionCount: accountMeta.transaction_count || 0,
    summary: mergedSummary,
    categories: account.categories || [],
    merchants: account.merchants || [],
    transactions: account.transactions || [],
    quickPrompts: account.quick_prompts || [],
    context: account.context || {
      card: account.account_name,
      accountName: account.account_name,
      accountPid: account.account_pid,
      dateRange: "Current window",
      windowStart: account.window_start,
      windowEnd: account.window_end,
      focus: `${mergedSummary.topCategory} + ${mergedSummary.topMerchant}`,
    },
  };
}

function LoadingState() {
  return (
    <div className="panel" style={{ minHeight: "420px", display: "grid", placeItems: "center" }}>
      <div>
        <p className="section-label">Loading data</p>
        <h3>Connecting to backend accounts and dashboard summary...</h3>
      </div>
    </div>
  );
}

function formatBudgetPeriod(periodStart, periodEnd) {
  if (!periodStart || !periodEnd) return "No active budget";
  return `${periodStart} to ${periodEnd}`;
}

function getBudgetCategoryTone(status) {
  if (status === "overspent") return "is-danger";
  if (status === "at_risk") return "is-warning";
  return "is-good";
}

function App() {
  const [accountsMeta, setAccountsMeta] = useState([]);
  const [dashboard, setDashboard] = useState(null);
  const [selectedCardId, setSelectedCardId] = useState(null);
  const [activeTab, setActiveTab] = useState("Overview");
  const [assistantSeed, setAssistantSeed] = useState(null);
  const [windowPreset, setWindowPreset] = useState("all_time");
  const [windowRange, setWindowRange] = useState(null);
  const [windowDraftRange, setWindowDraftRange] = useState(null);
  const [windowPickerOpen, setWindowPickerOpen] = useState(false);
  const [loading, setLoading] = useState(true);
  const [isRefreshingDashboard, setIsRefreshingDashboard] = useState(false);
  const [plannerOverview, setPlannerOverview] = useState(null);
  const [isLoadingPlannerOverview, setIsLoadingPlannerOverview] = useState(false);
  const [plannerOverviewError, setPlannerOverviewError] = useState("");
  const [isSidebarExpanded, setIsSidebarExpanded] = useState(false);
  const [error, setError] = useState("");
  const [plannerOverviewRefreshToken, setPlannerOverviewRefreshToken] = useState(0);
  const hasLoadedDashboardRef = useRef(false);

  const selectedWindow = useMemo(
    () => ({
      start: toLocalDate(windowRange?.from || fromLocalDate(dashboard?.window?.start)),
      end: toLocalDate(windowRange?.to || fromLocalDate(dashboard?.window?.end)),
      label:
        windowPreset === "all_time"
          ? "All time"
          : formatWindowLabel(windowRange),
    }),
    [windowPreset, windowRange, dashboard?.window?.start, dashboard?.window?.end]
  );

  useEffect(() => {
    let cancelled = false;

    async function loadDashboard() {
      const shouldUseBackendDefault = windowPreset === "all_time" && !windowRange;
      const startDate = shouldUseBackendDefault ? null : toLocalDate(windowRange?.from);
      const endDate = shouldUseBackendDefault ? null : toLocalDate(windowRange?.to);
      if (!shouldUseBackendDefault && (!startDate || !endDate)) {
        return;
      }
      const isInitialLoad = !hasLoadedDashboardRef.current;
      if (isInitialLoad) {
        setLoading(true);
      } else {
        setIsRefreshingDashboard(true);
      }
      setError("");
      try {
        const dashboardResponse = shouldUseBackendDefault
          ? await fetchDashboardOverview()
          : await fetchDashboardOverview(startDate, endDate);
        if (cancelled) return;
        setDashboard(dashboardResponse);
        hasLoadedDashboardRef.current = true;
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Failed to load dashboard data");
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
          setIsRefreshingDashboard(false);
        }
      }
    }

    void loadDashboard();
    return () => {
      cancelled = true;
    };
  }, [windowPreset, windowRange?.from, windowRange?.to]);

  useEffect(() => {
    let cancelled = false;

    async function loadAccounts() {
      try {
        const accountsResponse = await fetchAccounts();
        if (!cancelled) {
          setAccountsMeta(accountsResponse);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Failed to load accounts");
        }
      }
    }

    void loadAccounts();
    return () => {
      cancelled = true;
    };
  }, []);

  const cards = useMemo(() => {
    const accountMetaMap = new Map(accountsMeta.map((account) => [account.account_pid, account]));
    return (dashboard?.accounts || []).map((account, index) =>
      buildCardViewModel(account, index, accountMetaMap.get(account.account_pid))
    );
  }, [accountsMeta, dashboard]);

  const selectedCard = useMemo(
    () => cards.find((card) => card.id === selectedCardId) || cards[0] || null,
    [cards, selectedCardId]
  );
  const portfolio = dashboard?.portfolio || {};

  useEffect(() => {
    if (!selectedCardId && cards.length > 0) {
      setSelectedCardId(cards[0].id);
    }
  }, [cards, selectedCardId]);

  const isAssistantView = activeTab === "AI Assistant";
  const isBudgetingTab = activeTab === "Budgeting Goals" || activeTab === "Budgeting Plan";
  const spendPieColors = ["#1f5c4d", "#aa7d2d", "#a04b2f", "#5a3d18", "#1b2c55"];
  const incomePieColors = ["#214d73", "#4d7ba3", "#7d5ba6", "#5a3d18", "#1f5c4d"];

  useEffect(() => {
    let cancelled = false;

    async function loadPlannerOverview() {
      if (!isBudgetingTab) return;
      setIsLoadingPlannerOverview(true);
      setPlannerOverviewError("");
      try {
        const overview = await fetchPlannerOverview();
        if (!cancelled) {
          setPlannerOverview(overview);
        }
      } catch (err) {
        if (!cancelled) {
          setPlannerOverviewError(err instanceof Error ? err.message : "Failed to load planner overview");
          setPlannerOverview(null);
        }
      } finally {
        if (!cancelled) {
          setIsLoadingPlannerOverview(false);
        }
      }
    }

    void loadPlannerOverview();
    return () => {
      cancelled = true;
    };
  }, [isBudgetingTab, plannerOverviewRefreshToken]);
  const stats = selectedCard
    ? [
        { label: "Current Balance", value: currency(selectedCard.balanceCurrent), note: selectedCard.deltaText },
        {
          label: "Selected Card Spend + Income",
          kind: "mini-card",
          spendValue: selectedCard.summary.totalSpend,
          spendLabel: "Spend",
          incomeValue: selectedCard.summary.totalIncome,
          incomeLabel: "Income",
          note: "Current cycle activity",
        },
        { label: "Top Category", value: selectedCard.summary.topCategory, note: selectedCard.utilizationText },
        { label: "Top Merchant", value: selectedCard.summary.topMerchant, note: "Best AI follow-up target" },
        { label: "Net Cash Flow", value: selectedCard.summary.netCashFlow, note: selectedCard.summary.aiSuggestion },
      ]
    : [];

  function handleCardSelect(cardId) {
    setSelectedCardId(cardId);
    setActiveTab("Card Details");
  }

  function handleAssistantOpen() {
    setActiveTab("AI Assistant");
  }

  function handlePresetSelect(preset) {
    setWindowPreset(preset);
    if (preset === "all_time") {
      setWindowRange(null);
      setWindowDraftRange(null);
    } else {
      const range = getPresetRange(preset);
      setWindowRange(range);
      setWindowDraftRange(range);
    }
    setWindowPickerOpen(false);
  }

  function handleWindowSelect(range) {
    if (!range?.from || !range?.to) {
      setWindowDraftRange(range);
      return;
    }
    setWindowPreset("custom");
    setWindowRange(range);
    setWindowDraftRange(range);
    setWindowPickerOpen(false);
  }

  function handleTransactionAsk(transaction) {
    if (!selectedCard) return;
    const prompt = `Explain this transaction on ${selectedCard.name}: ${transaction.date} · ${transaction.merchant} · ${currency(transaction.amount)} · ${transaction.category}. Is it expected or unusual?`;
    setAssistantSeed({
      id: `${selectedCard.id}-${transaction.date}-${transaction.merchant}-${Date.now()}`,
      text: prompt,
    });
    setActiveTab("AI Assistant");
  }

  function clearAssistantSeed() {
    setAssistantSeed(null);
  }

  const handlePlannerStateChange = useCallback((nextPlannerState) => {
    if (!nextPlannerState) {
      return;
    }

    const savedPlan = nextPlannerState.latest_saved_plan || nextPlannerState.last_create_payload;
    if (!savedPlan) {
      return;
    }

    setPlannerOverviewRefreshToken((current) => current + 1);
  }, []);

  function renderOverviewTab() {
    const incomeRaw = portfolio.summary?.totalIncome;
    const totalIncome = typeof incomeRaw === "number" ? currency(incomeRaw) : (incomeRaw || currency(0));
    const totalSpendRaw = portfolio.summary?.totalSpend;
    const netCashRaw = portfolio.summary?.netCashFlow;
    const totalSpend = typeof totalSpendRaw === "number" ? currency(totalSpendRaw) : (totalSpendRaw || currency(0));
    const netCashFlow = typeof netCashRaw === "number" ? currency(netCashRaw) : (netCashRaw || currency(0));
    const topCategory = portfolio.topCategories?.[0]?.category || "n/a";
    const spendMix = (portfolio.categoryMix || []).filter((entry) => Number(entry?.amount || 0) > 0);
    const incomeMix = (portfolio.incomeMix || []).filter((entry) => Number(entry?.amount || 0) > 0);

    return (
      <>
        <section className="overview-focus-grid">
          <article className="panel overview-income-panel">
            <p className="overview-income-title">Money You Made</p>
            <p className="section-label">Income in selected timeframe</p>
            <h3 className="overview-income-value">{totalIncome}</h3>
            <p className="panel-note">{selectedWindow.label}</p>
          </article>

          <article className="panel overview-spendmix-panel">
            <div className="panel-header">
              <div>
                <p className="section-label">Cash Flow Breakdown</p>
                <h3>Where spend and income are concentrated</h3>
              </div>
            </div>
            {spendMix.length || incomeMix.length ? (
              <>
                <div className="overview-spendmix-visuals">
                  <div className="overview-spendmix-pie-card">
                    <span className="overview-spendmix-label">Spend sources</span>
                    <div className="overview-spendmix-pie-visual">
                      {spendMix.length ? (
                        <ResponsiveContainer width="100%" height="100%">
                          <PieChart margin={{ top: 10, right: 70, bottom: 10, left: 70 }}>
                            <Tooltip content={<SimpleListTooltip />} />
                            <Pie
                              data={spendMix}
                              dataKey="amount"
                              nameKey="category"
                              innerRadius={42}
                              outerRadius={78}
                              paddingAngle={2}
                              labelLine
                              label={renderPieCategoryLabel}
                            >
                              {spendMix.map((entry, index) => (
                                <Cell key={entry.category} fill={spendPieColors[index % spendPieColors.length]} />
                              ))}
                            </Pie>
                          </PieChart>
                        </ResponsiveContainer>
                      ) : (
                        <div className="overview-spendmix-empty">
                          <p className="panel-note">No spending data for this timeframe yet.</p>
                        </div>
                      )}
                    </div>
                  </div>

                  <div className="overview-spendmix-pie-card">
                    <span className="overview-spendmix-label">Income categories</span>
                    <div className="overview-spendmix-pie-visual">
                      {incomeMix.length ? (
                        <ResponsiveContainer width="100%" height="100%">
                          <PieChart margin={{ top: 10, right: 70, bottom: 10, left: 70 }}>
                            <Tooltip content={<SimpleListTooltip />} />
                            <Pie
                              data={incomeMix}
                              dataKey="amount"
                              nameKey="source"
                              innerRadius={42}
                              outerRadius={78}
                              paddingAngle={2}
                              labelLine
                              label={renderPieCategoryLabel}
                            >
                              {incomeMix.map((entry, index) => (
                                <Cell key={entry.source} fill={incomePieColors[index % incomePieColors.length]} />
                              ))}
                            </Pie>
                          </PieChart>
                        </ResponsiveContainer>
                      ) : (
                        <div className="overview-spendmix-empty">
                          <p className="panel-note">No income data for this timeframe yet.</p>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </>
            ) : (
              <div className="overview-spendmix-empty">
                <p className="panel-note">No spending data for this timeframe yet.</p>
              </div>
            )}
          </article>
        </section>

        <section className="overview-secondary-grid">
          <article className="stat-card">
            <p className="section-label">Total Spend</p>
            <h3>{totalSpend}</h3>
            <p>Total amount spent in this timeframe.</p>
          </article>
          <article className="stat-card">
            <p className="section-label">Net Cash Flow</p>
            <h3>{netCashFlow}</h3>
            <p>Income minus spend for this window.</p>
          </article>
          <article className="stat-card">
            <p className="section-label">Top Category</p>
            <h3>{topCategory}</h3>
            <p>The largest spending category selected period.</p>
          </article>
        </section>
      </>
    );
  }

  function renderCardsTab() {
    return (
      <>
        <section className="stats-grid">
          {stats.map((stat) => (
            <article
              key={stat.label}
              className={`stat-card ${stat.kind === "mini-card" ? "stat-card--mini" : ""}`}
            >
              <p className="section-label">{stat.label}</p>
              {stat.kind === "mini-card" ? (
                <div className="mini-stat-card">
                  <div className="mini-stat-card__col mini-stat-card__col--spend">
                    <span className="mini-stat-card__label">{stat.spendLabel}</span>
                    <strong>{stat.spendValue}</strong>
                  </div>
                  <div className="mini-stat-card__col mini-stat-card__col--income">
                    <span className="mini-stat-card__label">{stat.incomeLabel}</span>
                    <strong>{stat.incomeValue}</strong>
                  </div>
                </div>
              ) : (
                <h3>{stat.value}</h3>
              )}
              <p>{stat.note}</p>
            </article>
          ))}
        </section>

        <section className="cards-section">
          <div className="section-header">
            <div>
              <p className="section-label">Cards On File</p>
              <h3>Select a card to drive the entire workspace</h3>
            </div>
            <p className="section-copy">
              The selected card controls dashboard stats, chart-like summaries, transactions, and default AI
              prompts.
            </p>
          </div>

          <div className="card-rail">
            {cards.map((card) => (
              <button
                key={card.id}
                className={`finance-card ${card.id === selectedCardId ? "active" : ""}`}
                style={{ background: card.tint }}
                type="button"
                onClick={() => handleCardSelect(card.id)}
              >
                <div className="card-brand">
                  <span>{card.network}</span>
                  <span>{card.transactionCount} tx</span>
                </div>
                <h4 className="card-name">{card.name}</h4>
                <p className="card-metric">{currency(card.balanceCurrent)}</p>
                <div className="card-foot">
                  <span>{card.deltaText}</span>
                  <span>Balance current</span>
                </div>
              </button>
            ))}
          </div>
        </section>

        <section className="dashboard-grid">
          <article className="panel">
            <div className="panel-header">
              <div>
                <p className="section-label">Category Spend</p>
                <h3>Where the money is going</h3>
              </div>
              <span className="panel-note">Selected card only</span>
            </div>
            <div className="category-bars">
              {selectedCard.categories.map((item) => {
                const maxAmount = Math.max(...selectedCard.categories.map((category) => category.amount));
                return (
                  <div key={item.category} className="category-row">
                    <div className="category-meta">
                      <strong>{item.category}</strong>
                      <span>{currency(item.amount)}</span>
                    </div>
                    <div className="category-track">
                      <div className="category-fill" style={{ width: `${(item.amount / maxAmount) * 100}%` }} />
                    </div>
                  </div>
                );
              })}
            </div>
          </article>

          <article className="panel">
            <div className="panel-header">
              <div>
                <p className="section-label">Merchant Highlights</p>
                <h3>Top payees this cycle</h3>
              </div>
              <span className="panel-note">Useful for AI follow-ups</span>
            </div>
            <div className="merchant-list">
              {selectedCard.merchants.map((merchant) => (
                <div key={merchant.payee} className="merchant-row">
                  <div>
                    <strong>{merchant.payee}</strong>
                    <div className="panel-note">{merchant.amount}</div>
                  </div>
                  <strong>{merchant.amount}</strong>
                </div>
              ))}
            </div>
          </article>
        </section>

        <section className="transactions-and-chat">
          <article className="panel transactions-panel">
            <div className="panel-header">
              <div>
                <p className="section-label">Recent Transactions</p>
                <h3>Ask the AI about any row you see here</h3>
              </div>
            </div>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Date</th>
                    <th>Merchant</th>
                    <th>Category</th>
                    <th>Amount</th>
                  </tr>
                </thead>
                <tbody>
                  {selectedCard.transactions.map((tx) => (
                    <tr
                      key={`${tx.date}-${tx.merchant}-${tx.amount}`}
                      className="transaction-row"
                      role="button"
                      tabIndex={0}
                      aria-label={`Ask AI about transaction ${tx.merchant} on ${tx.date} for ${currency(tx.amount)}`}
                      onClick={() => handleTransactionAsk(tx)}
                      onKeyDown={(event) => {
                        if (event.key === "Enter" || event.key === " ") {
                          event.preventDefault();
                          handleTransactionAsk(tx);
                        }
                      }}
                    >
                      <td>{tx.date}</td>
                      <td>{tx.merchant}</td>
                      <td>{tx.category}</td>
                      <td className="amount-negative">
                        <span>{currency(tx.amount)}</span>
                        <span className="transaction-ask">Ask AI</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </article>

          <ChatPanel
            card={selectedCard}
            analysisWindow={selectedWindow}
            seedMessage={assistantSeed?.text || ""}
            seedMessageId={assistantSeed?.id || ""}
            onSeedConsumed={clearAssistantSeed}
          />
        </section>
      </>
    );
  }

  function renderSpendingAnalysisTab() {
    const topCategory = selectedCard?.summary?.topCategory || "n/a";
    const topMerchant = selectedCard?.summary?.topMerchant || "n/a";
    const topPortfolioCategory = portfolio.topCategories?.[0];

    return (
      <>
        <section className="stats-grid">
          <article className="stat-card">
            <p className="section-label">Behavior Signal</p>
            <h3>{topCategory}</h3>
            <p>Top category driving spend on the selected card.</p>
          </article>
          <article className="stat-card">
            <p className="section-label">Top Merchant</p>
            <h3>{topMerchant}</h3>
            <p>Most frequent merchant influence in this window.</p>
          </article>
          <article className="stat-card">
            <p className="section-label">Portfolio Pattern</p>
            <h3>{topPortfolioCategory?.category || "n/a"}</h3>
            <p>Strongest spend concentration across all cards.</p>
          </article>
          <article className="stat-card">
            <p className="section-label">Analysis Window</p>
            <h3>{selectedWindow.label}</h3>
            <p>Current time frame used for behavior analysis.</p>
          </article>
        </section>

        <section className="dashboard-grid">
          <article className="panel">
            <div className="panel-header">
              <div>
                <p className="section-label">Category Drivers</p>
                <h3>Where spending behavior concentrates</h3>
              </div>
              <span className="panel-note">Selected card</span>
            </div>
            <div className="category-bars">
              {(selectedCard?.categories || []).slice(0, 6).map((entry) => (
                <div key={entry.category} className="category-row">
                  <div className="category-meta">
                    <span>{entry.category}</span>
                    <strong>{currency(entry.amount)}</strong>
                  </div>
                  <div className="category-track">
                    <div className="category-fill" style={{ width: `${Math.max(8, entry.share_pct || 0)}%` }} />
                  </div>
                </div>
              ))}
            </div>
          </article>

          <article className="panel">
            <div className="panel-header">
              <div>
                <p className="section-label">Behavior Notes</p>
                <h3>What this period suggests</h3>
              </div>
            </div>
            <div className="merchant-list">
              <div className="merchant-row">
                <span>Concentration risk</span>
                <strong>{topCategory} heavy</strong>
              </div>
              <div className="merchant-row">
                <span>Largest merchant factor</span>
                <strong>{topMerchant}</strong>
              </div>
              <div className="merchant-row">
                <span>Suggested next question</span>
                <strong>Why did this pattern shift vs previous window?</strong>
              </div>
            </div>
          </article>
        </section>
      </>
    );
  }

  function renderBudgetingTab() {
    const isGoalsView = activeTab === "Budgeting Goals";
    const activePlan = plannerOverview?.active_plan || {};
    const currentStatus = plannerOverview?.current_status || {};
    const budgetSummary = currentStatus.summary || {};
    const categoryStatuses = currentStatus.categories || [];
    const hasActivePlan = activePlan.status !== "missing" && currentStatus.status !== "missing";
    const overspentCategories = categoryStatuses.filter((item) => item.status === "overspent");
    const atRiskCategories = categoryStatuses.filter((item) => item.status === "at_risk");
    const onTrackCount = categoryStatuses.filter((item) => item.status === "on_track").length;
    const savingsStatus = categoryStatuses.find((item) => item.category_name === "Savings") || null;

    return (
      <>
        <section className="budgeting-overview-band">
          <article className="panel budgeting-overview-card">
            <div className="panel-header">
              <div>
                <p className="section-label">Active Budget Plan</p>
                <h3>{hasActivePlan ? "Live plan visibility" : "No active budget yet"}</h3>
              </div>
              {hasActivePlan ? (
                <span className="budget-pill">{formatBudgetPeriod(activePlan.period_start, activePlan.period_end)}</span>
              ) : null}
            </div>

            {plannerOverviewError ? (
              <div className="chat-error chat-error--compact" role="alert">
                {plannerOverviewError}
              </div>
            ) : isLoadingPlannerOverview ? (
              <p className="panel-note">Loading the active plan and live budget status...</p>
            ) : hasActivePlan ? (
              <>
                <p className="panel-note">
                  The Budgeting tab now keeps the saved plan visible while you talk to the planner agent, so it’s easy to compare the active budget against any proposed draft.
                </p>
                <div className="budgeting-overview-metrics">
                  <div className="mini-stat-card__col">
                    <span className="mini-stat-card__label">Spend So Far</span>
                    <strong>{currency(budgetSummary.total_actual || 0)}</strong>
                  </div>
                  <div className="mini-stat-card__col">
                    <span className="mini-stat-card__label">Budget Left</span>
                    <strong>{currency(budgetSummary.total_remaining || 0)}</strong>
                  </div>
                  <div className="mini-stat-card__col">
                    <span className="mini-stat-card__label">Budget Limit</span>
                    <strong>{currency(budgetSummary.total_target || 0)}</strong>
                  </div>
                  <div className="mini-stat-card__col">
                    <span className="mini-stat-card__label">Utilization</span>
                    <strong>{Number(budgetSummary.utilization_pct || 0).toFixed(1)}%</strong>
                  </div>
                  <div className="mini-stat-card__col">
                    <span className="mini-stat-card__label">Savings Progress</span>
                    <strong>
                      {savingsStatus ? currency(savingsStatus.actual_amount) : "Not tracked"}
                    </strong>
                    <p className="budgeting-overview-metric-note">
                      {savingsStatus
                        ? `${currency(savingsStatus.target_amount)} target · ${currency(savingsStatus.remaining_amount)} remaining`
                        : "Add a Savings target to track progress here."}
                    </p>
                  </div>
                </div>
              </>
            ) : (
              <p className="panel-note">
                There isn’t an active budget plan yet. Use the planner chat to create a draft, revise it if needed, and approve it to save the first active budget.
              </p>
            )}
          </article>
        </section>

        <section className="budgeting-stage">
          <div className="budgeting-active-plan-stack">
            {!isGoalsView ? (
              <article className="panel budgeting-workspace-card">
                <div className="panel-header">
                  <div>
                    <p className="section-label">Budgeting Workspace</p>
                    <h3>Execution plan</h3>
                  </div>
                </div>
                <p className="panel-note">
                  Use the planner chat to turn this window into a working monthly budget, revise category targets, and save the plan once you approve it.
                </p>
                <div className="merchant-list">
                  <div className="merchant-row">
                    <span>Ask for a fresh draft</span>
                    <strong>Create a one-month budget</strong>
                  </div>
                  <div className="merchant-row">
                    <span>Revise the proposal</span>
                    <strong>Raise, lower, or protect categories</strong>
                  </div>
                  <div className="merchant-row">
                    <span>Finalize when ready</span>
                    <strong>Approve the budget to save it</strong>
                  </div>
                </div>
              </article>
            ) : null}

            <article className="panel budgeting-targets-card">
              <div className="panel-header">
                <div>
                  <p className="section-label">Active Plan Detail</p>
                  <h3>{hasActivePlan ? "Category targets and live status" : "Waiting for first saved plan"}</h3>
                </div>
              </div>

              {hasActivePlan ? (
                <div className="budget-target-list">
                  {categoryStatuses.map((item) => (
                    <div key={item.category_name} className="budget-target-row">
                      <div className="budget-target-row-head">
                        <strong>{item.category_name}</strong>
                        <span className={`budget-status-pill ${getBudgetCategoryTone(item.status)}`}>
                          {item.status.replace("_", " ")}
                        </span>
                      </div>
                      <div className="budget-target-bar-shell">
                        <div className="budget-target-bar-meta">
                          <span>Spent {currency(item.actual_amount)}</span>
                          <strong>{Number(item.utilization_pct || 0).toFixed(1)}%</strong>
                        </div>
                        <div className="category-track budget-target-track">
                          <div
                            className={`category-fill budget-target-fill ${getBudgetCategoryTone(item.status)}`}
                            style={{ width: `${Math.max(6, Math.min(Number(item.utilization_pct || 0), 100))}%` }}
                          />
                        </div>
                      </div>
                      <div className="budget-target-row-metrics">
                        <span>Target {currency(item.target_amount)}</span>
                        <span>Left {currency(item.remaining_amount)}</span>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="panel-note">
                  Once a budget is approved, its category targets and live spend status will appear here for quick reference.
                </p>
              )}
            </article>

            <article className="panel budgeting-health-card">
              <div className="panel-header">
                <div>
                  <p className="section-label">Plan Health</p>
                  <h3>{hasActivePlan ? "What needs attention" : "Planner guidance"}</h3>
                </div>
              </div>
              {hasActivePlan ? (
                <div className="merchant-list">
                  <div className="merchant-row">
                    <span>Overspent categories</span>
                    <strong>{overspentCategories.length}</strong>
                  </div>
                  <div className="merchant-row">
                    <span>At-risk categories</span>
                    <strong>{atRiskCategories.length}</strong>
                  </div>
                  <div className="merchant-row">
                    <span>On-track categories</span>
                    <strong>{onTrackCount}</strong>
                  </div>
                </div>
              ) : (
                <div className="merchant-list">
                  <div className="merchant-row">
                    <span>Best first step</span>
                    <strong>Create a new draft</strong>
                  </div>
                  <div className="merchant-row">
                    <span>Then</span>
                    <strong>Revise until the targets feel realistic</strong>
                  </div>
                  <div className="merchant-row">
                    <span>Finally</span>
                    <strong>Approve to activate the plan</strong>
                  </div>
                </div>
              )}
            </article>

          </div>

          <ChatPanel
            card={selectedCard}
            analysisWindow={selectedWindow}
            mode="planner"
            onPlannerStateChange={handlePlannerStateChange}
          />
        </section>
      </>
    );
  }

  return (
    <div className="app-shell">
      <aside className={`sidebar ${isSidebarExpanded ? "is-expanded" : ""}`}>
        <div className="sidebar-rail-stack">
          <button
            className="sidebar-toggle-button"
            type="button"
            onClick={() => setIsSidebarExpanded((current) => !current)}
            aria-expanded={isSidebarExpanded}
            aria-controls="sidebar-navigation"
            title={isSidebarExpanded ? "Collapse sidebar" : "Expand sidebar"}
          >
            {isSidebarExpanded ? <ChevronLeft size={16} /> : <ChevronRight size={16} />}
            <span className="sr-only">{isSidebarExpanded ? "Collapse sidebar" : "Expand sidebar"}</span>
          </button>
          <div className="sidebar-rail" aria-label="Quick navigation">
            {railNavItems.map((item) => {
              const Icon = item.icon;
              const isActive = item.tab === "Budgeting Goals" ? isBudgetingTab : activeTab === item.tab;
              return (
                <button
                  key={item.label}
                  className={`sidebar-rail-button ${isActive ? "active" : ""}`}
                  type="button"
                  onClick={() => setActiveTab(item.tab)}
                  aria-label={item.label}
                  title={item.label}
                >
                  <Icon size={15} />
                </button>
              );
            })}
          </div>
        </div>
        <div className="sidebar-content" id="sidebar-navigation">
          <div className="brand-block">
            <p className="eyebrow">Personal Finance Copilot</p>
          </div>

          <nav className="nav-groups">
            <button
              className={`nav-item ${activeTab === "Overview" ? "active" : ""}`}
              type="button"
              onClick={() => setActiveTab("Overview")}
            >
              Overview
            </button>
            <button
              className={`nav-item ${activeTab === "Card Details" ? "active" : ""}`}
              type="button"
              onClick={() => setActiveTab("Card Details")}
            >
              Card Details
            </button>
            <button
              className={`nav-item ${activeTab === "Spending Analysis" ? "active" : ""}`}
              type="button"
              onClick={() => setActiveTab("Spending Analysis")}
            >
              Spending Analysis
            </button>
            <div className="nav-section">
              <button
                className={`nav-item ${isBudgetingTab ? "active" : ""}`}
                type="button"
                onClick={() => setActiveTab("Budgeting Goals")}
              >
                Budgeting
              </button>
              <div className="nav-subgroups">
                <button
                  className={`nav-sub-item ${activeTab === "Budgeting Goals" ? "active" : ""}`}
                  type="button"
                  onClick={() => setActiveTab("Budgeting Goals")}
                >
                  Goal
                </button>
                <button
                  className={`nav-sub-item ${activeTab === "Budgeting Plan" ? "active" : ""}`}
                  type="button"
                  onClick={() => setActiveTab("Budgeting Plan")}
                >
                  Plan
                </button>
              </div>
            </div>
          </nav>
        </div>
      </aside>

      <main className="main-panel">
        <header className="hero">
          <div>
            <h2>Financial Snapshot</h2>
          </div>
          <div className="hero-actions">
            <div className="window-picker">
                <button
                  className="ghost-button window-picker-trigger"
                  type="button"
                  onClick={() => {
                  setWindowDraftRange(undefined);
                  setWindowPickerOpen((current) => !current);
                }}
                aria-expanded={windowPickerOpen}
              >
                {windowPreset === "custom"
                  ? selectedWindow.label
                  : WINDOW_PRESETS.find((preset) => preset.value === windowPreset)?.label || selectedWindow.label}
              </button>
              {windowPickerOpen ? (
                <div className="window-picker-popover">
                  <div className="window-picker-presets" role="tablist" aria-label="Date window presets">
                    {WINDOW_PRESETS.map((preset) => (
                      <button
                        key={preset.value}
                        className={`window-pill ${windowPreset === preset.value ? "active" : ""}`}
                        type="button"
                        onClick={() => handlePresetSelect(preset.value)}
                      >
                        {preset.label}
                      </button>
                    ))}
                  </div>
                  <p className="window-picker-note">Click a start date, then an end date to apply a custom range.</p>
                  <DayPicker
                    mode="range"
                    selected={windowDraftRange}
                    onSelect={handleWindowSelect}
                    min={1}
                    numberOfMonths={2}
                    showOutsideDays
                  />
                </div>
              ) : null}
            </div>
          </div>
        </header>

        {loading ? (
          <LoadingState />
        ) : error ? (
          <div className="chat-error" role="alert">
            {error}
          </div>
        ) : selectedCard ? (
          <div className={`dashboard-scene ${isRefreshingDashboard ? "is-refreshing" : ""}`}>
            <div className="dashboard-refresh-overlay" aria-hidden="true">
              <span />
              <p>Updating window...</p>
            </div>
            {isAssistantView ? (
              <section className="assistant-stage">
                <ChatPanel
                  card={selectedCard}
                  analysisWindow={selectedWindow}
                  seedMessage={assistantSeed?.text || ""}
                  seedMessageId={assistantSeed?.id || ""}
                  onSeedConsumed={clearAssistantSeed}
                />
              </section>
            ) : (
              <>
                {activeTab === "Overview" ? renderOverviewTab() : null}
                {activeTab === "Card Details" ? renderCardsTab() : null}
                {activeTab === "Spending Analysis" ? renderSpendingAnalysisTab() : null}
                {isBudgetingTab ? renderBudgetingTab() : null}
              </>
            )}
          </div>
        ) : null}

        {!isAssistantView ? (
          <button
            className="assistant-launcher"
            type="button"
            onClick={handleAssistantOpen}
            aria-label="Open assistant"
            title="Open assistant"
          >
            <MessageCircle className="assistant-launcher-icon" size={16} aria-hidden="true" />
            <span className="assistant-launcher-label">Assistant</span>
          </button>
        ) : null}
      </main>
    </div>
  );
}

export default App;
