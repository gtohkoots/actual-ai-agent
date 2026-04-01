import { useEffect, useMemo, useRef, useState } from "react";
import { Sparkles } from "lucide-react";
import { DayPicker } from "react-day-picker";
import { startOfMonth, subDays } from "date-fns";
import "react-day-picker/style.css";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  Cell,
  CartesianGrid,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import ChatPanel from "./components/ChatPanel";
import { fetchAccounts, fetchDashboardOverview } from "./api/dashboard";

const navItems = ["Overview", "Cards"];
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

function formatChartDay(value) {
  if (!value) return "";
  const date = new Date(`${value}T00:00:00`);
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
  }).format(date);
}

function PortfolioTrendTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;

  const income = payload.find((entry) => entry.dataKey === "income")?.value || 0;
  const expense = payload.find((entry) => entry.dataKey === "expense")?.value || 0;
  const net = payload.find((entry) => entry.dataKey === "net")?.value || 0;

  return (
    <div className="chart-tooltip">
      <strong>{formatChartDay(label)}</strong>
      <span>Income: {currency(income)}</span>
      <span>Spend: {currency(expense)}</span>
      <span>Net: {currency(net)}</span>
    </div>
  );
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
  const [error, setError] = useState("");
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
  const overviewChartColors = ["#1f5c4d", "#aa7d2d", "#a04b2f", "#5a3d18", "#1b2c55"];
  const portfolioBalances = cards.map((card) => card.balanceCurrent);
  const highestBalanceCard = cards.reduce(
    (best, card) => (!best || card.balanceCurrent > best.balanceCurrent ? card : best),
    null
  );
  const averageBalance = portfolioBalances.length
    ? portfolioBalances.reduce((sum, value) => sum + value, 0) / portfolioBalances.length
    : 0;
  const overviewStats = dashboard
    ? [
        { label: "Portfolio Current Balance", value: portfolio.summary?.totalBalance || currency(0), note: "Live balances across all cards on file" },
        { label: "Active Cards", value: String(cards.length).padStart(2, "0"), note: "Cards currently on file" },
        {
          label: "Highest Balance Card",
          value: highestBalanceCard ? highestBalanceCard.name : "n/a",
          note: highestBalanceCard ? currency(highestBalanceCard.balanceCurrent) : "No cards loaded",
        },
        {
          label: "Average Card Balance",
          value: currency(averageBalance),
          note: "Average across live balances",
        },
      ]
    : [];
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
    setActiveTab("Cards");
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

  function renderOverviewTab() {
    const overviewHeatmap = portfolio.dailyHeatmap || [];
    const heatmapMaxWeek = Math.max(...overviewHeatmap.map((day) => day.week), 0);
    const heatmapMaxAmount = Math.max(...overviewHeatmap.map((day) => day.amount), 1);

    return (
      <>
        <section className="stats-grid">
          {overviewStats.map((stat) => (
            <article key={stat.label} className="stat-card">
              <p className="section-label">{stat.label}</p>
              <h3>{stat.value}</h3>
              <p>{stat.note}</p>
            </article>
          ))}
        </section>

        <section className="panel overview-analysis-panel">
          <div className="panel-header">
            <div>
              <p className="section-label">Income + Spend Analysis</p>
              <h3>Portfolio movement over time</h3>
            </div>
            <span className="panel-note">All cards in range</span>
          </div>
          <div className="overview-summary-row">
            <div className="overview-summary-chip">
              <span>Total income</span>
              <strong>{portfolio.summary?.totalIncome || currency(0)}</strong>
            </div>
            <div className="overview-summary-chip">
              <span>Total spend</span>
              <strong>{portfolio.summary?.totalSpend || currency(0)}</strong>
            </div>
            <div className="overview-summary-chip">
              <span>Net cash flow</span>
              <strong>{portfolio.summary?.netCashFlow || currency(0)}</strong>
            </div>
          </div>
          <div className="portfolio-chart-shell">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={(portfolio.series || []).slice(-14)} margin={{ top: 10, right: 16, left: -8, bottom: 0 }}>
                <defs>
                  <linearGradient id="incomeGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#1f5c4d" stopOpacity={0.38} />
                    <stop offset="95%" stopColor="#1f5c4d" stopOpacity={0.05} />
                  </linearGradient>
                  <linearGradient id="expenseGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#a04b2f" stopOpacity={0.32} />
                    <stop offset="95%" stopColor="#a04b2f" stopOpacity={0.04} />
                  </linearGradient>
                </defs>
                <CartesianGrid stroke="rgba(27, 26, 23, 0.08)" vertical={false} />
                <XAxis
                  dataKey="date"
                  tickFormatter={formatChartDay}
                  tickLine={false}
                  axisLine={false}
                  minTickGap={16}
                />
                <YAxis
                  tickFormatter={(value) => `$${value.toFixed(0)}`}
                  tickLine={false}
                  axisLine={false}
                  width={46}
                />
                <Tooltip content={<PortfolioTrendTooltip />} />
                <Area
                  type="monotone"
                  dataKey="income"
                  stroke="#1f5c4d"
                  fill="url(#incomeGradient)"
                  strokeWidth={2.2}
                  dot={false}
                />
                <Area
                  type="monotone"
                  dataKey="expense"
                  stroke="#a04b2f"
                  fill="url(#expenseGradient)"
                  strokeWidth={2.2}
                  dot={false}
                />
                <Area
                  type="monotone"
                  dataKey="net"
                  stroke="#aa7d2d"
                  fill="none"
                  strokeWidth={2}
                  dot={false}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
          <div className="overview-analysis-footer">
            <div className="overview-summary-row overview-summary-row--compact">
              <div className="overview-summary-chip">
                <span>Selected Window</span>
                <strong>{selectedWindow.label}</strong>
              </div>
              <div className="overview-summary-chip">
                <span>Highest Category</span>
                <strong>{portfolio.topCategories?.[0]?.category || "n/a"}</strong>
              </div>
              <div className="overview-summary-chip">
                <span>Peak Spend Day</span>
                <strong>
                  {portfolio.dailyHeatmap?.length
                    ? portfolio.dailyHeatmap.reduce((peak, day) => (day.amount > peak.amount ? day : peak), portfolio.dailyHeatmap[0]).date
                    : "n/a"}
                </strong>
              </div>
            </div>

            <div className="overview-analysis-subgrid">
              <div className="overview-analysis-subcard">
                <div className="panel-header panel-header--compact">
                  <div>
                    <p className="section-label">Portfolio Spend Mix</p>
                    <h4>Where the portfolio spend is concentrated</h4>
                  </div>
                  <span className="panel-note">Selected window</span>
                </div>
                <div className="overview-donut-shell overview-donut-shell--compact">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Tooltip content={<SimpleListTooltip />} />
                      <Pie
                        data={portfolio.categoryMix || []}
                        dataKey="amount"
                        nameKey="category"
                        innerRadius={52}
                        outerRadius={84}
                        paddingAngle={3}
                      >
                        {(portfolio.categoryMix || []).map((entry, index) => (
                          <Cell key={entry.category} fill={overviewChartColors[index % overviewChartColors.length]} />
                        ))}
                      </Pie>
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="overview-analysis-subcard">
                <div className="panel-header panel-header--compact">
                  <div>
                    <p className="section-label">Daily Heatmap</p>
                    <h4>Spend intensity by day</h4>
                  </div>
                  <span className="panel-note">Optional detail</span>
                </div>
                <div
                  className="heatmap-grid"
                  style={{ gridTemplateColumns: `repeat(${heatmapMaxWeek + 1}, minmax(0, 1fr))` }}
                >
                  {overviewHeatmap.map((day) => {
                    const intensity = day.amount / heatmapMaxAmount;
                    return (
                      <div
                        key={day.date}
                        className="heatmap-cell"
                        style={{
                          opacity: 0.2 + intensity * 0.8,
                          backgroundColor: "#1f5c4d",
                          gridColumn: day.week + 1,
                          gridRow: day.weekday + 1,
                        }}
                        title={`${day.date} · ${currency(day.amount)}`}
                      >
                        <span>{new Date(`${day.date}T00:00:00`).getDate()}</span>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>

          </div>
        </section>

        <section className="overview-grid">
          <article className="panel overview-comparison-panel">
            <div className="panel-header">
              <div>
                <p className="section-label">Comparison Views</p>
                <h3>Cards and category shifts in one compact row</h3>
              </div>
              <span className="panel-note">Vertical bars</span>
            </div>
            <div className="overview-comparison-grid">
              <div className="overview-comparison-card">
                <div className="panel-header panel-header--compact">
                  <div>
                    <p className="section-label">Card Comparison</p>
                    <h4>Spend, income, balance</h4>
                  </div>
                </div>
                <div className="overview-chart-shell overview-chart-shell--compact">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={portfolio.accountComparison || []} layout="vertical" margin={{ top: 4, right: 16, left: 8, bottom: 4 }}>
                      <CartesianGrid stroke="rgba(27, 26, 23, 0.08)" horizontal={false} />
                      <XAxis type="number" tickLine={false} axisLine={false} />
                      <YAxis type="category" dataKey="accountName" tickLine={false} axisLine={false} width={98} />
                      <Tooltip content={<SimpleListTooltip />} />
                      <Bar dataKey="spend" fill="#a04b2f" radius={[0, 999, 999, 0]} barSize={12} />
                      <Bar dataKey="income" fill="#1f5c4d" radius={[0, 999, 999, 0]} barSize={12} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="overview-comparison-card">
                <div className="panel-header panel-header--compact">
                  <div>
                    <p className="section-label">Top Movers</p>
                    <h4>Biggest category shifts</h4>
                  </div>
                </div>
                <div className="overview-chart-shell overview-chart-shell--compact">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={(portfolio.topMovers || []).slice(0, 5)} layout="vertical" margin={{ top: 4, right: 16, left: 8, bottom: 4 }}>
                      <CartesianGrid stroke="rgba(27, 26, 23, 0.08)" horizontal={false} />
                      <XAxis type="number" tickLine={false} axisLine={false} />
                      <YAxis type="category" dataKey="category" tickLine={false} axisLine={false} width={98} />
                      <Tooltip content={<SimpleListTooltip />} />
                      <Bar dataKey="delta" fill="#aa7d2d" radius={[0, 999, 999, 0]} barSize={12} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
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

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-block">
          <p className="eyebrow">Finance Agent</p>
          <h1>Cards, spend, and AI guidance in one place.</h1>
          <p className="sidebar-copy">
            A dashboard-first workspace where the user explores card activity and asks grounded questions
            without leaving the context of the data.
          </p>
        </div>

        <nav className="nav-groups">
          {navItems.map((item) => (
            <button
              key={item}
              className={`nav-item ${item === activeTab ? "active" : ""}`}
              type="button"
              onClick={() => setActiveTab(item)}
            >
              {item}
            </button>
          ))}
        </nav>

        <section className="context-panel">
          <p className="section-label">Active Context</p>
          {selectedCard ? (
            <div className="context-chips">
              <span className="chip">Card: {selectedCard.context.card}</span>
              <span className="chip">Window: {dashboard?.selected_window || selectedCard.context.dateRange}</span>
              <span className="chip">Focus: {selectedCard.context.focus}</span>
              <span className="chip">PID: {selectedCard.id.slice(0, 8)}...</span>
            </div>
          ) : (
            <p className="sidebar-copy">Loading accounts from the backend...</p>
          )}
        </section>
      </aside>

      <main className="main-panel">
        <header className="hero">
          <div>
            <p className="eyebrow">{dashboard?.month_label || "Loading month..."}</p>
            <h2>Card spending cockpit</h2>
            <p className="hero-copy">
              Switch cards, inspect category drift, and carry that context directly into the AI conversation.
            </p>
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
            <button className="primary-button" type="button">
              Generate weekly insight
            </button>
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
            ) : activeTab === "Overview" ? (
              renderOverviewTab()
            ) : (
              renderCardsTab()
            )}
          </div>
        ) : null}

        {!isAssistantView ? (
          <button
            className="assistant-launcher"
            type="button"
            onClick={handleAssistantOpen}
            aria-label="Open AI assistant"
            title="Open AI assistant"
          >
            <span className="assistant-launcher-ring" aria-hidden="true">
              <span className="assistant-launcher-core">
                <Sparkles size={22} />
              </span>
            </span>
            <span className="assistant-launcher-label">AI</span>
          </button>
        ) : null}
      </main>
    </div>
  );
}

export default App;
