import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ChartPie, ChevronLeft, ChevronRight, CreditCard, LayoutDashboard, MessageCircle, PiggyBank, RefreshCw, TrendingUp, Upload } from "lucide-react";
import { DayPicker } from "react-day-picker";
import { startOfMonth, subDays } from "date-fns";
import "react-day-picker/style.css";
import {
  Bar,
  BarChart,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import AssistantShell from "./components/AssistantShell";
import ChatPanel from "./components/ChatPanel";
import CollapsiblePanel from "./components/CollapsiblePanel";
import { fetchAccounts, fetchDashboardOverview } from "./api/dashboard";
import {
  fetchIndustryExposure,
  fetchInvestmentsOverview,
  fetchSingleNameExposure,
  importFidelityPositionsCsv,
  refreshIndustryExposure,
  refreshInvestmentExposure,
} from "./investments/api";
import { fetchPlannerOverview } from "./planner/api";

const railNavItems = [
  { label: "Overview", icon: LayoutDashboard, tab: "Overview" },
  { label: "Card Details", icon: CreditCard, tab: "Card Details" },
  { label: "Spending Analysis", icon: TrendingUp, tab: "Spending Analysis" },
  { label: "Budgeting", icon: PiggyBank, tab: "Budgeting Goals" },
  { label: "Investment", icon: ChartPie, tab: "Investment" },
];
const PLANNER_BUCKET_OPTIONS = [
  { id: "inflow", label: "Inflow", description: "Counts as income for budget planning and can raise the total budget cap." },
  { id: "recurring_inflow", label: "Recurring Inflow", description: "Stable income the planner can trust more heavily when forecasting future periods." },
  { id: "savings", label: "Savings", description: "Transfers or categories that represent intentional saving rather than everyday spend." },
  { id: "exclude", label: "Exclude", description: "Ignore this category when generating budget recommendations." },
  { id: "fixed", label: "Fixed", description: "Recurring expenses that should stay close to baseline, like rent or subscriptions." },
  { id: "essential", label: "Essential", description: "Needs that can flex a bit, but usually should be protected in the budget." },
  { id: "discretionary", label: "Discretionary", description: "Flexible spend that the planner can trim first to preserve savings targets." },
];
const DEFAULT_BUCKET_ASSIGNMENTS = {
  "Rent Transfer": "inflow",
  income: "recurring_inflow",
  "One-time deposit": "inflow",
  Paycheck: "recurring_inflow",
  "Starting Balances": "inflow",
  Savings: "savings",
  "Ignored - expense": "exclude",
  "Internal Transfer Expense": "exclude",
  "Internal Transfer Income": "exclude",
  "Lease Buyout": "exclude",
  "One-time expense": "exclude",
  "Pay Credit Card": "exclude",
  Bills: "fixed",
  Rent: "fixed",
  Subscription: "fixed",
  Grocery: "essential",
  "Bills (Flexible)": "essential",
  Gas: "essential",
  Home: "essential",
  Work: "essential",
};
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
const INVESTMENT_COLORS = ["#1f5c4d", "#aa7d2d", "#a04b2f", "#214d73", "#7d5ba6", "#5a3d18", "#6f8e6d", "#8a6250"];

function currency(value) {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 2,
  }).format(value || 0);
}

function signedCurrency(value) {
  const amount = Number(value || 0);
  return `${amount >= 0 ? "+" : ""}${currency(amount)}`;
}

function signedPercent(value) {
  const amount = Number(value || 0);
  return `${amount >= 0 ? "+" : ""}${amount.toFixed(2)}%`;
}

function gainTone(value) {
  return Number(value || 0) >= 0 ? "is-positive" : "is-negative";
}

function formatSnapshotTimestamp(value) {
  if (!value) return "Not imported yet";
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(new Date(value));
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

function ExposureTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const item = payload[0]?.payload;
  if (!item) return null;

  return (
    <div className="chart-tooltip investment-exposure-tooltip">
      <strong>{item.symbol} · {item.name || "Single-name exposure"}</strong>
      <span>{currency(item.exposure_value)} · {Number(item.percent_of_portfolio || 0).toFixed(2)}% of portfolio</span>
      {(item.contributions || []).slice(0, 4).map((contribution) => (
        <span key={`${item.symbol}-${contribution.source_symbol}`}>
          {contribution.source_symbol}: {currency(contribution.exposure_value)} from {Number(contribution.weight_percent || 0).toFixed(2)}%
        </span>
      ))}
    </div>
  );
}

function IndustryTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const item = payload[0]?.payload;
  if (!item) return null;

  return (
    <div className="chart-tooltip investment-exposure-tooltip">
      <strong>{item.industry}</strong>
      <span>{item.sector} · {Number(item.percent_of_portfolio || 0).toFixed(2)}% · {currency(item.exposure_value)}</span>
      {(item.top_companies || []).slice(0, 4).map((company) => (
        <span key={`${item.industry}-${company.symbol}`}>
          {company.symbol}: {Number(company.percent_of_portfolio || 0).toFixed(2)}% · {currency(company.exposure_value)}
        </span>
      ))}
    </div>
  );
}

function ExposureAxisTick({ x, y, payload }) {
  const item = payload?.payload || {};
  const symbol = item.symbol || payload?.value || "";
  const name = item.name || "";

  return (
    <g transform={`translate(${x},${y})`}>
      <text x={0} y={-3} textAnchor="end" fill="#2f2a24" fontSize={12} fontWeight={800}>
        {symbol}
      </text>
      {name ? (
        <text x={0} y={12} textAnchor="end" fill="#8a8176" fontSize={10}>
          {name.length > 20 ? `${name.slice(0, 20)}...` : name}
        </text>
      ) : null}
    </g>
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

function getDefaultPlannerBucket(categoryName) {
  return DEFAULT_BUCKET_ASSIGNMENTS[categoryName] || "discretionary";
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
  const [investmentsOverview, setInvestmentsOverview] = useState(null);
  const [investmentsError, setInvestmentsError] = useState("");
  const [isLoadingInvestments, setIsLoadingInvestments] = useState(false);
  const [isImportingInvestments, setIsImportingInvestments] = useState(false);
  const [investmentImportFile, setInvestmentImportFile] = useState(null);
  const [investmentAsOfDate, setInvestmentAsOfDate] = useState(formatLocalDate(new Date()));
  const [investmentRefreshToken, setInvestmentRefreshToken] = useState(0);
  const [investmentExposure, setInvestmentExposure] = useState(null);
  const [investmentExposureError, setInvestmentExposureError] = useState("");
  const [isLoadingInvestmentExposure, setIsLoadingInvestmentExposure] = useState(false);
  const [isRefreshingInvestmentExposure, setIsRefreshingInvestmentExposure] = useState(false);
  const [investmentExposureThreshold, setInvestmentExposureThreshold] = useState(1);
  const [investmentExposureChartType, setInvestmentExposureChartType] = useState("bar");
  const [industryExposure, setIndustryExposure] = useState(null);
  const [industryExposureError, setIndustryExposureError] = useState("");
  const [isLoadingIndustryExposure, setIsLoadingIndustryExposure] = useState(false);
  const [isRefreshingIndustryExposure, setIsRefreshingIndustryExposure] = useState(false);
  const [industryExposureChartType, setIndustryExposureChartType] = useState("bar");
  const [isSidebarExpanded, setIsSidebarExpanded] = useState(false);
  const [isAssistantShellOpen, setIsAssistantShellOpen] = useState(false);
  const [assistantShellMode, setAssistantShellMode] = useState("analysis");
  const [plannerCategoryAssignments, setPlannerCategoryAssignments] = useState({});
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

  const isBudgetingTab = activeTab === "Budgeting Goals" || activeTab === "Budgeting Plan";
  const isInvestmentTab = activeTab === "Investment";
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

  useEffect(() => {
    let cancelled = false;

    async function loadInvestmentsOverview() {
      if (!isInvestmentTab) return;
      setIsLoadingInvestments(true);
      setInvestmentsError("");
      try {
        const overview = await fetchInvestmentsOverview();
        if (!cancelled) {
          setInvestmentsOverview(overview);
        }
      } catch (err) {
        if (!cancelled) {
          setInvestmentsError(err instanceof Error ? err.message : "Failed to load investment overview");
        }
      } finally {
        if (!cancelled) {
          setIsLoadingInvestments(false);
        }
      }
    }

    void loadInvestmentsOverview();
    return () => {
      cancelled = true;
    };
  }, [isInvestmentTab, investmentRefreshToken]);

  useEffect(() => {
    let cancelled = false;

    async function loadInvestmentExposure() {
      if (!isInvestmentTab) return;
      setIsLoadingInvestmentExposure(true);
      setInvestmentExposureError("");
      try {
        const exposure = await fetchSingleNameExposure({ minPercent: investmentExposureThreshold, limit: 15 });
        if (!cancelled) {
          setInvestmentExposure(exposure);
        }
      } catch (err) {
        if (!cancelled) {
          setInvestmentExposureError(err instanceof Error ? err.message : "Failed to load investment exposure");
        }
      } finally {
        if (!cancelled) {
          setIsLoadingInvestmentExposure(false);
        }
      }
    }

    void loadInvestmentExposure();
    return () => {
      cancelled = true;
    };
  }, [isInvestmentTab, investmentExposureThreshold, investmentRefreshToken]);

  useEffect(() => {
    let cancelled = false;

    async function loadIndustryExposure() {
      if (!isInvestmentTab) return;
      setIsLoadingIndustryExposure(true);
      setIndustryExposureError("");
      try {
        const exposure = await fetchIndustryExposure();
        if (!cancelled) {
          setIndustryExposure(exposure);
        }
      } catch (err) {
        if (!cancelled) {
          setIndustryExposureError(err instanceof Error ? err.message : "Failed to load industry exposure");
        }
      } finally {
        if (!cancelled) {
          setIsLoadingIndustryExposure(false);
        }
      }
    }

    void loadIndustryExposure();
    return () => {
      cancelled = true;
    };
  }, [isInvestmentTab, investmentRefreshToken]);

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

  function handleAssistantOpen(nextMode) {
    setAssistantShellMode(nextMode || (isBudgetingTab ? "planner" : "analysis"));
    setIsAssistantShellOpen(true);
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
    setAssistantShellMode("analysis");
    setIsAssistantShellOpen(true);
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

  async function handleInvestmentImport(event) {
    event.preventDefault();
    if (!investmentImportFile || isImportingInvestments) return;

    const form = event.currentTarget;
    setIsImportingInvestments(true);
    setInvestmentsError("");
    try {
      await importFidelityPositionsCsv(investmentImportFile, investmentAsOfDate);
      setInvestmentImportFile(null);
      form.reset();
      setInvestmentRefreshToken((current) => current + 1);
    } catch (err) {
      setInvestmentsError(err instanceof Error ? err.message : "Failed to import Fidelity positions");
    } finally {
      setIsImportingInvestments(false);
    }
  }

  async function handleInvestmentExposureRefresh() {
    if (isRefreshingInvestmentExposure) return;
    setIsRefreshingInvestmentExposure(true);
    setInvestmentExposureError("");
    try {
      console.info("[investments] Starting exposure refresh", { force: true, threshold: investmentExposureThreshold });
      const refreshResult = await refreshInvestmentExposure({ force: true });
      console.info("[investments] Exposure refresh result", refreshResult);
      const exposure = await fetchSingleNameExposure({ minPercent: investmentExposureThreshold, limit: 15 });
      console.info("[investments] Single-name exposure result", {
        status: exposure?.status,
        itemCount: exposure?.items?.length || 0,
        unresolvedCount: exposure?.unresolved?.length || 0,
        summary: exposure?.summary || {},
      });
      setInvestmentExposure(exposure);
      if (refreshResult.failed?.length) {
        setInvestmentExposureError(
          `Alpha Vantage refresh failed for ${refreshResult.failed.map((item) => item.symbol).join(", ")}. Check backend logs for details.`
        );
      }
    } catch (err) {
      console.error("[investments] Exposure refresh failed", err);
      setInvestmentExposureError(err instanceof Error ? err.message : "Failed to refresh investment exposure");
    } finally {
      setIsRefreshingInvestmentExposure(false);
    }
  }

  async function handleIndustryExposureRefresh() {
    if (isRefreshingIndustryExposure) return;
    setIsRefreshingIndustryExposure(true);
    setIndustryExposureError("");
    try {
      console.info("[investments] Starting industry exposure refresh", { force: true });
      const refreshResult = await refreshIndustryExposure({ force: true });
      console.info("[investments] Industry exposure refresh result", refreshResult);
      const exposure = await fetchIndustryExposure();
      console.info("[investments] Industry exposure result", {
        status: exposure?.status,
        itemCount: exposure?.items?.length || 0,
        summary: exposure?.summary || {},
      });
      setIndustryExposure(exposure);
    } catch (err) {
      console.error("[investments] Industry exposure refresh failed", err);
      setIndustryExposureError(err instanceof Error ? err.message : "Failed to refresh industry exposure");
    } finally {
      setIsRefreshingIndustryExposure(false);
    }
  }

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

        <section className="transactions-and-chat">
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

  function renderInvestmentTab() {
    const summary = investmentsOverview?.summary || {};
    const holdings = investmentsOverview?.holdings || [];
    const latestImport = investmentsOverview?.latest_import || null;
    const hasSnapshot = investmentsOverview?.status === "ready";
    const allocationData = holdings.filter((item) => Number(item.current_value || 0) > 0);
    const exposureItems = investmentExposure?.items || [];
    const exposureSummary = investmentExposure?.summary || {};
    const exposureChartData = exposureItems.map((item) => ({
      ...item,
      chartLabel: item.symbol,
    }));
    const industryItems = industryExposure?.items || [];
    const industrySummary = industryExposure?.summary || {};
    const industryChartData = industryItems.map((item) => ({
      ...item,
      chartLabel: item.industry,
    }));

    return (
      <section className="investment-page">
        <div className="investment-page__heading">
          <div>
            <p className="section-label">Investment Portfolio</p>
            <h3>{hasSnapshot ? "Fidelity positions snapshot" : "Import your Fidelity positions"}</h3>
            <p className="panel-note">
              {hasSnapshot
                ? `Snapshot as of ${latestImport?.as_of_date || "the latest import"} · refreshed ${formatSnapshotTimestamp(latestImport?.imported_at)}`
                : "Upload a Fidelity positions CSV to create your portfolio view."}
            </p>
          </div>
          <form className="investment-import-form" onSubmit={handleInvestmentImport}>
            <label className="investment-import-file">
              <Upload size={16} aria-hidden="true" />
              <span>{investmentImportFile?.name || "Choose Fidelity CSV"}</span>
              <input
                type="file"
                accept=".csv,text/csv"
                onChange={(event) => setInvestmentImportFile(event.target.files?.[0] || null)}
              />
            </label>
            <label className="investment-import-date">
              <span>As of</span>
              <input
                type="date"
                value={investmentAsOfDate}
                onChange={(event) => setInvestmentAsOfDate(event.target.value)}
              />
            </label>
            <button className="primary-button investment-import-button" type="submit" disabled={!investmentImportFile || isImportingInvestments}>
              {isImportingInvestments ? <RefreshCw className="spin" size={15} aria-hidden="true" /> : <Upload size={15} aria-hidden="true" />}
              {isImportingInvestments ? "Importing" : "Import"}
            </button>
          </form>
        </div>

        {investmentsError ? (
          <div className="chat-error" role="alert">
            {investmentsError}
          </div>
        ) : null}

        {isLoadingInvestments && !investmentsOverview ? (
          <LoadingState />
        ) : hasSnapshot ? (
          <>
            <div className="investment-metrics-grid">
              <article className="stat-card investment-metric investment-metric--primary">
                <p className="section-label">Portfolio Value</p>
                <h3>{currency(summary.total_value)}</h3>
                <p>{summary.holding_count} positions across {summary.account_count} account{summary.account_count === 1 ? "" : "s"}</p>
              </article>
              <article className="stat-card investment-metric">
                <p className="section-label">Today</p>
                <h3 className={gainTone(summary.day_gain_loss_amount)}>{signedCurrency(summary.day_gain_loss_amount)}</h3>
                <p className={gainTone(summary.day_gain_loss_amount)}>{signedPercent(summary.day_gain_loss_percent)}</p>
              </article>
              <article className="stat-card investment-metric">
                <p className="section-label">Total Gain / Loss</p>
                <h3 className={gainTone(summary.total_gain_loss_amount)}>{signedCurrency(summary.total_gain_loss_amount)}</h3>
                <p className={gainTone(summary.total_gain_loss_amount)}>{signedPercent(summary.total_gain_loss_percent)}</p>
              </article>
              <article className="stat-card investment-metric">
                <p className="section-label">Cost Basis</p>
                <h3>{currency(summary.cost_basis_total)}</h3>
                <p>Across positions with reported basis</p>
              </article>
              <article className="stat-card investment-metric">
                <p className="section-label">Cash Position</p>
                <h3>{currency(summary.cash_value)}</h3>
                <p>{summary.total_value ? ((summary.cash_value / summary.total_value) * 100).toFixed(2) : "0.00"}% of portfolio</p>
              </article>
            </div>

            <div className="investment-overview-grid">
              <article className="panel investment-allocation-panel">
                <div className="panel-header">
                  <div>
                    <p className="section-label">Holdings Allocation</p>
                    <h3>Where your portfolio is concentrated</h3>
                  </div>
                </div>
                <div className="investment-allocation-shell">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Tooltip content={<SimpleListTooltip />} />
                      <Pie data={allocationData} dataKey="current_value" nameKey="symbol" innerRadius={66} outerRadius={104} paddingAngle={2}>
                        {allocationData.map((entry, index) => (
                          <Cell key={`${entry.account_id}-${entry.symbol}`} fill={INVESTMENT_COLORS[index % INVESTMENT_COLORS.length]} />
                        ))}
                      </Pie>
                    </PieChart>
                  </ResponsiveContainer>
                  <div className="investment-allocation-center">
                    <strong>{holdings.length}</strong>
                    <span>positions</span>
                  </div>
                </div>
                <div className="investment-allocation-legend">
                  {allocationData.map((item, index) => (
                    <div key={`${item.account_id}-${item.symbol}`} className="investment-allocation-row">
                      <span className="investment-allocation-swatch" style={{ background: INVESTMENT_COLORS[index % INVESTMENT_COLORS.length] }} />
                      <strong>{item.symbol}</strong>
                      <span>{Number(item.percent_of_portfolio || 0).toFixed(2)}%</span>
                      <span>{currency(item.current_value)}</span>
                    </div>
                  ))}
                </div>
              </article>

              <article className="panel investment-exposure-panel">
                <div className="panel-header investment-exposure-header">
                  <div>
                    <p className="section-label">Single-Name Exposure</p>
                    <h3>Stocks inside your ETFs</h3>
                    <p className="panel-note">
                      {investmentExposure?.status === "ready"
                        ? `Resolved ${Number(exposureSummary.resolved_exposure_percent || 0).toFixed(2)}% · excludes ${Number(exposureSummary.excluded_percent || 0).toFixed(2)}% cash/fixed income`
                        : "Refresh ETF holdings to calculate look-through stock exposure."}
                    </p>
                  </div>
                  <div className="investment-exposure-actions">
                    <label>
                      <span>View</span>
                      <select value={investmentExposureChartType} onChange={(event) => setInvestmentExposureChartType(event.target.value)}>
                        <option value="bar">Bar</option>
                        <option value="pie">Pie</option>
                      </select>
                    </label>
                    <label>
                      <span>Threshold</span>
                      <select value={investmentExposureThreshold} onChange={(event) => setInvestmentExposureThreshold(Number(event.target.value))}>
                        <option value={0.5}>0.5%</option>
                        <option value={1}>1%</option>
                        <option value={2}>2%</option>
                      </select>
                    </label>
                    <button className="ghost-button" type="button" onClick={handleInvestmentExposureRefresh} disabled={isRefreshingInvestmentExposure}>
                      <RefreshCw className={isRefreshingInvestmentExposure ? "spin" : ""} size={15} aria-hidden="true" />
                      {isRefreshingInvestmentExposure ? "Refreshing" : "Refresh"}
                    </button>
                  </div>
                </div>
                {investmentExposureError ? (
                  <div className="chat-error" role="alert">
                    {investmentExposureError}
                  </div>
                ) : null}
                {isLoadingInvestmentExposure && !investmentExposure ? (
                  <div className="investment-exposure-empty">Loading exposure data...</div>
                ) : exposureChartData.length ? (
                  <>
                    <div className="investment-exposure-chart">
                      <ResponsiveContainer width="100%" height="100%">
                        {investmentExposureChartType === "pie" ? (
                          <PieChart>
                            <Tooltip content={<ExposureTooltip />} />
                            <Pie
                              data={exposureChartData}
                              dataKey="exposure_value"
                              nameKey="symbol"
                              innerRadius={62}
                              outerRadius={106}
                              paddingAngle={2}
                            >
                              {exposureChartData.map((entry, index) => (
                                <Cell key={`exposure-${entry.symbol}`} fill={INVESTMENT_COLORS[index % INVESTMENT_COLORS.length]} />
                              ))}
                            </Pie>
                          </PieChart>
                        ) : (
                          <BarChart data={exposureChartData} layout="vertical" margin={{ top: 8, right: 28, bottom: 8, left: 92 }}>
                            <XAxis type="number" hide domain={[0, "dataMax"]} />
                            <YAxis
                              type="category"
                              dataKey="chartLabel"
                              width={132}
                              tickLine={false}
                              axisLine={false}
                              tick={<ExposureAxisTick />}
                            />
                            <Tooltip content={<ExposureTooltip />} />
                            <Bar dataKey="percent_of_portfolio" fill="#1f5c4d" radius={[0, 10, 10, 0]} />
                          </BarChart>
                        )}
                      </ResponsiveContainer>
                    </div>
                    <div className="investment-exposure-list">
                      {exposureItems.slice(0, 6).map((item) => (
                        <div key={item.symbol} className="investment-exposure-row">
                          <div>
                            <strong>{item.symbol}</strong>
                            <span>{item.name || "Underlying stock"}</span>
                          </div>
                          <div>
                            <strong>{Number(item.percent_of_portfolio || 0).toFixed(2)}%</strong>
                            <span>{currency(item.exposure_value)}</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </>
                ) : (
                  <div className="investment-exposure-empty">
                    <strong>No single-name exposure yet</strong>
                    <span>Use Refresh after configuring `ALPHA_VANTAGE_API_KEY` on the backend.</span>
                  </div>
                )}
                {investmentExposure?.unresolved?.length ? (
                  <p className="panel-note investment-exposure-footnote">
                    Unresolved funds: {investmentExposure.unresolved.map((item) => item.symbol).join(", ")}
                  </p>
                ) : null}
              </article>

            </div>

            <article className="panel investment-industry-panel">
              <div className="panel-header investment-exposure-header">
                <div>
                  <p className="section-label">Industry Exposure</p>
                  <h3>Industries behind your holdings</h3>
                  <p className="panel-note">
                    {industryExposure?.status === "ready"
                      ? `${industrySummary.industry_count || 0} industries · classified ${formatSnapshotTimestamp(industrySummary.classified_at)}`
                      : "Refresh after single-name exposure is available to classify companies by industry."}
                  </p>
                </div>
                <div className="investment-exposure-actions">
                  <label>
                    <span>View</span>
                    <select value={industryExposureChartType} onChange={(event) => setIndustryExposureChartType(event.target.value)}>
                      <option value="bar">Bar</option>
                      <option value="pie">Pie</option>
                    </select>
                  </label>
                  <button className="ghost-button" type="button" onClick={handleIndustryExposureRefresh} disabled={isRefreshingIndustryExposure}>
                    <RefreshCw className={isRefreshingIndustryExposure ? "spin" : ""} size={15} aria-hidden="true" />
                    {isRefreshingIndustryExposure ? "Classifying" : "Refresh"}
                  </button>
                </div>
              </div>
              {industryExposureError ? (
                <div className="chat-error" role="alert">
                  {industryExposureError}
                </div>
              ) : null}
              {isLoadingIndustryExposure && !industryExposure ? (
                <div className="investment-exposure-empty">Loading industry exposure...</div>
              ) : industryChartData.length ? (
                <>
                  <div className="investment-industry-chart">
                    <ResponsiveContainer width="100%" height="100%">
                      {industryExposureChartType === "pie" ? (
                        <PieChart>
                          <Tooltip content={<IndustryTooltip />} />
                          <Pie
                            data={industryChartData}
                            dataKey="exposure_value"
                            nameKey="industry"
                            innerRadius={70}
                            outerRadius={120}
                            paddingAngle={2}
                          >
                            {industryChartData.map((entry, index) => (
                              <Cell key={`industry-${entry.industry}`} fill={INVESTMENT_COLORS[index % INVESTMENT_COLORS.length]} />
                            ))}
                          </Pie>
                        </PieChart>
                      ) : (
                        <BarChart data={industryChartData} layout="vertical" margin={{ top: 8, right: 28, bottom: 8, left: 112 }}>
                          <XAxis type="number" hide domain={[0, "dataMax"]} />
                          <YAxis type="category" dataKey="chartLabel" width={168} tickLine={false} axisLine={false} />
                          <Tooltip content={<IndustryTooltip />} />
                          <Bar dataKey="percent_of_portfolio" fill="#aa7d2d" radius={[0, 10, 10, 0]} />
                        </BarChart>
                      )}
                    </ResponsiveContainer>
                  </div>
                  <div className="investment-industry-list">
                    {industryItems.slice(0, 8).map((item, index) => (
                      <div key={`${item.sector}-${item.industry}`} className="investment-exposure-row investment-industry-row">
                        <span className="investment-allocation-swatch" style={{ background: INVESTMENT_COLORS[index % INVESTMENT_COLORS.length] }} />
                        <div>
                          <strong>{item.industry}</strong>
                          <span>{item.sector} · {item.company_count} compan{item.company_count === 1 ? "y" : "ies"}</span>
                        </div>
                        <div>
                          <strong>{Number(item.percent_of_portfolio || 0).toFixed(2)}%</strong>
                          <span>{currency(item.exposure_value)}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </>
              ) : (
                <div className="investment-exposure-empty">
                  <strong>No industry exposure yet</strong>
                  <span>Refresh this card after single-name exposure has been generated.</span>
                </div>
              )}
            </article>

            <article className="panel investment-holdings-panel">
              <div className="panel-header">
                <div>
                  <p className="section-label">Positions</p>
                  <h3>Current holdings</h3>
                </div>
                <span className="panel-note">{holdings.length} positions</span>
              </div>
              <div className="table-wrap">
                <table className="investment-holdings-table">
                  <thead>
                    <tr>
                      <th>Symbol</th>
                      <th>Description</th>
                      <th>Quantity</th>
                      <th>Last price</th>
                      <th>Value</th>
                      <th>Portfolio</th>
                      <th>Total gain / loss</th>
                    </tr>
                  </thead>
                  <tbody>
                    {holdings.map((holding) => (
                      <tr key={`${holding.account_id}-${holding.symbol}`}>
                        <td>
                          <strong>{holding.symbol}</strong>
                          {holding.is_cash_like ? <span className="investment-cash-badge">Cash</span> : null}
                        </td>
                        <td>{holding.description}</td>
                        <td>{holding.quantity ?? "—"}</td>
                        <td>{holding.last_price == null ? "—" : currency(holding.last_price)}</td>
                        <td><strong>{currency(holding.current_value)}</strong></td>
                        <td>{Number(holding.percent_of_portfolio || 0).toFixed(2)}%</td>
                        <td className={gainTone(holding.total_gain_loss_amount)}>
                          {holding.total_gain_loss_amount == null ? "—" : signedCurrency(holding.total_gain_loss_amount)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </article>
          </>
        ) : (
          <article className="panel investment-empty-state">
            <ChartPie size={32} aria-hidden="true" />
            <h3>No investment snapshot yet</h3>
            <p>Choose your exported Fidelity positions CSV above to populate this portfolio view.</p>
          </article>
        )}
      </section>
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
    const plannerCategories = Array.from(
      new Set([
        ...categoryStatuses.map((item) => item.category_name).filter(Boolean),
        ...(selectedCard?.categories || []).map((item) => item.category).filter(Boolean),
      ])
    )
      .sort((left, right) => left.localeCompare(right))
      .map((categoryName) => ({
        categoryName,
        bucket: plannerCategoryAssignments[categoryName] || getDefaultPlannerBucket(categoryName),
        liveStatus: categoryStatuses.find((item) => item.category_name === categoryName) || null,
      }));
    const bucketCounts = plannerCategories.reduce((counts, item) => {
      counts[item.bucket] = (counts[item.bucket] || 0) + 1;
      return counts;
    }, {});

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
            {isGoalsView ? (
              <>
                <CollapsiblePanel
                  sectionLabel="Planner Categories"
                  title="Teach the planner how your categories behave"
                  className="budgeting-buckets-guide-card"
                  defaultOpen={false}
                  collapsedLabel="View bucket guide"
                  expandedLabel="Hide bucket guide"
                  summary={(
                    <p className="panel-note">
                      Budget recommendations depend on category buckets. Open this guide when you want a quick refresher on what counts as income, savings, fixed commitments, and flexible spend.
                    </p>
                  )}
                >
                  <div className="budgeting-bucket-guide-grid">
                    {PLANNER_BUCKET_OPTIONS.map((bucket) => (
                      <div key={bucket.id} className="budgeting-bucket-guide-item">
                        <div className="budgeting-bucket-guide-head">
                          <strong>{bucket.label}</strong>
                          <span>{bucketCounts[bucket.id] || 0} assigned</span>
                        </div>
                        <p>{bucket.description}</p>
                      </div>
                    ))}
                  </div>
                </CollapsiblePanel>

                <CollapsiblePanel
                  sectionLabel="Category Assignment"
                  title={plannerCategories.length ? "Customize planner buckets when you need to" : "No categories available yet"}
                  className="budgeting-category-mapper-card"
                  defaultOpen={false}
                  collapsedLabel="Edit assignments"
                  expandedLabel="Hide assignment editor"
                  collapsible={plannerCategories.length > 0}
                  summary={
                    plannerCategories.length ? (
                      <p className="panel-note">
                        Most users will only need this when the planner misclassifies a category. Open the editor to review or override the current bucket suggestions.
                      </p>
                    ) : (
                      <p className="panel-note">
                        Load a saved budget or recent spending categories first, then the planner category setup will appear here.
                      </p>
                    )
                  }
                >
                  {plannerCategories.length ? (
                    <div className="planner-category-mapper-list">
                      {plannerCategories.map((item) => (
                        <div key={item.categoryName} className="planner-category-mapper-row">
                          <div className="planner-category-mapper-copy">
                            <div className="planner-category-mapper-head">
                              <strong>{item.categoryName}</strong>
                              {item.liveStatus ? (
                                <span className={"budget-status-pill " + getBudgetCategoryTone(item.liveStatus.status)}>
                                  {item.liveStatus.status.replace("_", " ")}
                                </span>
                              ) : (
                                <span className="planner-category-mapper-badge">No live budget status</span>
                              )}
                            </div>
                            <p>{PLANNER_BUCKET_OPTIONS.find((bucket) => bucket.id === item.bucket)?.description}</p>
                          </div>
                          <label className="planner-category-mapper-select">
                            <span className="sr-only">Planner bucket for {item.categoryName}</span>
                            <select
                              value={item.bucket}
                              onChange={(event) =>
                                setPlannerCategoryAssignments((current) => ({
                                  ...current,
                                  [item.categoryName]: event.target.value,
                                }))
                              }
                            >
                              {PLANNER_BUCKET_OPTIONS.map((bucket) => (
                                <option key={bucket.id} value={bucket.id}>
                                  {bucket.label}
                                </option>
                              ))}
                            </select>
                          </label>
                        </div>
                      ))}
                    </div>
                  ) : null}
                </CollapsiblePanel>
              </>
            ) : (
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
            )}
            <CollapsiblePanel
              sectionLabel="Active Plan Detail"
              title={hasActivePlan ? "Category targets and live status" : "Waiting for first saved plan"}
              className="budgeting-targets-card"
              defaultOpen={true}
              collapsedLabel="View targets"
              expandedLabel="Hide targets"
              collapsible={hasActivePlan}
              summary={
                hasActivePlan ? (
                  <p className="panel-note">
                    Track each saved category against live spend, then open the detail list when you want to inspect pacing, pressure, and remaining budget by category.
                  </p>
                ) : (
                  <p className="panel-note">
                    Once a budget is approved, its category targets and live spend status will appear here for quick reference.
                  </p>
                )
              }
            >
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
              ) : null}
            </CollapsiblePanel>

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
            <button
              className={`nav-item ${activeTab === "Investment" ? "active" : ""}`}
              type="button"
              onClick={() => setActiveTab("Investment")}
            >
              Investment
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
            <h2>{isInvestmentTab ? "Investment Portfolio" : "Financial Snapshot"}</h2>
          </div>
          {!isInvestmentTab ? <div className="hero-actions">
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
          </div> : null}
        </header>

        {isInvestmentTab ? (
          renderInvestmentTab()
        ) : loading ? (
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
            <>
              {activeTab === "Overview" ? renderOverviewTab() : null}
              {activeTab === "Card Details" ? renderCardsTab() : null}
              {activeTab === "Spending Analysis" ? renderSpendingAnalysisTab() : null}
              {isBudgetingTab ? renderBudgetingTab() : null}
            </>
          </div>
        ) : null}

        {selectedCard ? (
          <>
            <AssistantShell
              open={isAssistantShellOpen}
              mode={assistantShellMode}
              onModeChange={setAssistantShellMode}
              onClose={() => setIsAssistantShellOpen(false)}
              card={selectedCard}
              analysisWindow={selectedWindow}
              seedMessage={assistantSeed?.text || ""}
              seedMessageId={assistantSeed?.id || ""}
              onSeedConsumed={clearAssistantSeed}
            />
            <button
              className={`assistant-launcher ${isAssistantShellOpen ? "assistant-launcher--active" : ""}`}
              type="button"
              onClick={() => {
                if (isAssistantShellOpen) {
                  setIsAssistantShellOpen(false);
                  return;
                }
                handleAssistantOpen();
              }}
              aria-label={isAssistantShellOpen ? "Close assistant" : "Open assistant"}
              title={isAssistantShellOpen ? "Close assistant" : "Open assistant"}
            >
              <MessageCircle className="assistant-launcher-icon" size={16} aria-hidden="true" />
              <span className="assistant-launcher-label">Assistant</span>
            </button>
          </>
        ) : null}
      </main>
    </div>
  );
}

export default App;
