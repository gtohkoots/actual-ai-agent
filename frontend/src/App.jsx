import { useEffect, useMemo, useState } from "react";
import { LayoutDashboard, Sparkles } from "lucide-react";

import ChatPanel from "./components/ChatPanel";
import { fetchAccounts, fetchDashboardOverview } from "./api/dashboard";

const navItems = ["Overview", "Cards", "Transactions", "Reports"];

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

function pickCardTint(index) {
  return CARD_TINTS[index % CARD_TINTS.length];
}

function buildCardViewModel(account, index, accountMeta = {}) {
  const summary = account.summary || {};
  const mergedSummary = {
    totalSpend: summary.totalSpend || currency(account.cycle_spend),
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
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;

    async function loadDashboard() {
      setLoading(true);
      setError("");
      try {
        const [accountsResponse, dashboardResponse] = await Promise.all([
          fetchAccounts(),
          fetchDashboardOverview(),
        ]);
        if (cancelled) return;
        setAccountsMeta(accountsResponse);
        setDashboard(dashboardResponse);
        const defaultPid = dashboardResponse?.accounts?.[0]?.account_pid || accountsResponse?.[0]?.account_pid || null;
        setSelectedCardId(defaultPid);
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Failed to load dashboard data");
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    void loadDashboard();
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

  useEffect(() => {
    if (!selectedCardId && cards.length > 0) {
      setSelectedCardId(cards[0].id);
    }
  }, [cards, selectedCardId]);

  const isAssistantView = activeTab === "AI Assistant";
  const stats = selectedCard
    ? [
        { label: "Current Balance", value: currency(selectedCard.balanceCurrent), note: selectedCard.deltaText },
        { label: "Selected Card Spend", value: selectedCard.summary.totalSpend, note: "Current cycle activity" },
        { label: "Top Category", value: selectedCard.summary.topCategory, note: selectedCard.utilizationText },
        { label: "Top Merchant", value: selectedCard.summary.topMerchant, note: "Best AI follow-up target" },
        { label: "Net Cash Flow", value: selectedCard.summary.netCashFlow, note: selectedCard.summary.aiSuggestion },
      ]
    : [];

  function handleCardSelect(cardId) {
    setSelectedCardId(cardId);
    setActiveTab("Overview");
  }

  function handleAssistantOpen() {
    setActiveTab("AI Assistant");
  }

  function handleDashboardOpen() {
    setActiveTab("Overview");
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
            <button className="ghost-button" type="button">
              {dashboard?.selected_window || "Loading window..."}
            </button>
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
          isAssistantView ? (
            <section className="assistant-stage">
              <ChatPanel card={selectedCard} seedMessage={assistantSeed?.text || ""} seedMessageId={assistantSeed?.id || ""} onSeedConsumed={clearAssistantSeed} />
            </section>
          ) : (
            <>
              <section className="stats-grid">
                {stats.map((stat) => (
                  <article key={stat.label} className="stat-card">
                    <p className="section-label">{stat.label}</p>
                    <h3>{stat.value}</h3>
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
                  seedMessage={assistantSeed?.text || ""}
                  seedMessageId={assistantSeed?.id || ""}
                  onSeedConsumed={clearAssistantSeed}
                />
              </section>
            </>
          )
        ) : null}

        <button
          className={`assistant-launcher ${isAssistantView ? "assistant-launcher--active" : ""}`}
          type="button"
          onClick={isAssistantView ? handleDashboardOpen : handleAssistantOpen}
          aria-label={isAssistantView ? "Return to dashboard" : "Open AI assistant"}
          title={isAssistantView ? "Return to dashboard" : "Open AI assistant"}
        >
          <span className="assistant-launcher-ring" aria-hidden="true">
            <span className="assistant-launcher-core">
              {isAssistantView ? <LayoutDashboard size={22} /> : <Sparkles size={22} />}
            </span>
          </span>
          <span className="assistant-launcher-label">
            {isAssistantView ? "Dashboard" : "AI"}
          </span>
        </button>
      </main>
    </div>
  );
}

export default App;
