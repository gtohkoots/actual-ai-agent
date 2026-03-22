import { useState } from "react";

import ChatPanel from "./components/ChatPanel";
import { dashboardData } from "./mockData";

const navItems = ["Overview", "Cards", "Transactions", "Reports", "AI Assistant"];

function currency(value) {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 2
  }).format(value);
}

function App() {
  const [selectedCardId, setSelectedCardId] = useState(dashboardData.cards[0].id);
  const [activeTab, setActiveTab] = useState("Overview");
  const selectedCard = dashboardData.cards.find((card) => card.id === selectedCardId);
  const isAssistantView = activeTab === "AI Assistant";

  const stats = [
    { label: "Selected Card Spend", value: selectedCard.summary.totalSpend, note: selectedCard.deltaText },
    { label: "Top Category", value: selectedCard.summary.topCategory, note: selectedCard.utilizationText },
    { label: "Top Merchant", value: selectedCard.summary.topMerchant, note: "Best AI follow-up target" },
    { label: "Credit Limit", value: selectedCard.creditLimit, note: selectedCard.summary.aiSuggestion }
  ];

  function handleCardSelect(cardId) {
    setSelectedCardId(cardId);
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
          <div className="context-chips">
            <span className="chip">Card: {selectedCard.context.card}</span>
            <span className="chip">Window: {selectedCard.context.dateRange}</span>
            <span className="chip">Focus: {selectedCard.context.focus}</span>
          </div>
        </section>
      </aside>

      <main className="main-panel">
        <header className="hero">
          <div>
            <p className="eyebrow">{dashboardData.monthLabel}</p>
            <h2>Card spending cockpit</h2>
            <p className="hero-copy">
              Switch cards, inspect category drift, and carry that context directly into the AI conversation.
            </p>
          </div>
          <div className="hero-actions">
            <button className="ghost-button" type="button">
              {dashboardData.selectedWindow}
            </button>
            <button className="primary-button" type="button">
              Generate weekly insight
            </button>
          </div>
        </header>

        {isAssistantView ? (
          <section className="assistant-stage">
            <div className="assistant-intro panel">
              <p className="section-label">AI Assistant</p>
              <h3>Chat with your finance data in context</h3>
              <p className="section-copy">
                The selected card still anchors the conversation, but this view puts the chatbot first so the
                assistant feels like the primary workflow.
              </p>
              <div className="assistant-mini-stats">
                {stats.map((stat) => (
                  <article key={stat.label} className="assistant-stat">
                    <p className="section-label">{stat.label}</p>
                    <strong>{stat.value}</strong>
                    <span>{stat.note}</span>
                  </article>
                ))}
              </div>
            </div>

            <ChatPanel card={selectedCard} />
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
                {dashboardData.cards.map((card) => (
                  <button
                    key={card.id}
                    className={`finance-card ${card.id === selectedCardId ? "active" : ""}`}
                    style={{ background: card.tint }}
                    type="button"
                    onClick={() => handleCardSelect(card.id)}
                  >
                    <div className="card-brand">
                      <span>{card.network}</span>
                      <span>•• {card.last4}</span>
                    </div>
                    <h4 className="card-name">{card.name}</h4>
                    <p className="card-metric">{currency(card.cycleSpend)}</p>
                    <div className="card-foot">
                      <span>{card.deltaText}</span>
                      <span>{card.utilizationText}</span>
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
                      <div key={item.name} className="category-row">
                        <div className="category-meta">
                          <strong>{item.name}</strong>
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
                    <div key={merchant.name} className="merchant-row">
                      <div>
                        <strong>{merchant.name}</strong>
                        <div className="panel-note">{merchant.note}</div>
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
                        <tr key={`${tx.date}-${tx.merchant}-${tx.amount}`}>
                          <td>{tx.date}</td>
                          <td>{tx.merchant}</td>
                          <td>{tx.category}</td>
                          <td className="amount-negative">{currency(tx.amount)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </article>

              <ChatPanel card={selectedCard} />
            </section>
          </>
        )}
      </main>
    </div>
  );
}

export default App;
