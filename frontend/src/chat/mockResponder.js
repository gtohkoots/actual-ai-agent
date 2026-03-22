function currency(value) {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 2,
  }).format(value);
}

function buildSources(card) {
  return [
    {
      label: `${card.name} snapshot`,
      detail: `${card.context.dateRange} card summary`,
    },
    {
      label: "Weekly snapshot",
      detail: `${card.context.focus} and merchant patterns`,
    },
    {
      label: "Historical reports",
      detail: "Past advice and recurring themes",
    },
  ];
}

function buildActions(card) {
  return [
    `Compare ${card.name} with last month`,
    `Show more detail for ${card.summary.topMerchant}`,
    `Search similar weeks for ${card.summary.topCategory}`,
  ];
}

export function createWelcomeMessage(card) {
  return {
    id: `welcome-${card.id}`,
    role: "assistant",
    status: "ready",
    content: `You're looking at **${card.name}**. I already know the active date window is **${card.context.dateRange}** and the current focus is **${card.context.focus}**.\n\nAsk me about merchants, categories, unusual activity, or what to do next.`,
    sources: buildSources(card),
    actions: buildActions(card),
  };
}

function buildReply(card, text) {
  const lower = text.toLowerCase();
  const topMerchant = card.summary.topMerchant;
  const topCategory = card.summary.topCategory;

  if (lower.includes("compare") || lower.includes("last month")) {
    return {
      content: `### Comparison for ${card.name}\n\n- Current spend: **${card.summary.totalSpend}**\n- Top category: **${topCategory}**\n- Main merchant pressure: **${topMerchant}**\n\nThe card looks slightly elevated versus a normal baseline, but the pattern is still explainable by the same few categories. We should compare it against the prior month once the backend is wired up.`,
      sources: buildSources(card),
      actions: [`Open ${card.name} details`, `Compare against previous month`, `Find similar weeks`],
    };
  }

  if (lower.includes("merchant") || lower.includes(topMerchant.toLowerCase())) {
    return {
      content: `### Merchant focus: ${topMerchant}\n\nThis merchant is the biggest single driver in the selected card view.\n\n- Category: **${topCategory}**\n- Current card spend: **${card.summary.totalSpend}**\n- Pattern: recurring or clustered spend is likely worth checking first`,
      sources: [
        { label: `${topMerchant} transactions`, detail: "Selected card's recent activity" },
        { label: "Historical search", detail: "Similar merchant mentions in reports" },
      ],
      actions: [`Inspect ${topMerchant}`, `Show recent transactions`, `Search reports for ${topMerchant}`],
    };
  }

  if (lower.includes("recurring") || lower.includes("subscription")) {
    return {
      content: `### Recurring charges\n\nI would group the obvious repeaters first and then flag anything that is not aligned with the selected card's core use.\n\n- Subscriptions already visible in the card summary\n- Repeated merchant clusters\n- Anything that appears in multiple weekly snapshots\n\nIf you want, I can later connect this to the historical document store and surface recurring patterns automatically.`,
      sources: buildSources(card),
      actions: ["Group subscriptions", "Show recurring merchants", "Look for unusual renewals"],
    };
  }

  const amountHint = card.merchants
    .map((merchant) => `- **${merchant.name}**: ${merchant.amount}`)
    .join("\n");

  return {
    content: `### Quick read on ${card.name}\n\nThis card is best described as **${card.utilizationText.toLowerCase()}**.\n\n**What stands out**\n- Top category: **${topCategory}**\n- Main merchant: **${topMerchant}**\n- Total spend: **${card.summary.totalSpend}**\n\n**Merchant list**\n${amountHint}\n\nThe strongest next question is usually whether this is normal for the card or a temporary spike.`,
    sources: buildSources(card),
    actions: buildActions(card),
  };
}

export function sendMockChatMessage({ card, message }) {
  const reply = buildReply(card, message);

  return new Promise((resolve) => {
    setTimeout(() => {
      resolve({
        content: reply.content,
        sources: reply.sources,
        actions: reply.actions,
        createdAt: new Date().toISOString(),
      });
    }, 700);
  });
}

export function summarizeContext(card) {
  return [
    `Card: ${card.context.card}`,
    `Window: ${card.context.dateRange}`,
    `Focus: ${card.context.focus}`,
    `Spend: ${currency(card.cycleSpend)}`,
  ];
}
