export const dashboardData = {
  monthLabel: "March 2026",
  selectedWindow: "Last 30 days",
  cards: [
    {
      id: "amex-green",
      name: "Amex Green",
      network: "American Express",
      tint: "linear-gradient(135deg, #235446 0%, #122b24 100%)",
      last4: "2760",
      cycleSpend: 1842.73,
      deltaText: "-4.2% vs last month",
      utilizationText: "Dining and travel heavy",
      creditLimit: "$8,000",
      context: {
        card: "Amex Green",
        accountName: "Amex Green Card",
        accountPid: "9bbb85d7-a78a-4542-a644-5a78184ce110",
        dateRange: "March 2026",
        windowStart: "2026-03-01",
        windowEnd: "2026-03-31",
        focus: "Travel + Dining",
      },
      summary: {
        totalSpend: "$1,842.73",
        topCategory: "Travel",
        topMerchant: "United Airlines",
        aiSuggestion: "Ask why travel spend jumped this month",
      },
      categories: [
        { name: "Travel", amount: 730.42 },
        { name: "Dining", amount: 482.11 },
        { name: "Transit", amount: 290.2 },
        { name: "Subscriptions", amount: 182.0 },
        { name: "Other", amount: 158.0 }
      ],
      merchants: [
        { name: "United Airlines", amount: "$421.88", note: "Flight rebooking" },
        { name: "Eiko Cafe Two", amount: "$120.55", note: "Top dining merchant" },
        { name: "Uber", amount: "$94.77", note: "Transit cluster" },
        { name: "Clear", amount: "$78.00", note: "Recurring travel service" }
      ],
      transactions: [
        { date: "2026-03-20", merchant: "United Airlines", category: "Travel", amount: -421.88 },
        { date: "2026-03-19", merchant: "Eiko Cafe Two", category: "Dining", amount: -43.75 },
        { date: "2026-03-19", merchant: "Uber", category: "Transit", amount: -28.42 },
        { date: "2026-03-18", merchant: "Clear", category: "Subscriptions", amount: -78.0 },
        { date: "2026-03-17", merchant: "Delta Sky Club", category: "Travel", amount: -158.0 }
      ],
      quickPrompts: [
        "Summarize why Amex Green is elevated this month",
        "Which merchants are driving this card's travel spend?",
        "Are there any recurring charges I should review on this card?"
      ],
      assistantReply:
        "This card is being pulled upward mostly by travel activity rather than everyday drift. The biggest driver is United Airlines, with supporting spend from transit and airport-related services. If the user is trying to reduce monthly volatility, travel bookings and travel subscriptions are the first levers to inspect."
    },
    {
      id: "boa-travel",
      name: "BofA Travel Rewards",
      network: "Visa Signature",
      tint: "linear-gradient(135deg, #51301f 0%, #1f130e 100%)",
      last4: "4421",
      cycleSpend: 967.24,
      deltaText: "+11.8% vs last month",
      utilizationText: "Groceries and home-heavy",
      creditLimit: "$6,500",
      context: {
        card: "BofA Travel Rewards",
        accountName: "BOA Checking",
        accountPid: "5c857f11-7465-43e0-9711-7b8166568b48",
        dateRange: "March 2026",
        windowStart: "2026-03-01",
        windowEnd: "2026-03-31",
        focus: "Groceries + Bills",
      },
      summary: {
        totalSpend: "$967.24",
        topCategory: "Grocery",
        topMerchant: "Costco",
        aiSuggestion: "Ask whether grocery spend is trending up"
      },
      categories: [
        { name: "Grocery", amount: 410.31 },
        { name: "Bills", amount: 238.9 },
        { name: "Shopping", amount: 174.0 },
        { name: "Gas", amount: 88.03 },
        { name: "Other", amount: 56.0 }
      ],
      merchants: [
        { name: "Costco", amount: "$178.73", note: "Largest merchant this cycle" },
        { name: "Progressive", amount: "$113.50", note: "Insurance bill" },
        { name: "Trader Joe's", amount: "$98.44", note: "Recurring grocery pattern" },
        { name: "Chevron", amount: "$52.30", note: "Gas spending stable" }
      ],
      transactions: [
        { date: "2026-03-21", merchant: "Costco", category: "Grocery", amount: -178.73 },
        { date: "2026-03-20", merchant: "Progressive", category: "Bills", amount: -113.5 },
        { date: "2026-03-19", merchant: "Trader Joe's", category: "Grocery", amount: -98.44 },
        { date: "2026-03-18", merchant: "Amazon", category: "Shopping", amount: -74.0 },
        { date: "2026-03-16", merchant: "Chevron", category: "Gas", amount: -52.3 }
      ],
      quickPrompts: [
        "Compare grocery spend on this card to previous weeks",
        "Which bills on this card look recurring?",
        "Explain the Costco concentration this cycle"
      ],
      assistantReply:
        "This card looks like the best candidate for a practical budgeting view because its spend is concentrated in repeatable categories: groceries, bills, and fuel. That makes it a strong place to surface recurring merchants, compare historical weekly snapshots, and suggest savings opportunities without much noise."
    },
    {
      id: "amex-gold",
      name: "Amex Gold",
      network: "Visa Infinite",
      tint: "linear-gradient(135deg, #1b2c55 0%, #0d1425 100%)",
      last4: "9008",
      cycleSpend: 1326.88,
      deltaText: "+2.1% vs last month",
      utilizationText: "Mixed lifestyle spending",
      creditLimit: "$12,000",
      context: {
        card: "Amex Gold",
        accountName: "Amex Gold Card",
        accountPid: "d37c0ee5-4f84-4204-a929-1a74032975fa",
        dateRange: "March 2026",
        windowStart: "2026-03-01",
        windowEnd: "2026-03-31",
        focus: "Entertainment + Shopping",
      },
      summary: {
        totalSpend: "$1,326.88",
        topCategory: "Entertainment",
        topMerchant: "Ticketmaster",
        aiSuggestion: "Ask if this card should be the default for subscriptions"
      },
      categories: [
        { name: "Entertainment", amount: 501.2 },
        { name: "Shopping", amount: 366.54 },
        { name: "Dining", amount: 218.14 },
        { name: "Subscriptions", amount: 143.0 },
        { name: "Other", amount: 98.0 }
      ],
      merchants: [
        { name: "Ticketmaster", amount: "$301.20", note: "Event purchases" },
        { name: "Nordstrom", amount: "$173.44", note: "Shopping spike" },
        { name: "Netflix", amount: "$19.99", note: "Subscription candidate" },
        { name: "Spotify", amount: "$10.99", note: "Subscription candidate" }
      ],
      transactions: [
        { date: "2026-03-21", merchant: "Ticketmaster", category: "Entertainment", amount: -301.2 },
        { date: "2026-03-20", merchant: "Nordstrom", category: "Shopping", amount: -173.44 },
        { date: "2026-03-19", merchant: "Netflix", category: "Subscriptions", amount: -19.99 },
        { date: "2026-03-18", merchant: "Spotify", category: "Subscriptions", amount: -10.99 },
        { date: "2026-03-17", merchant: "Local Bistro", category: "Dining", amount: -64.88 }
      ],
      quickPrompts: [
        "What stands out as unusual on Chase Sapphire?",
        "Group this card's subscriptions for me",
        "Should entertainment spend on this card concern me?"
      ],
      assistantReply:
        "This card reads more like a discretionary-spend lens than a baseline household-spend lens. The frontend should treat it differently by emphasizing spikes, lifestyle trends, and subscription grouping rather than recurring essentials."
    }
  ]
};
