import { getBackendBaseUrl, normalizeMessages } from "../api/backend";

export async function sendChatMessage({ message, conversationId, history, context }) {
  const response = await fetch(`${getBackendBaseUrl()}/api/analysis/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      message,
      conversation_id: conversationId || null,
      history: normalizeMessages(history),
      context,
    }),
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Request failed with status ${response.status}`);
  }

  return response.json();
}

export async function fetchChatConversation(conversationId) {
  if (!conversationId) {
    return null;
  }

  const response = await fetch(`${getBackendBaseUrl()}/api/analysis/chat/conversations/${conversationId}`);
  if (response.status === 404) {
    return null;
  }
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Request failed with status ${response.status}`);
  }

  return response.json();
}

export async function fetchChatConversations(accountPid, limit = 8) {
  const params = new URLSearchParams();
  if (accountPid) params.set("account_pid", accountPid);
  if (limit) params.set("limit", String(limit));
  const query = params.toString();
  const response = await fetch(`${getBackendBaseUrl()}/api/analysis/chat/conversations${query ? `?${query}` : ""}`);
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Request failed with status ${response.status}`);
  }
  return response.json();
}

export async function deleteChatConversation(conversationId) {
  const response = await fetch(`${getBackendBaseUrl()}/api/analysis/chat/conversations/${conversationId}`, {
    method: "DELETE",
  });

  if (!response.ok && response.status !== 204) {
    const detail = await response.text();
    throw new Error(detail || `Request failed with status ${response.status}`);
  }
}


export async function fetchAnalysisCategories({ accountPid, accountName, startDate, endDate } = {}) {
  const params = new URLSearchParams();
  if (accountPid) params.set("account_pid", accountPid);
  if (accountName) params.set("account_name", accountName);
  if (startDate) params.set("start_date", startDate);
  if (endDate) params.set("end_date", endDate);
  const query = params.toString();
  const response = await fetch(`${getBackendBaseUrl()}/api/analysis/options/categories${query ? `?${query}` : ""}`);
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Request failed with status ${response.status}`);
  }
  const payload = await response.json();
  return Array.isArray(payload.items) ? payload.items : [];
}

export async function fetchAnalysisPayees({ accountPid, accountName, startDate, endDate, limit = 50 } = {}) {
  const params = new URLSearchParams();
  if (accountPid) params.set("account_pid", accountPid);
  if (accountName) params.set("account_name", accountName);
  if (startDate) params.set("start_date", startDate);
  if (endDate) params.set("end_date", endDate);
  if (limit) params.set("limit", String(limit));
  const query = params.toString();
  const response = await fetch(`${getBackendBaseUrl()}/api/analysis/options/payees${query ? `?${query}` : ""}`);
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Request failed with status ${response.status}`);
  }
  const payload = await response.json();
  return Array.isArray(payload.items) ? payload.items : [];
}


export async function fetchAnalysisAccounts() {
  const response = await fetch(`${getBackendBaseUrl()}/api/accounts`);
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Request failed with status ${response.status}`);
  }
  const payload = await response.json();
  return Array.isArray(payload) ? payload : [];
}
