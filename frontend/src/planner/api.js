import { getBackendBaseUrl, normalizeMessages } from "../api/backend";

export async function fetchPlannerOverview() {
  const response = await fetch(`${getBackendBaseUrl()}/api/planner/overview`);
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Request failed with status ${response.status}`);
  }
  return response.json();
}

export async function sendPlannerMessage({ message, conversationId, history, context }) {
  const response = await fetch(`${getBackendBaseUrl()}/api/planner/chat`, {
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

export async function fetchPlannerConversation(conversationId) {
  if (!conversationId) {
    return null;
  }

  const response = await fetch(`${getBackendBaseUrl()}/api/chat/conversations/${conversationId}`);
  if (response.status === 404) {
    return null;
  }
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Request failed with status ${response.status}`);
  }

  return response.json();
}

export async function fetchPlannerConversations(accountPid, limit = 8) {
  const params = new URLSearchParams();
  if (accountPid) params.set("account_pid", accountPid);
  if (limit) params.set("limit", String(limit));
  const query = params.toString();
  const response = await fetch(`${getBackendBaseUrl()}/api/chat/conversations${query ? `?${query}` : ""}`);
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Request failed with status ${response.status}`);
  }
  return response.json();
}

export async function deletePlannerConversation(conversationId) {
  const response = await fetch(`${getBackendBaseUrl()}/api/chat/conversations/${conversationId}`, {
    method: "DELETE",
  });

  if (!response.ok && response.status !== 204) {
    const detail = await response.text();
    throw new Error(detail || `Request failed with status ${response.status}`);
  }
}
