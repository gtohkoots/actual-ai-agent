const DEFAULT_BACKEND_URL = "http://127.0.0.1:8000";

function getBackendBaseUrl() {
  return import.meta.env.VITE_BACKEND_URL?.trim() || DEFAULT_BACKEND_URL;
}

function normalizeMessages(messages) {
  return messages
    .filter((message) => message && message.role && message.content)
    .map((message) => ({
      role: message.role,
      content: message.content,
    }));
}

export async function sendChatMessage({ message, conversationId, history, context }) {
  const response = await fetch(`${getBackendBaseUrl()}/api/chat`, {
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

