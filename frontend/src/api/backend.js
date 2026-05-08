const DEFAULT_BACKEND_URL = "http://127.0.0.1:8000";

export function getBackendBaseUrl() {
  return import.meta.env.VITE_BACKEND_URL?.trim() || DEFAULT_BACKEND_URL;
}

export function normalizeMessages(messages) {
  return (messages || [])
    .filter((message) => message && message.role && message.content)
    .map((message) => ({
      role: message.role,
      content: message.content,
    }));
}
