const DEFAULT_BACKEND_URL = "http://127.0.0.1:8000";

function getBackendBaseUrl() {
  return import.meta.env.VITE_BACKEND_URL?.trim() || DEFAULT_BACKEND_URL;
}

async function requestJson(path) {
  const response = await fetch(`${getBackendBaseUrl()}${path}`);
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Request failed with status ${response.status}`);
  }
  return response.json();
}

export function fetchAccounts() {
  return requestJson("/api/accounts");
}

export function fetchDashboardOverview(startDate, endDate) {
  const params = new URLSearchParams();
  if (startDate) params.set("start_date", startDate);
  if (endDate) params.set("end_date", endDate);
  const query = params.toString();
  return requestJson(`/api/dashboard${query ? `?${query}` : ""}`);
}

