import { getBackendBaseUrl } from "../api/backend";

async function responseJson(response) {
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Request failed with status ${response.status}`);
  }
  return response.json();
}

export async function fetchInvestmentsOverview() {
  return responseJson(await fetch(`${getBackendBaseUrl()}/api/investments/overview`));
}

export async function fetchSingleNameExposure({ minPercent = 1, limit = 15 } = {}) {
  const params = new URLSearchParams();
  params.set("min_percent", String(minPercent));
  params.set("limit", String(limit));
  return responseJson(await fetch(`${getBackendBaseUrl()}/api/investments/exposure/single-name?${params.toString()}`));
}

export async function refreshInvestmentExposure({ force = false } = {}) {
  const params = new URLSearchParams();
  params.set("force", force ? "true" : "false");
  return responseJson(
    await fetch(`${getBackendBaseUrl()}/api/investments/exposure/refresh?${params.toString()}`, {
      method: "POST",
    })
  );
}

export async function fetchIndustryExposure() {
  return responseJson(await fetch(`${getBackendBaseUrl()}/api/investments/industry/exposure`));
}

export async function refreshIndustryExposure({ force = false } = {}) {
  const params = new URLSearchParams();
  params.set("force", force ? "true" : "false");
  return responseJson(
    await fetch(`${getBackendBaseUrl()}/api/investments/industry/refresh?${params.toString()}`, {
      method: "POST",
    })
  );
}

export async function importFidelityPositionsCsv(file, asOfDate) {
  const params = new URLSearchParams();
  if (file?.name) params.set("filename", file.name);
  if (asOfDate) params.set("as_of_date", asOfDate);
  const query = params.toString();

  return responseJson(
    await fetch(`${getBackendBaseUrl()}/api/investments/import/fidelity-positions-csv${query ? `?${query}` : ""}`, {
      method: "POST",
      headers: {
        "Content-Type": "text/csv",
      },
      body: file,
    })
  );
}
