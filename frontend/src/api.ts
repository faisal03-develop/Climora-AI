const API = import.meta.env.DEV ? "" : "";

export type Horizon = "1d" | "7d" | "month" | "year";

export interface Status {
  data_loaded: boolean;
  data_error: string | null;
  date_range: string[] | null;
  row_count: number;
  last_load_mw: number | null;
  target_column: string;
  models: Record<Horizon, boolean>;
  horizons: { id: Horizon; label: string; available: boolean; steps: number }[];
}

export interface History {
  timestamps: string[];
  actual: number[];
  day_ahead: number[];
}

export interface Forecast {
  horizon: Horizon;
  horizon_label: string;
  last_timestamp: string;
  forecast_steps: number;
  forecast_mw: number[];
  forecast_timestamps: string[];
  forecast_peak_mw: number;
  forecast_mean_mw: number;
}

export interface Metrics {
  available: boolean;
  MAE?: number;
  RMSE?: number;
  R2?: number;
  horizon?: string;
}

export interface Eda {
  shape: number[];
  date_range: string[];
  mean_load: number;
  max_load: number;
  min_load: number;
  correlation_forecast: number;
  missing_count: number;
}

async function fetchJson<T>(url: string): Promise<T> {
  const res = await fetch(`${API}${url}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || res.statusText);
  }
  return res.json();
}

export const api = {
  status: () => fetchJson<Status>("/api/status"),
  history: (horizon: Horizon, points = 672) =>
    fetchJson<History>(`/api/history?horizon=${horizon}&points=${points}`),
  forecast: (horizon: Horizon) => fetchJson<Forecast>(`/api/forecast?horizon=${horizon}`),
  metrics: (horizon: Horizon) => fetchJson<Metrics>(`/api/metrics?horizon=${horizon}`),
  eda: () => fetchJson<Eda>("/api/eda"),
  upload: async (file: File) => {
    const form = new FormData();
    form.append("file", file);
    const res = await fetch(`${API}/api/upload`, { method: "POST", body: form });
    if (!res.ok) throw new Error((await res.json()).detail || "Upload failed");
    return res.json();
  },
};
