// Mirrors contracts/schemas.py. Change the Python contract first, then this file.

export type Reason = { feature: string; effect: number; text: string };
export type PredictResponse = {
  result: number;
  label: string;
  confidence: number;
  reasons: Reason[];
  action: string;
  impact_kzt: number;
};
export type ChartPoint = { x: string | number; y: number; highlight?: boolean };
export type Insight = { id: string; title: string; number: string; chart_data: ChartPoint[]; takeaway: string };
export type FieldMeta = {
  name: string;
  label: string;
  type: "number" | "category" | "text";
  min?: number | null;
  max?: number | null;
  step?: number | null;
  options?: string[] | null;
  default?: unknown;
  whatif: boolean;
};
export type Meta = { product_name: string; primary_field: string; fields: FieldMeta[] };

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(`/api${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!r.ok) {
    let msg = `Ошибка ${r.status}`;
    try {
      const j = await r.json();
      if (j?.detail) msg = typeof j.detail === "string" ? j.detail : msg;
    } catch { /* ignore */ }
    throw new Error(msg);
  }
  return r.json() as Promise<T>;
}

export const api = {
  meta: () => req<Meta>("/meta"),
  client: (id: string) => req<Record<string, number | string>>(`/clients/${encodeURIComponent(id)}`),
  predict: (input: Record<string, unknown>) =>
    req<PredictResponse>("/predict", { method: "POST", body: JSON.stringify({ input }) }),
  insights: () => req<Insight[]>("/insights"),
};
