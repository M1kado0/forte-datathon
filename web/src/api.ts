// Mirrors contracts/schemas.py; UI uses the same handlers without a jury API key.
export type EvidenceFrame = { timestamp_sec: number; image_base64: string };
export type RuleType = { id: string; name: string; citation_suffix: string; rules: { title: string; approved_by: string; punkt: string | null; url: string; citation: string }[] };
export type Violation = { type: string; timecode_sec: number; description: string; evidence_frames: EvidenceFrame[] };
export type Job = {
  job_id: string;
  status: "processing" | "done" | "failed";
  verdict: "violation" | "no_violation" | null;
  violations: Violation[];
};

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`/api/ui${path}`, init);
  } catch {
    throw new Error("Сервис не ответил. Проверьте соединение и повторите проверку.");
  }
  if (!response.ok) {
    let message = `Ошибка ${response.status}. Повторите проверку.`;
    try {
      const body = await response.json();
      if (typeof body.detail === "string") message = body.detail;
    } catch { /* keep the HTTP error */ }
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

export const api = {
  rules: () => req<RuleType[]>("/rules"),
  analyze(file: File) {
    const body = new FormData();
    body.append("file", file);
    return req<{ job_id: string }>("/analyze", { method: "POST", body });
  },
  job: (id: string) => req<Job>(`/jobs/${encodeURIComponent(id)}`, { signal: AbortSignal.timeout(60_000) }),
};
