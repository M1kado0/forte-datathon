// Mirrors contracts/schemas.py; UI uses the same handlers without a jury API key.
export type EvidenceFrame = { timestamp_sec: number; image_base64: string };
export type Violation = { type: string; timecode_sec: number; description: string; evidence_frames: EvidenceFrame[] };
export type Job = {
  job_id: string;
  status: "processing" | "done" | "failed";
  verdict: "violation" | "no_violation" | null;
  violations: Violation[];
};

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api/ui${path}`, init);
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
  analyze(file: File) {
    const body = new FormData();
    body.append("file", file);
    return req<{ job_id: string }>("/analyze", { method: "POST", body });
  },
  job: (id: string) => req<Job>(`/jobs/${encodeURIComponent(id)}`),
};
