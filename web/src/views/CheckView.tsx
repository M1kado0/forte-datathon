import { useEffect, useRef, useState, type FormEvent } from "react";
import { api, type FieldMeta, type PredictResponse } from "@/api";
import { Button, Card, ErrorBox, Input, Label, Skeleton } from "@/components/ui";
import { cn, formatKzt, formatPct } from "@/lib/utils";

const EXAMPLES = ["KZ-100077", "KZ-100014", "KZ-100005"]; // pick real demo IDs tomorrow

type Status = "idle" | "loading" | "ok" | "error";

export function CheckView() {
  const [clientId, setClientId] = useState("");
  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState("");
  const [base, setBase] = useState<Record<string, number | string> | null>(null);
  const [res, setRes] = useState<PredictResponse | null>(null);
  const [first, setFirst] = useState<PredictResponse | null>(null);
  const [whatif, setWhatif] = useState<FieldMeta | null>(null);
  const [whatifValue, setWhatifValue] = useState<number>(0);

  useEffect(() => {
    api.meta().then((m) => setWhatif(m.fields.find((f) => f.whatif) ?? null)).catch(() => {});
  }, []);

  async function check(id: string) {
    const cid = id.trim();
    if (!cid) return;
    setClientId(cid);
    setStatus("loading");
    setError("");
    try {
      const [row, pred] = await Promise.all([api.client(cid), api.predict({ client_id: cid })]);
      setBase(row);
      setRes(pred);
      setFirst(pred);
      if (whatif) setWhatifValue(Number(row[whatif.name] ?? whatif.default ?? 0));
      setStatus("ok");
    } catch (e) {
      setError((e as Error).message);
      setStatus("error");
    }
  }

  // live what-if (debounced)
  const timer = useRef<number | undefined>(undefined);
  function onWhatif(v: number) {
    setWhatifValue(v);
    if (!whatif || !base) return;
    window.clearTimeout(timer.current);
    timer.current = window.setTimeout(async () => {
      try {
        setRes(await api.predict({ client_id: clientId, [whatif.name]: v }));
      } catch { /* keep last result */ }
    }, 120);
  }

  const submit = (e: FormEvent) => {
    e.preventDefault();
    check(clientId);
  };

  return (
    <>
      {/* Hero: the only gradient on the screen */}
      <section className="bg-hero">
        <div className="mx-auto max-w-6xl px-4 pb-24 pt-12 sm:px-8 sm:pt-16">
          <h1 className="max-w-xl text-3xl font-extrabold leading-tight text-white sm:text-[44px]">
            Проверить клиента за&nbsp;5&nbsp;секунд
          </h1>
          <p className="mt-3 max-w-lg text-base text-white sm:text-lg">
            Риск, причины и рекомендуемое действие — на одном экране.
          </p>
        </div>
      </section>

      <div className="mx-auto -mt-14 max-w-6xl px-4 sm:px-8">
        <Card className="p-4 sm:p-6">
          <form onSubmit={submit} className="flex flex-col gap-3 sm:flex-row">
            <label htmlFor="cid" className="sr-only">ID клиента</label>
            <Input
              id="cid"
              autoFocus
              value={clientId}
              onChange={(e) => setClientId(e.target.value)}
              placeholder="ID клиента, например KZ-100077"
              autoComplete="off"
            />
            <Button type="submit" disabled={!clientId.trim() || status === "loading"} className="sm:w-56 shrink-0">
              {status === "loading" ? "Проверяем…" : "Проверить клиента"}
            </Button>
          </form>
          <div className="mt-3 flex flex-wrap items-center gap-x-3 gap-y-1 text-sm text-muted-foreground">
            <span>Попробуйте:</span>
            {EXAMPLES.map((id) => (
              <button key={id} onClick={() => check(id)} className="font-semibold text-forte-dark hover:text-primary">
                {id}
              </button>
            ))}
          </div>
        </Card>

        <div className="mt-6 pb-10">
          {status === "idle" && <EmptyState />}
          {status === "loading" && <ResultSkeleton />}
          {status === "error" && <ErrorBox message={error} onRetry={() => check(clientId)} />}
          {status === "ok" && res && first && (
            <ResultView
              res={res}
              first={first}
              whatif={whatif}
              whatifValue={whatifValue}
              onWhatif={onWhatif}
            />
          )}
        </div>
      </div>
    </>
  );
}

function EmptyState() {
  return (
    <div className="rounded-[var(--radius-card)] border border-dashed bg-white/60 px-6 py-12 text-center">
      <p className="font-semibold text-forte-dark">Здесь появится результат проверки</p>
      <p className="mt-1 text-sm text-muted-foreground">Введите ID клиента или выберите пример выше.</p>
    </div>
  );
}

function ResultSkeleton() {
  return (
    <div className="grid gap-6 lg:grid-cols-5">
      <Card className="lg:col-span-2 space-y-4">
        <Skeleton className="h-4 w-24" />
        <Skeleton className="h-14 w-40" />
        <Skeleton className="h-4 w-full" />
        <Skeleton className="h-20 w-full" />
      </Card>
      <Card className="lg:col-span-3 space-y-4">
        <Skeleton className="h-4 w-48" />
        {[...Array(5)].map((_, i) => <Skeleton key={i} className="h-8 w-full" />)}
      </Card>
    </div>
  );
}

const BAND: Record<string, string> = {
  "Высокий риск": "bg-danger",
  "Средний риск": "bg-warning",
  "Низкий риск": "bg-success",
};

function ResultView({
  res, first, whatif, whatifValue, onWhatif,
}: {
  res: PredictResponse;
  first: PredictResponse;
  whatif: FieldMeta | null;
  whatifValue: number;
  onWhatif: (v: number) => void;
}) {
  const maxEffect = Math.max(...res.reasons.map((r) => Math.abs(r.effect)), 1e-9);
  const flipped = res.label !== first.label;

  return (
    <div className="grid gap-6 lg:grid-cols-5">
      {/* Verdict */}
      <Card className="lg:col-span-2 flex flex-col">
        <Label>Вероятность дефолта</Label>
        <div className="mt-2 flex items-baseline gap-3">
          <span className="text-6xl font-extrabold tracking-tight text-forte-dark">{formatPct(res.result)}</span>
        </div>
        <div className="mt-3 inline-flex items-center gap-2 text-base font-bold text-forte-dark">
          <span className={cn("h-2.5 w-2.5 rounded-full", BAND[res.label] ?? "bg-muted-foreground")} />
          {res.label}
        </div>

        <div className="mt-6 rounded-[var(--radius-btn)] bg-accent p-4">
          <Label className="text-forte-dark/80">Рекомендация</Label>
          <p className="mt-1 text-lg font-bold text-forte-dark">{res.action}</p>
          <p className="mt-2 text-sm text-forte-dark">
            Ожидаемый эффект: <span className="font-bold">{formatKzt(res.impact_kzt)}</span>
          </p>
        </div>
      </Card>

      {/* Reasons */}
      <Card className="lg:col-span-3">
        <h2 className="text-lg font-bold text-forte-dark">Почему такой результат</h2>
        <ul className="mt-4 space-y-3">
          {res.reasons.map((r) => (
            <li key={r.feature} className="grid grid-cols-1 items-center gap-1.5 sm:grid-cols-[1fr_minmax(80px,40%)] sm:gap-4">
              <span className="text-[15px]">{r.text}</span>
              <span className="h-2.5 rounded-full bg-muted">
                <span
                  className={cn("block h-full rounded-full", r.effect > 0 ? "bg-primary" : "bg-[var(--chart-grey)]")}
                  style={{ width: `${(Math.abs(r.effect) / maxEffect) * 100}%` }}
                />
              </span>
            </li>
          ))}
        </ul>
        <p className="mt-4 text-xs text-muted-foreground">
          <span className="inline-block h-2 w-3 rounded-full bg-primary align-middle" /> повышает риск ·{" "}
          <span className="inline-block h-2 w-3 rounded-full bg-[var(--chart-grey)] align-middle" /> снижает риск
        </p>
      </Card>

      {/* What-if: the live flip (memorable moment candidate) */}
      {whatif && (
        <Card className="lg:col-span-5">
          <div className="flex flex-wrap items-baseline justify-between gap-2">
            <h2 className="text-lg font-bold text-forte-dark">Что если?</h2>
            <p className="text-sm text-muted-foreground">
              Было {formatPct(first.result)} → сейчас{" "}
              <span className={cn("font-bold", flipped ? "text-primary" : "text-forte-dark")}>
                {formatPct(res.result)} · {res.label}
              </span>
            </p>
          </div>
          <label className="mt-4 block">
            <span className="flex justify-between text-sm font-semibold">
              <span>{whatif.label}</span>
              <span className="text-forte-dark">{whatifValue}</span>
            </span>
            <input
              type="range"
              className="mt-3 w-full"
              min={whatif.min ?? 0}
              max={whatif.max ?? 100}
              step={whatif.step ?? 1}
              value={whatifValue}
              onChange={(e) => onWhatif(Number(e.target.value))}
            />
          </label>
        </Card>
      )}
    </div>
  );
}
