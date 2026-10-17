import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, type Insight } from "@/api";
import { Card, ErrorBox, Label, Skeleton } from "@/components/ui";
import { cn } from "@/lib/utils";

export function InsightsView() {
  const [data, setData] = useState<Insight[] | null>(null);
  const [error, setError] = useState("");

  const load = () => {
    setError("");
    setData(null);
    api.insights().then(setData).catch((e) => setError((e as Error).message));
  };
  useEffect(load, []);

  return (
    <div className="mx-auto max-w-6xl px-4 py-8 sm:px-8 sm:py-10">
      <h1 className="text-2xl font-extrabold text-forte-dark sm:text-3xl">Что мы нашли в данных</h1>
      <div className="mt-6">
        {error && <ErrorBox message={error} onRetry={load} />}
        {!error && !data && (
          <div className="grid gap-6 lg:grid-cols-2">
            {[0, 1].map((i) => (
              <Card key={i} className="space-y-3">
                <Skeleton className="h-5 w-3/4" />
                <Skeleton className="h-10 w-24" />
                <Skeleton className="h-48 w-full" />
              </Card>
            ))}
          </div>
        )}
        {data && data.length === 0 && (
          <div className="rounded-[var(--radius-card)] border border-dashed bg-white/60 px-6 py-12 text-center text-muted-foreground">
            Инсайтов пока нет — запустите <code className="font-semibold">make all</code>.
          </div>
        )}
        {data && data.length > 0 && (
          <div className="grid gap-6 lg:grid-cols-2">
            {data.map((ins, i) => (
              <InsightCard key={ins.id} ins={ins} hero={i === 0} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function InsightCard({ ins, hero }: { ins: Insight; hero: boolean }) {
  return (
    <Card className={cn(hero && "lg:col-span-2")}>
      <h2 className={cn("font-bold text-forte-dark", hero ? "text-xl sm:text-2xl" : "text-lg")}>{ins.title}</h2>
      <div className="mt-3 flex items-baseline gap-3">
        <span className={cn("font-extrabold text-primary", hero ? "text-5xl" : "text-4xl")}>{ins.number}</span>
      </div>
      <div className={cn("mt-4", hero ? "h-64" : "h-48")}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={ins.chart_data} margin={{ top: 8, right: 8, left: -16, bottom: 0 }}>
            <CartesianGrid vertical={false} stroke="var(--border)" />
            <XAxis dataKey="x" tickLine={false} axisLine={false} tick={{ fontSize: 12, fill: "var(--text-muted)" }} />
            <YAxis tickLine={false} axisLine={false} tick={{ fontSize: 12, fill: "var(--text-muted)" }} unit="%" />
            <Tooltip cursor={{ fill: "var(--bg-muted)" }} formatter={(v: number) => [`${v}%`, "Доля дефолтов"]} />
            <Bar dataKey="y" radius={[6, 6, 0, 0]}>
              {ins.chart_data.map((d, i) => (
                <Cell key={i} fill={d.highlight ? "var(--forte-magenta)" : "var(--chart-grey)"} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <Label className="mt-4">Вывод</Label>
      <p className="mt-1 text-[15px]">{ins.takeaway}</p>
    </Card>
  );
}
