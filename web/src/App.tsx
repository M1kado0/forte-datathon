import { useEffect, useState } from "react";
import { Header, type Tab } from "@/components/Header";
import { CheckView } from "@/views/CheckView";
import { InsightsView } from "@/views/InsightsView";

function tabFromHash(): Tab {
  return window.location.hash === "#insights" ? "insights" : "check";
}

export default function App() {
  const [tab, setTab] = useState<Tab>(tabFromHash());

  useEffect(() => {
    const on = () => setTab(tabFromHash());
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);

  return (
    <div className="min-h-dvh flex flex-col">
      <Header tab={tab} />
      <main className="flex-1">{tab === "check" ? <CheckView /> : <InsightsView />}</main>
      <footer className="px-4 sm:px-8 py-6 text-xs text-muted-foreground text-center">
        DATATHON '26 · прототип команды · данные демонстрационные
      </footer>
    </div>
  );
}
