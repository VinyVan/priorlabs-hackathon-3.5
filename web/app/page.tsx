"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useLang } from "../components/Lang";
import Gauge from "../components/Gauge";

type PredictResp = {
  label: number;
  proba: number;
  backend: string;
  recommendation: string;
};

export default function Home() {
  const { lang, t } = useLang();
  const [resp, setResp] = useState<PredictResp | null>(null);
  const [state, setState] = useState<"loading" | "ok" | "error">("loading");

  useEffect(() => {
    async function load() {
      try {
        const s = await fetch("/api/py/sample").then((r) => {
          if (!r.ok) throw new Error("sample");
          return r.json();
        });
        const p = await fetch("/api/py/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ row: s.row, model: "tabpfn", lang }),
        }).then((r) => {
          if (!r.ok) throw new Error("predict");
          return r.json();
        });
        setResp(p);
        setState("ok");
      } catch {
        setState("error");
      }
    }
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [lang]);

  const verdict =
    resp == null ? "" : resp.label === 1 ? t.hero.cultivated : t.hero.notCultivated;

  return (
    <div className="flex flex-col gap-6 pt-8">
      <span className="text-xs font-semibold tracking-wide uppercase text-green-800 dark:text-green-300">
        {t.hero.badge}
      </span>
      <h1 className="text-4xl font-extrabold">{t.hero.title}</h1>
      <p className="text-lg opacity-80">{t.hero.subtitle}</p>
      <div className="flex gap-3">
        <Link href="/predict" className="px-4 py-2 rounded-xl bg-green-700 text-white font-semibold">
          {t.hero.ctaPredict}
        </Link>
        <Link href="/compare" className="px-4 py-2 rounded-xl border font-semibold">
          {t.hero.ctaCompare}
        </Link>
      </div>

      <section className="card p-5">
        <h2 className="font-bold mb-3">{t.hero.liveTitle}</h2>
        {state === "loading" && <p className="text-sm opacity-70">{t.hero.liveLoading}</p>}
        {state === "error" && <p className="text-sm text-red-600">{t.hero.liveError}</p>}
        {state === "ok" && resp && (
          <div className="flex flex-wrap items-center gap-6">
            <Gauge proba={resp.proba} label={verdict} />
            <div>
              <p className="text-2xl font-bold">{verdict}</p>
              <p className="text-sm opacity-70">
                {t.hero.cropProba} = {resp.proba.toFixed(3)} · {resp.backend}
              </p>
              <p className="text-sm mt-2 max-w-md">{resp.recommendation}</p>
            </div>
          </div>
        )}
      </section>
    </div>
  );
}
