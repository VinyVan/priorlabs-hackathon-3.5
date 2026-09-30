"use client";

import { useEffect, useState } from "react";
import { useLang } from "../../components/Lang";

type Metrics = {
  n_train: number;
  n_test: number;
  n_features: number;
  tabpfn: { backend: string; accuracy: number; f1: number; auc: number };
  baseline: { backend: string; accuracy: number; f1: number; auc: number };
};

export default function ComparePage() {
  const { lang, t } = useLang();
  const [m, setM] = useState<Metrics | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    fetch("/api/py/metrics")
      .then((r) => {
        if (!r.ok) throw new Error("metrics");
        return r.json();
      })
      .then(setM)
      .catch(() => setError(true));
  }, []);

  if (error) return <p className="pt-6 text-sm text-red-600">{t.compare.error}</p>;
  if (!m) return <p className="pt-6 text-sm opacity-70">{t.compare.loading}</p>;

  const tabWins = m.tabpfn.auc >= m.baseline.auc;
  const winner =
    lang === "fr"
      ? tabWins
        ? `Gagnant : TabPFN-3.5 (AUC ${m.tabpfn.auc.toFixed(3)} vs ${m.baseline.auc.toFixed(3)}).`
        : `Gagnant : baseline ${m.baseline.backend} (AUC ${m.baseline.auc.toFixed(3)} vs ${m.tabpfn.auc.toFixed(3)}).`
      : tabWins
        ? `Winner: TabPFN-3.5 (AUC ${m.tabpfn.auc.toFixed(3)} vs ${m.baseline.auc.toFixed(3)}).`
        : `Winner: ${m.baseline.backend} baseline (AUC ${m.baseline.auc.toFixed(3)} vs ${m.tabpfn.auc.toFixed(3)}).`;

  const rows: Array<[string, number, number]> = [
    [t.compare.accuracy, m.tabpfn.accuracy, m.baseline.accuracy],
    [t.compare.f1, m.tabpfn.f1, m.baseline.f1],
    [t.compare.auc, m.tabpfn.auc, m.baseline.auc],
  ];

  return (
    <div className="flex flex-col gap-5 pt-6">
      <h1 className="text-3xl font-extrabold">{t.compare.title}</h1>
      <p className="opacity-80">{t.compare.subtitle}</p>
      <div className="card p-5 overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left border-b">
              <th className="py-2">{t.compare.metric}</th>
              <th className="py-2">TabPFN-3.5 ({m.tabpfn.backend})</th>
              <th className="py-2">
                Baseline ({m.baseline.backend})
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map(([k, a, b]) => (
              <tr key={k} className="border-b last:border-0">
                <td className="py-2 font-semibold">{k}</td>
                <td className="py-2 font-mono">{a.toFixed(3)}</td>
                <td className="py-2 font-mono">{b.toFixed(3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="text-xs opacity-60 mt-2">
          n_train={m.n_train} · n_test={m.n_test} · n_features={m.n_features}
        </p>
      </div>
      <p className="text-lg font-bold">{winner}</p>
    </div>
  );
}
