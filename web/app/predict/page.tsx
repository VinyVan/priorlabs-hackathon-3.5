"use client";

import { useState } from "react";
import { useLang } from "../../components/Lang";
import Gauge from "../../components/Gauge";
import Bars, { Signal } from "../../components/Bars";

type PredictResp = {
  label: number;
  proba: number;
  backend: string;
  recommendation: string;
  explanation: Signal[];
  text: string;
  n_rows: number;
};

export default function PredictPage() {
  const { lang, t } = useLang();
  const [model, setModel] = useState<"tabpfn" | "baseline">("tabpfn");
  const [file, setFile] = useState<File | null>(null);
  const [resp, setResp] = useState<PredictResp | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState("");

  async function runWithRow(row: Record<string, unknown>) {
    setRunning(true);
    setError("");
    try {
      const r = await fetch("/api/py/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ row, model, lang }),
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      setResp(await r.json());
    } catch {
      setError(t.predict.error);
    } finally {
      setRunning(false);
    }
  }

  async function useSample() {
    setRunning(true);
    setError("");
    try {
      const s = await fetch("/api/py/sample").then((r) => {
        if (!r.ok) throw new Error("sample");
        return r.json();
      });
      await runWithRow(s.row);
    } catch {
      setError(t.predict.error);
      setRunning(false);
    }
  }

  async function runFile() {
    if (!file) return;
    setRunning(true);
    setError("");
    try {
      const fd = new FormData();
      fd.append("file", file);
      fd.append("model", model);
      fd.append("lang", lang);
      const r = await fetch("/api/py/predict", { method: "POST", body: fd });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      setResp(await r.json());
    } catch {
      setError(t.predict.error);
    } finally {
      setRunning(false);
    }
  }

  const verdict =
    resp == null ? "" : resp.label === 1 ? (lang === "fr" ? "Cultivée" : "Cropland") : lang === "fr" ? "Non cultivée" : "Not cropland";

  return (
    <div className="flex flex-col gap-5 pt-6">
      <h1 className="text-3xl font-extrabold">{t.predict.title}</h1>
      <p className="opacity-80">{t.predict.subtitle}</p>

      <div className="card p-5 flex flex-col gap-4">
        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold">{t.predict.model} :</span>
          <button
            onClick={() => setModel("tabpfn")}
            className={`px-3 py-1 rounded-lg text-sm font-medium ${model === "tabpfn" ? "bg-green-700 text-white" : "border"}`}
          >
            {t.predict.tabpfn}
          </button>
          <button
            onClick={() => setModel("baseline")}
            className={`px-3 py-1 rounded-lg text-sm font-medium ${model === "baseline" ? "bg-green-700 text-white" : "border"}`}
          >
            {t.predict.baseline}
          </button>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <button onClick={useSample} disabled={running} className="px-4 py-2 rounded-xl bg-green-700 text-white font-semibold disabled:opacity-50">
            {t.predict.useSample}
          </button>
          <label className="px-4 py-2 rounded-xl border cursor-pointer">
            {t.predict.upload}
            <input
              type="file"
              accept=".csv"
              className="hidden"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            />
          </label>
          {file && <span className="text-sm font-mono">{file.name}</span>}
          <button onClick={runFile} disabled={running || !file} className="px-4 py-2 rounded-xl border font-semibold disabled:opacity-50">
            {running ? t.predict.running : t.predict.run}
          </button>
        </div>
        <p className="text-xs opacity-60">{t.predict.hint}</p>
        {error && <p className="text-sm text-red-600">{error}</p>}
      </div>

      {resp && (
        <div className="card p-5 flex flex-col gap-4">
          <div className="flex flex-wrap items-center gap-6">
            <Gauge proba={resp.proba} label={verdict} />
            <div>
              <p className="text-2xl font-bold">{verdict}</p>
              <p className="text-sm opacity-70">
                {t.predict.proba} = {resp.proba.toFixed(3)} · {resp.backend} · n={resp.n_rows}
              </p>
            </div>
          </div>
          <div>
            <h3 className="font-bold mb-2">{t.predict.signals}</h3>
            <Bars signals={resp.explanation ?? []} />
          </div>
          <div>
            <h3 className="font-bold mb-1">{t.predict.reco}</h3>
            <p className="text-sm">{resp.recommendation}</p>
          </div>
        </div>
      )}
    </div>
  );
}
