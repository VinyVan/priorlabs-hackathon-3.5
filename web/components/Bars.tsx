export type Signal = { feature: string; value: number; z: number; global_score: number };

export default function Bars({ signals }: { signals: Signal[] }) {
  const max = Math.max(1e-9, ...signals.map((s) => Math.abs(s.z)));
  return (
    <div className="flex flex-col gap-2">
      {signals.map((s) => (
        <div key={s.feature}>
          <div className="flex justify-between text-xs">
            <span className="font-mono">{s.feature}</span>
            <span>
              {s.value.toFixed(3)} (z={s.z >= 0 ? "+" : ""}
              {s.z.toFixed(2)})
            </span>
          </div>
          <div className="h-2 rounded bg-gray-200 dark:bg-gray-700">
            <div
              className="h-2 rounded bg-green-700"
              style={{ width: `${(Math.abs(s.z) / max) * 100}%` }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}
