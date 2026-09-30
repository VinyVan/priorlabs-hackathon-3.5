export default function Gauge({ proba, label }: { proba: number; label: string }) {
  const pct = Math.max(0, Math.min(1, proba));
  const r = 54;
  const c = 2 * Math.PI * r;
  const filled = c * pct;
  const color = pct >= 0.5 ? "#2f7d32" : "#b3541e";
  return (
    <div className="flex items-center gap-4">
      <svg width="140" height="140" viewBox="0 0 140 140" role="img" aria-label="gauge">
        <circle cx="70" cy="70" r={r} fill="none" strokeWidth="14" stroke="#e5e7eb" />
        <circle
          cx="70"
          cy="70"
          r={r}
          fill="none"
          strokeWidth="14"
          stroke={color}
          strokeLinecap="round"
          strokeDasharray={`${filled} ${c}`}
          transform="rotate(-90 70 70)"
        />
        <text x="70" y="66" textAnchor="middle" fontSize="22" fontWeight="bold" fill="currentColor">
          {(pct * 100).toFixed(1)}%
        </text>
        <text x="70" y="86" textAnchor="middle" fontSize="11" fill="currentColor">
          {label}
        </text>
      </svg>
    </div>
  );
}
