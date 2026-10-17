// CONSILIUM mark: optic disc (cyan) and cup (magenta) — the same colours the
// server-rendered mask overlay uses.
export function BrandMark({ size = 28 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden="true" className="brand-mark">
      <circle cx="16" cy="16" r="14" fill="none" stroke="url(#bm-g)" strokeWidth="2" />
      <circle cx="18.5" cy="15" r="7" fill="none" stroke="#22d3ee" strokeWidth="2" />
      <circle cx="18.5" cy="15" r="3.4" fill="none" stroke="#f472d0" strokeWidth="2" />
      <defs>
        <linearGradient id="bm-g" x1="0" y1="0" x2="32" y2="32">
          <stop offset="0" stopColor="#60a5fa" />
          <stop offset="1" stopColor="#2dd4bf" />
        </linearGradient>
      </defs>
    </svg>
  );
}

export function Brand({ size }: { size?: number }) {
  return (
    <span className="brand">
      <BrandMark {...(size ? { size } : {})} />
      <span className="brand-word">CONSILIUM</span>
    </span>
  );
}
