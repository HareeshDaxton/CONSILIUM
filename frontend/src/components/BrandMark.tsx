// CONSILIUM mark: an eye in the brand's warm fundus palette — outer field
// (amber → ember), retina (amber), optic disc (ivory). The cyan/magenta
// disc/cup colours stay reserved for the clinical mask overlay.
export function BrandMark({ size = 28 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden="true" className="brand-mark">
      <circle cx="16" cy="16" r="14" fill="none" stroke="url(#bm-g)" strokeWidth="2" />
      <circle cx="18.5" cy="15" r="7" fill="none" stroke="#f0a35e" strokeWidth="2" />
      <circle cx="18.5" cy="15" r="3.4" fill="#f7efe2" />
      <defs>
        <linearGradient id="bm-g" x1="0" y1="0" x2="32" y2="32">
          <stop offset="0" stopColor="#f2c46d" />
          <stop offset="1" stopColor="#e0703a" />
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
