"""Forecast ranges from dev data only (tm30): Wilson 95% for p from 27/30, cost at bounds, OVERHEAD 40."""
import math, sys
from pathlib import Path

def wilson(k, n, z=1.959963984540054):
    ph = k / n; d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return c - h, c + h

S, P, Lm = 143.1, 177.5, 38.93          # tm30 means from crit.txt (scheme, PPM o1 in-domain, mean length)
R = 8 * Lm + 40                         # raw fallback with 40-bit overhead
lo, hi = wilson(27, 30)
out = [f"Wilson 27/30: {lo:.3f} - {hi:.3f}"]
for p in (lo, 0.85, 0.90, 0.95, hi):
    c = S + (1 - p) * (16 + R)
    out.append(f"p={p:.3f} cost {c:.1f} diff vs o1 {c - P:+.1f}")
Path(sys.argv[1]).write_text("\n".join(out) + "\n")
