"""Generate the spectrogram profile banner (light + dark SVG).

Synthesizes a short "dawn chorus" of bird-like syllables, computes a real
STFT spectrogram, colors it and embeds it as WebP in an SVG with axes, title
and an animated playhead.

    uv run --with numpy --with pillow scripts/make_banner.py
"""

import base64
import io
from pathlib import Path

import numpy as np
from PIL import Image

SR = 24_000          # -> 0..12 kHz on the y axis
DUR = 9.0            # seconds
W, H = 1280, 320     # SVG size
IMG_W, IMG_H = 1280, 320
RNG = np.random.default_rng(7)
OUT = Path(__file__).resolve().parent.parent / "assets"

t_all = np.arange(int(SR * DUR)) / SR
signal = np.zeros_like(t_all)


def add(start, f_of_t, dur, amp=1.0, harmonics=(1.0, 0.25, 0.08)):
    """Add a syllable with instantaneous frequency f_of_t(u), u in [0, 1]."""
    n = int(dur * SR)
    i0 = int(start * SR)
    if i0 + n > len(signal):
        n = len(signal) - i0
    u = np.arange(n) / max(n - 1, 1)
    f = f_of_t(u)
    phase = 2 * np.pi * np.cumsum(f) / SR
    env = np.sin(np.pi * u) ** 1.5
    # drop harmonics above Nyquist (they would alias back as mirrored ghosts)
    s = sum(a * np.sin((k + 1) * phase) * ((k + 1) * f < 0.95 * SR / 2)
            for k, a in enumerate(harmonics))
    signal[i0:i0 + n] += amp * env * s


# --- the "chorus" ---------------------------------------------------------
# Left third stays quiet-ish so the title is readable.

# 1) descending whistles (like a chiffchaff / tit), repeated
for k, st in enumerate(np.arange(3.2, 5.0, 0.32)):
    hi = 7200 - 250 * (k % 2) + RNG.normal(0, 120)
    st += RNG.normal(0, 0.02)
    add(st, lambda u, hi=hi: hi - 3200 * u ** 0.8, 0.12 + RNG.normal(0, 0.01),
        0.9 * RNG.uniform(0.7, 1.0))

# 2) fast trill
for k, st in enumerate(np.arange(5.3, 6.4, 0.055)):
    lo = 5200 - 600 * k / 20
    add(st, lambda u, lo=lo: lo + 1800 * np.sin(np.pi * u), 0.04,
        0.55 * (0.4 + 0.6 * np.sin(np.pi * k / 20)), harmonics=(1.0, 0.1))

# 3) warbling FM phrase
add(6.6, lambda u: 3600 + 900 * np.sin(2 * np.pi * 9 * u) + 1500 * u, 0.9, 0.8,
    harmonics=(1.0, 0.35, 0.12))

# 4) up-slurred "whee-oo" calls
for st in (7.7, 8.25):
    add(st, lambda u: 2600 + 3400 * np.sin(np.pi * u) ** 2, 0.42, 0.9,
        harmonics=(1.0, 0.18))

# 5) a few soft, low calls drifting into the title area (very faint)
for st, f0 in ((0.6, 2200), (1.5, 2600), (2.4, 2400)):
    add(st, lambda u, f0=f0: f0 + 700 * u, 0.25, 0.12)

# 6) high, thin "seee" notes in the background
for st in (3.9, 6.0, 8.0):
    add(st, lambda u: 9800 - 400 * u, 0.5, 0.18, harmonics=(1.0,))

# background noise: pink-ish, stronger at low frequencies (wind / distance)
noise = RNG.normal(size=len(signal))
spec = np.fft.rfft(noise)
freqs = np.fft.rfftfreq(len(noise), 1 / SR)
spec /= np.sqrt(np.maximum(freqs, 30))
noise = np.fft.irfft(spec, n=len(noise))
signal += 0.02 * noise / noise.std()

# --- STFT -----------------------------------------------------------------
nfft, hop = 1024, int(SR * DUR / IMG_W)
win = np.hanning(nfft)
frames = np.lib.stride_tricks.sliding_window_view(
    np.pad(signal, (nfft // 2, nfft // 2)), nfft)[::hop][:IMG_W]
S = np.abs(np.fft.rfft(frames * win, axis=1)).T          # (freq, time)
S_db = 20 * np.log10(S + 1e-9)
S_db -= S_db.max()
v = np.clip((S_db + 70) / 70, 0, 1)                       # 70 dB range
v = np.flipud(v)                                          # high freq on top
v = np.array(Image.fromarray((v * 255).astype(np.uint8)).resize(
    (IMG_W, IMG_H), Image.BICUBIC)) / 255.0
v = v ** 1.35


def colormap(stops):
    xs = np.array([s[0] for s in stops])
    cols = np.array([[int(s[1][i:i + 2], 16) for i in (1, 3, 5)] for s in stops])
    return lambda x: np.stack([np.interp(x, xs, cols[:, c]) for c in range(3)], -1)


THEMES = {
    "dark": dict(
        cmap=colormap([(0, "#0d1117"), (0.35, "#0f2a2a"), (0.6, "#11795a"),
                       (0.82, "#3fd68a"), (1, "#e8ffd0")]),
        bg="#0d1117", fg="#e6edf3", muted="#7d8590", accent="#3fd68a",
        grid="#ffffff", grid_op=0.06),
    "light": dict(
        cmap=colormap([(0, "#ffffff"), (0.35, "#e3f4ea"), (0.6, "#7cc79a"),
                       (0.82, "#1f8a57"), (1, "#0b3d2a")]),
        bg="#ffffff", fg="#1f2328", muted="#59636e", accent="#1a7f4b",
        grid="#000000", grid_op=0.06),
}


def img_b64(rgb):
    buf = io.BytesIO()
    Image.fromarray(rgb.astype(np.uint8)).save(buf, "WEBP", quality=82, method=6)
    return base64.b64encode(buf.getvalue()).decode()


def build(name, th):
    # fade the spectrogram out under the title on the left
    x = np.linspace(0, 1, IMG_W)
    fade = np.clip((x - 0.12) / 0.3, 0, 1) ** 1.2
    rgb = th["cmap"](v * (0.15 + 0.85 * fade)[None, :])
    img = img_b64(rgb)

    khz_lines = "".join(
        f'<line x1="0" x2="{W}" y1="{H - k / 12 * H:.1f}" y2="{H - k / 12 * H:.1f}"/>'
        for k in (2, 4, 6, 8, 10))
    khz_labels = "".join(
        f'<text x="{W - 10}" y="{H - k / 12 * H - 4:.1f}" text-anchor="end">{k} kHz</text>'
        for k in (2, 4, 6, 8, 10))

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-label="Jairus — LimitlessGreen: bioacoustics, audio tools and FPV">
  <style>
    .t {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans", Helvetica, Arial, sans-serif; }}
    .m {{ font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace; }}
    .ph {{ animation: sweep 9s linear infinite; }}
    @keyframes sweep {{ from {{ transform: translateX({W * 0.30:.0f}px); }} to {{ transform: translateX({W}px); }} }}
    @media (prefers-reduced-motion: reduce) {{ .ph {{ animation: none; opacity: 0; }} }}
  </style>
  <defs>
    <clipPath id="r"><rect width="{W}" height="{H}" rx="12"/></clipPath>
    <linearGradient id="g" x1="0" x2="1">
      <stop offset="0" stop-color="{th['accent']}" stop-opacity="0"/>
      <stop offset="1" stop-color="{th['accent']}" stop-opacity="0.35"/>
    </linearGradient>
  </defs>
  <g clip-path="url(#r)">
    <rect width="{W}" height="{H}" fill="{th['bg']}"/>
    <image width="{W}" height="{H}" preserveAspectRatio="none" href="data:image/webp;base64,{img}"/>
    <g stroke="{th['grid']}" stroke-opacity="{th['grid_op']}" stroke-width="1">{khz_lines}</g>
    <g class="m" font-size="12" fill="{th['muted']}" fill-opacity="0.8">{khz_labels}</g>
    <g class="ph">
      <rect x="-60" width="60" height="{H}" fill="url(#g)"/>
      <rect x="-1" width="2" height="{H}" fill="{th['accent']}"/>
    </g>
  </g>
  <g class="t">
    <text x="56" y="150" font-size="64" font-weight="700" fill="{th['fg']}" letter-spacing="-1">Jairus</text>
    <text x="58" y="190" font-size="22" fill="{th['muted']}">Bioacoustics · Audio tools · FPV</text>
  </g>
  <g class="m" font-size="17" fill="{th['accent']}">
    <text x="58" y="250">$ listen --to nature --and build tools for it</text>
  </g>
</svg>
'''
    path = OUT / f"banner-{name}.svg"
    path.write_text(svg)
    print(f"{path.name}: {len(svg) / 1024:.0f} kB")


OUT.mkdir(exist_ok=True)
for name, th in THEMES.items():
    build(name, th)
