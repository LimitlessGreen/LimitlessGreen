"""Generate the spectrogram profile banner (light + dark SVG).

Synthesizes a short "dawn chorus" of bird-like syllables plus the word
"LimitlessGreen" written into the audio as tones, computes a real
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
    # natural pitch jitter: a slow random walk of a few tens of Hz
    jitter = np.cumsum(RNG.normal(0, 1, n))
    jitter = (jitter - np.linspace(0, jitter[-1], n)) / (np.sqrt(n) + 1e-9)
    f = f_of_t(u) * (1 + 0.004 * jitter)
    phase = 2 * np.pi * np.cumsum(f) / SR
    env = np.sin(np.pi * u) ** 1.5 * (1 + 0.15 * np.sin(2 * np.pi * RNG.uniform(20, 60) * u * dur))
    # drop harmonics above Nyquist (they would alias back as mirrored ghosts)
    s = sum(a * np.sin((k + 1) * phase) * ((k + 1) * f < 0.95 * SR / 2)
            for k, a in enumerate(harmonics))
    signal[i0:i0 + n] += amp * env * s


# --- spectrogram text -----------------------------------------------------
# "LimitlessGreen" is written *into the sound*: the text is rendered as a
# bitmap (rows = frequencies, columns = time) and every row becomes a sine
# tone whose loudness follows the pixels.  The STFT below then reveals it.

TEXT = "LimitlessGreen"
TEXT_T = (3.45, 8.55)        # seconds
TEXT_F = (5000, 9700)        # Hz
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def write_text():
    from PIL import ImageDraw, ImageFilter, ImageFont

    try:
        font = ImageFont.truetype(FONT, 200)
    except OSError:
        font = ImageFont.load_default(200)
    x0, y0, x1, y1 = font.getbbox(TEXT)
    mask = Image.new("L", (x1 - x0 + 20, y1 - y0 + 20))
    ImageDraw.Draw(mask).text((10 - x0, 10 - y0), TEXT, font=font, fill=255)

    # one tone on every 3rd STFT bin centre: a Hann window spreads a
    # bin-centred tone over exactly its own bin +-1, so every bin is fed by
    # a single tone -> no beating between neighbours, clean letters
    bin_hz = SR / 1024
    freqs = np.arange(np.ceil(TEXT_F[0] / bin_hz / 3) * 3,
                      TEXT_F[1] / bin_hz, 3) * bin_hz
    n_rows = len(freqs)
    n_cols = int((TEXT_T[1] - TEXT_T[0]) * 400)         # 2.5 ms envelope steps
    mask = mask.resize((n_cols, n_rows), Image.LANCZOS).filter(
        ImageFilter.GaussianBlur(0.8))
    m = np.flipud(np.asarray(mask, dtype=float) / 255)  # row 0 = lowest freq

    i0, i1 = int(TEXT_T[0] * SR), int(TEXT_T[1] * SR)
    t = t_all[i0:i1]
    cols = np.linspace(0, n_cols - 1, len(t))
    for j, f in enumerate(freqs):
        if m[j].max() < 0.05:
            continue
        env = np.interp(cols, np.arange(n_cols), m[j])
        signal[i0:i1] += 0.2 * env * np.sin(2 * np.pi * f * t + RNG.uniform(0, 2 * np.pi))


# --- the "chorus" (below the text) ----------------------------------------
# Left third stays quiet-ish so the title is readable.  Harmonics are kept
# weak so they don't scribble over the lettering.
H_SOFT = (1.0, 0.05)

# 1) descending whistles, repeated
for k, st in enumerate(np.arange(3.7, 4.75, 0.26)):
    hi = 4300 - 200 * (k % 2) + RNG.normal(0, 100)
    st += RNG.normal(0, 0.02)
    add(st, lambda u, hi=hi: hi - 2000 * u ** 0.8, 0.12 + RNG.normal(0, 0.01),
        0.9 * RNG.uniform(0.7, 1.0), harmonics=H_SOFT)

# 2) fast trill
for k, st in enumerate(np.arange(4.85, 5.85, 0.055)):
    lo = 2900 - 400 * k / 18
    add(st, lambda u, lo=lo: lo + 1100 * np.sin(np.pi * u), 0.04,
        0.55 * (0.4 + 0.6 * np.sin(np.pi * k / 18)), harmonics=H_SOFT)

# 3) warbling FM phrase
add(6.05, lambda u: 2500 + 600 * np.sin(2 * np.pi * 8 * u) + 900 * u, 0.85, 0.8,
    harmonics=H_SOFT)

# 4) up-slurred "whee-oo" calls
for st in (7.15, 7.75, 8.35):
    add(st, lambda u: 1700 + 2300 * np.sin(np.pi * u) ** 2, 0.4, 0.9,
        harmonics=H_SOFT)

# 5) a few soft, low calls drifting into the title area (very faint)
for st, f0 in ((0.6, 2200), (1.5, 2600), (2.4, 2400)):
    add(st, lambda u, f0=f0: f0 + 700 * u, 0.25, 0.12)

# 6) high, thin "seee" notes above the text
for st in (3.4, 6.2):
    add(st, lambda u: 11000 - 300 * u, 0.45, 0.15, harmonics=(1.0,))

# 7) distant birds: many faint syllables everywhere -> a real dawn chorus
for _ in range(70):
    st = RNG.uniform(0, DUR - 0.3)
    f0 = RNG.uniform(1500, 8500)
    sweep = RNG.uniform(-1500, 1500)
    curve = RNG.choice([1.0, 0.5, 2.0])
    add(st, lambda u, f0=f0, sw=sweep, c=curve: np.clip(f0 + sw * u ** c, 600, 11500),
        RNG.uniform(0.04, 0.25), RNG.uniform(0.01, 0.05), harmonics=H_SOFT)

# a bit of room: exponentially decaying noise as impulse response (forest reverb)
ir_t = np.arange(int(0.35 * SR)) / SR
ir = RNG.normal(size=len(ir_t)) * np.exp(-ir_t / 0.07)
ir[0] = 0
wet = np.fft.irfft(np.fft.rfft(signal, 2 * len(signal)) * np.fft.rfft(ir, 2 * len(signal)))
signal += 0.06 * wet[:len(signal)] / (np.abs(ir).sum() ** 0.5)

write_text()

# background noise: pink-ish, stronger at low frequencies (wind / distance)
noise = RNG.normal(size=len(signal))
spec = np.fft.rfft(noise)
freqs = np.fft.rfftfreq(len(noise), 1 / SR)
spec /= np.sqrt(np.maximum(freqs, 30))
noise = np.fft.irfft(spec, n=len(noise))
signal += 0.03 * noise / noise.std()

# low rumble (wind, far-away traffic) below ~400 Hz
rumble = np.fft.rfft(RNG.normal(size=len(signal)))
rumble *= np.exp(-freqs / 180)
rumble = np.fft.irfft(rumble, n=len(signal))
signal += 0.15 * rumble / rumble.std()

# --- STFT -----------------------------------------------------------------
nfft, hop = 1024, int(SR * DUR / IMG_W)
win = np.hanning(nfft)
frames = np.lib.stride_tricks.sliding_window_view(
    np.pad(signal, (nfft // 2, nfft // 2)), nfft)[::hop][:IMG_W]
S = np.abs(np.fft.rfft(frames * win, axis=1)).T          # (freq, time)
S_db = 20 * np.log10(S + 1e-9)
S_db -= S_db.max()
v = np.clip((S_db + 78) / 78, 0, 1)                       # 78 dB range
v = np.flipud(v)                                          # high freq on top
v = np.array(Image.fromarray((v * 255).astype(np.uint8)).resize(
    (IMG_W, IMG_H), Image.BICUBIC)) / 255.0
v = v ** 1.2


def colormap(stops):
    xs = np.array([s[0] for s in stops])
    cols = np.array([[int(s[1][i:i + 2], 16) for i in (1, 3, 5)] for s in stops])
    return lambda x: np.stack([np.interp(x, xs, cols[:, c]) for c in range(3)], -1)


THEMES = {
    "dark": dict(
        cmap=colormap([(0, "#020806"), (0.3, "#06251a"), (0.5, "#0b5a33"),
                       (0.7, "#1f9a4a"), (0.86, "#7ed957"), (1, "#f2ffc8")]),
        bg="#020806", fg="#f0f6f0", muted="#9fb3a6", accent="#7ed957",
        grid="#ffffff", grid_op=0.06),
    "light": dict(
        cmap=colormap([(0, "#ffffff"), (0.3, "#eef7ee"), (0.5, "#b5dfb9"),
                       (0.7, "#4fae69"), (0.86, "#17703d"), (1, "#06331c")]),
        bg="#ffffff", fg="#14251a", muted="#4b5e52", accent="#17703d",
        grid="#000000", grid_op=0.06),
}


def img_b64(rgb):
    buf = io.BytesIO()
    Image.fromarray(rgb.astype(np.uint8)).save(buf, "WEBP", quality=82, method=6)
    return base64.b64encode(buf.getvalue()).decode()


def build(name, th):
    # full-bleed spectrogram; the title gets a soft scrim in the SVG instead
    rgb = th["cmap"](v)
    img = img_b64(rgb)

    khz_lines = "".join(
        f'<line x1="0" x2="{W}" y1="{H - k / 12 * H:.1f}" y2="{H - k / 12 * H:.1f}"/>'
        for k in (2, 4, 6, 8, 10))
    sec_ticks = "".join(
        f'<line x1="{s / DUR * W:.1f}" x2="{s / DUR * W:.1f}" y1="0" y2="6"/>'
        f'<text x="{s / DUR * W + 4:.1f}" y="15" stroke="none">{s} s</text>'
        for s in range(5, int(DUR) + 1))
    khz_labels = "".join(
        f'<text x="{W - 10}" y="{H - k / 12 * H - 4:.1f}" text-anchor="end">{k} kHz</text>'
        for k in (2, 4, 6, 8, 10))

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-label="Jairus: bioacoustics, audio tools and FPV. Spectrogram of bird calls with the word LimitlessGreen written into the sound.">
  <style>
    .t {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans", Helvetica, Arial, sans-serif; }}
    .m {{ font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace; }}
    .ph {{ animation: sweep 9s linear infinite; }}
    @keyframes sweep {{ from {{ transform: translateX({W * 0.30:.0f}px); }} to {{ transform: translateX({W}px); }} }}
    @media (prefers-reduced-motion: reduce) {{ .ph {{ animation: none; opacity: 0; }} }}
  </style>
  <defs>
    <clipPath id="r"><rect width="{W}" height="{H}" rx="12"/></clipPath>
    <linearGradient id="scrim" x1="0" x2="1">
      <stop offset="0" stop-color="{th['bg']}" stop-opacity="0.88"/>
      <stop offset="0.65" stop-color="{th['bg']}" stop-opacity="0.6"/>
      <stop offset="1" stop-color="{th['bg']}" stop-opacity="0"/>
    </linearGradient>
    <linearGradient id="g" x1="0" x2="1">
      <stop offset="0" stop-color="{th['accent']}" stop-opacity="0"/>
      <stop offset="1" stop-color="{th['accent']}" stop-opacity="0.35"/>
    </linearGradient>
  </defs>
  <g clip-path="url(#r)">
    <rect width="{W}" height="{H}" fill="{th['bg']}"/>
    <image width="{W}" height="{H}" preserveAspectRatio="none" href="data:image/webp;base64,{img}"/>
    <rect width="{W * 0.42:.0f}" height="{H}" fill="url(#scrim)"/>
    <g stroke="{th['grid']}" stroke-opacity="{th['grid_op']}" stroke-width="1">{khz_lines}</g>
    <g class="m" font-size="11" fill="{th['muted']}" stroke="{th['muted']}" fill-opacity="0.8">{sec_ticks}</g>
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
  <g class="m" font-size="15" fill="{th['accent']}">
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
