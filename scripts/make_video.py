"""Render the banner as an MP4 with a phosphor-screen playhead.

The beam sweeps left to right in sync with the audio.  Everything it passes
flares up (slightly over-driven towards white, with a bit of bloom) and then
decays exponentially back to a faint afterglow -- like a CRT / oscilloscope.

    uv run --with numpy --with pillow --with playwright scripts/make_video.py OUT.mp4

GitHub only shows a video player for files uploaded through the web UI, so
the MP4 is not committed; drag it into the README editor on github.com.
"""

import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_banner as mb  # noqa: E402  (regenerates the banner + wav)

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "banner-listen.mp4").resolve()
SCALE = 1.5
VW, VH = int(mb.W * SCALE), int(mb.H * SCALE)        # 1920 x 480
FPS = 30
TAIL = 0.8                                           # let the glow die out

# phosphor model: shared with the SVG banner
RESIDUAL, FLARE, TAU, TAIL = mb.RESIDUAL, mb.FLARE, mb.TAU, mb.TAIL
BEAM_W = 3           # px (video)

th = mb.THEMES["dark"]


def overlay_png():
    """Title, axes and formulas as a transparent layer (no spectrogram)."""
    from playwright.sync_api import sync_playwright

    svg = (mb.OUT / "banner-dark.svg").read_text()
    svg = re.sub(r"<image [^>]*/>", "", svg)
    svg = svg.replace(f'<rect width="{mb.W}" height="{mb.H}" fill="{th["bg"]}"/>', "")
    svg = svg.replace('<g class="ph beam">', '<g class="ph beam" style="display:none">')
    tmp = OUT.with_suffix(".overlay.svg")
    tmp.write_text(svg)
    with sync_playwright() as pw:
        b = pw.chromium.launch()
        p = b.new_page(viewport={"width": mb.W, "height": mb.H}, device_scale_factor=SCALE)
        p.goto(tmp.as_uri())
        p.wait_for_timeout(300)
        png = p.screenshot(omit_background=True)
        b.close()
    tmp.unlink()
    import io
    return np.asarray(Image.open(io.BytesIO(png)).convert("RGBA"), dtype=float) / 255


def main():
    ov = overlay_png()
    ov_rgb, ov_a = ov[..., :3] * 255, ov[..., 3:4]

    v = np.asarray(Image.fromarray((mb.v * 255).astype(np.uint8)).resize(
        (VW, VH), Image.BICUBIC), dtype=float) / 255
    col_t = (np.arange(VW) + 0.5) / VW * mb.DUR          # time of each column
    beam_col = np.array(th["beam"], float)

    n_frames = int((mb.DUR + TAIL) * FPS)
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{VW}x{VH}", "-r", str(FPS), "-i", "-",
         "-i", str(mb.OUT / "banner.wav"),
         "-af", f"apad=pad_dur={TAIL}",
         "-c:v", "libx264", "-preset", "slow", "-crf", "22", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "160k", "-t", f"{mb.DUR + TAIL}",
         "-movflags", "+faststart", str(OUT)],
        stdin=subprocess.PIPE)

    for i in range(n_frames):
        t = i / FPS
        age = t - col_t
        lit = age >= 0
        gain = RESIDUAL + np.where(lit, FLARE * np.exp(-np.clip(age, 0, None) / TAU), 0)
        e = v * gain[None, :]                               # excitation

        rgb = mb.phosphor(e, th)

        # bloom around the freshly excited part
        fresh = np.where(lit & (age < 3 * TAU), np.exp(-np.clip(age, 0, None) / TAU), 0)
        if fresh.any():
            rgb = rgb + mb.bloom(v * fresh[None, :], th)

        # the beam itself, with a soft halo
        if t <= mb.DUR:
            x = t / mb.DUR * VW
            d = np.abs(np.arange(VW) - x)
            halo = np.exp(-(d / 14) ** 2) * 0.25 + (d < BEAM_W / 2) * 0.85
            rgb = rgb * (1 - halo[None, :, None]) + beam_col * halo[None, :, None]

        rgb = rgb * (1 - ov_a) + ov_rgb * ov_a
        ff.stdin.write(np.clip(rgb, 0, 255).astype(np.uint8).tobytes())

    ff.stdin.close()
    ff.wait()
    print(f"{OUT.name}: {OUT.stat().st_size / 1024:.0f} kB, {n_frames} frames")


if __name__ == "__main__":
    main()
