# CONSILIUM vision server — for Deva's GPU laptop

This folder turns your GPU laptop into the **vision server** for CONSILIUM, reachable
over the internet through **ngrok** (local → public HTTPS).

```
Hareesh's laptop (CONSILIUM, CPU)                 Your laptop (this folder, GPU)
─────────────────────────────────                 ───────────────────────────────
user uploads fundus image + note
backend ── HTTPS POST /predict ──▶ ngrok ──▶      server.py
           (image bytes + token)                    SwinV2    -> P(glaucoma)
                                                    SegFormer -> optic disc + cup masks
backend ◀── JSON: probability + raw masks ────    (nothing stored, nothing else computed)
computes the cup-to-disc ratio, risk tier, quality checks,
runs the AI agents and writes the report
```

**Your server sends only the RAW model outputs.** It does **not** compute the
cup-to-disc ratio, a verdict, or any thresholds. CONSILIUM computes those itself, with
versioned and tested code: vertical CDR from the mask heights, calibrated tier, and
quality checks. This replaces the earlier `/predict`, `/segment` and `/diagnose` API,
whose CDR values are not used.

Models (pinned versions, Apache-2.0, fine-tuned on REFUGE, downloaded automatically):
- `pamixsun/swinv2_tiny_for_glaucoma_classification`: glaucoma classifier
- `pamixsun/segformer_for_optic_disc_cup_segmentation`: optic disc/cup segmenter

---

## What you need once

1. **NVIDIA GPU + driver.** `nvidia-smi` must work in a terminal.
2. **Python 3.10–3.12** from python.org (tick "Add python.exe to PATH").
3. **A free ngrok account:** https://dashboard.ngrok.com/signup
   - Copy your **authtoken**: https://dashboard.ngrok.com/get-started/your-authtoken
   - Optional, recommended: your free **static domain** (Dashboard → Domains), e.g.
     `pasta-hanky-grunt.ngrok-free.dev`. With it, the URL stays the same every time.

You do **not** need to install ngrok itself. It is downloaded automatically.

## Run (every time)

**Windows:** double-click **`run.bat`** (or run it in a terminal in this folder).
**Linux:** `./run.sh`

On the **first** run it automatically:
1. creates a Python environment (`.venv`) and installs PyTorch with CUDA plus the packages
   (a few GB, takes a while);
2. creates `Deva/.env` with a new shared **token**;
3. downloads both models into `models/` (about 290 MB);
4. loads them on the GPU and starts the server.

**First run only:** it will say *"ngrok needs your authtoken"*. Open `Deva/.env`, fill in

```
NGROK_AUTHTOKEN=<your authtoken>
NGROK_DOMAIN=<your static domain, optional>
```

then close the window and run `run.bat` again.

When it's ready it prints:

```
========================================================================
  CONSILIUM vision server is READY on NVIDIA GeForce RTX ....
========================================================================
  Send these two lines to Hareesh (he pastes them into his .env):

    VISION_ENDPOINT_URL=https://pasta-hanky-grunt.ngrok-free.dev
    VISION_ENDPOINT_TOKEN=3m1...xyz
```

**Send those two lines to Hareesh** privately (WhatsApp or similar; not in a public chat).
Keep the window open while he uses CONSILIUM. If you use a static domain, the URL and
token stay the same every run, so you only send them once.

## Check it works

In a second terminal in this folder:

```bat
.venv\Scripts\python check_server.py https://<your-ngrok-url>
```

It must end with `This server is ready for CONSILIUM.` You can also open
`https://<your-ngrok-url>/healthz` in a browser. It should show `"device": "cuda"`.

## Settings (`Deva/.env`)

| Setting | Meaning |
|---|---|
| `NGROK_AUTHTOKEN` | your ngrok authtoken (required for the public URL) |
| `NGROK_DOMAIN` | your static ngrok domain (optional; keeps the URL fixed) |
| `TUNNEL` | `ngrok` (default) or `none` (same Wi-Fi only: then send `http://<your-IP>:8100`) |
| `PORT` | local port, default `8100` |
| `DEVICE` | `auto` (GPU if available), `cuda` (refuse CPU), `cpu` |
| `CONSILIUM_VISION_TOKEN` | the shared secret; generated on first run. If you change it, tell Hareesh |

Different CUDA version? Before the first run: `set TORCH_INDEX=https://download.pytorch.org/whl/cu118`
(see https://pytorch.org/get-started/locally/). Then run `run.bat`.

## Troubleshooting

| Problem | Fix |
|---|---|
| `ngrok needs your authtoken` | Put `NGROK_AUTHTOKEN` in `Deva/.env`, restart. |
| `ngrok refused NGROK_DOMAIN` | The domain must be the one in *your* ngrok dashboard (or leave it empty). |
| `READY on CPU (no CUDA GPU)` | The CPU build of PyTorch got installed. Delete `.venv`, check `nvidia-smi`, set `TORCH_INDEX` to your CUDA version, run again. |
| Setup failed | Read the error, delete the `.venv` folder, run `run.bat` again. |
| Hareesh sees "Vision server offline" | Window closed, PC asleep, or ngrok URL changed (use a static domain). |
| Hareesh gets `401` | His `VISION_ENDPOINT_TOKEN` differs from `CONSILIUM_VISION_TOKEN` here. |
| Hareesh gets `VISION_REVISION_MISMATCH` | Use the `Deva` folder from the same project version as his. |

## Files

| File | What it does |
|---|---|
| `run.bat` / `run.sh` | One-click: setup on first run, then start |
| `server.py` | The server: token, model download, GPU loading, ngrok, `POST /predict`, `GET /healthz` |
| `tunnel.py` | ngrok tunnel (via pyngrok) |
| `wire.py` | Pinned model versions + the exact response format CONSILIUM expects (don't edit) |
| `download_models.py` | Model download (run automatically on first start) |
| `check_server.py` | Self-test of a running server (local or public URL) |
| `.env.example` | Settings template (`.env` is created from it automatically) |

Privacy: use only de-identified or research images (e.g. REFUGE). Images travel over
HTTPS (ngrok) or your LAN, are never written to disk, and only image size and timing are
logged. Every request must carry the shared token.
