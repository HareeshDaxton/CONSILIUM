"""Public HTTPS URL for this server via ngrok (local -> public).

Uses `pyngrok`, which downloads the ngrok agent automatically on first use.
Needs an ngrok account (free): put the authtoken in Deva/.env as NGROK_AUTHTOKEN
(https://dashboard.ngrok.com/get-started/your-authtoken). Optionally set
NGROK_DOMAIN to your free static domain (e.g. pasta-hanky-grunt.ngrok-free.dev)
so the URL stays the same every time you start the server.
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger("consilium-vision")


def _clean_domain(raw: str) -> str:
    domain = raw.strip()
    for prefix in ("https://", "http://"):
        if domain.startswith(prefix):
            domain = domain[len(prefix) :]
    return domain.rstrip("/")


def open_ngrok(port: int) -> str | None:
    """Start an ngrok tunnel to 127.0.0.1:<port>; return its https URL or None."""
    try:
        from pyngrok import ngrok
        from pyngrok.exception import PyngrokError
    except ImportError:
        log.error("pyngrok is not installed: run  pip install -r requirements.txt")
        return None

    # ngrok's own process log is noisy; our single message below explains failures.
    logging.getLogger("pyngrok").setLevel(logging.CRITICAL)
    token = os.environ.get("NGROK_AUTHTOKEN", "").strip()
    if token:
        ngrok.set_auth_token(token)
    options: dict[str, str] = {}
    domain = _clean_domain(os.environ.get("NGROK_DOMAIN", ""))
    if domain:
        options["domain"] = domain
    try:
        tunnel = ngrok.connect(str(port), "http", **options)
    except PyngrokError as exc:
        text = str(exc).lower()
        if "authtoken" in text or "authentication" in text or "err_ngrok_4018" in text:
            log.error(
                "ngrok needs your authtoken: put NGROK_AUTHTOKEN=... in Deva/.env "
                "(https://dashboard.ngrok.com/get-started/your-authtoken)"
            )
        elif "domain" in text:
            log.error(
                "ngrok refused NGROK_DOMAIN=%s - check the domain in your ngrok dashboard", domain
            )
        else:
            log.error("ngrok tunnel failed: %s", exc)
        return None
    url = tunnel.public_url or ""
    if url.startswith("http://"):
        url = "https://" + url[len("http://") :]
    return url or None


def close_ngrok() -> None:
    try:
        from pyngrok import ngrok

        ngrok.kill()
    except Exception as exc:  # best effort on shutdown
        log.debug("ngrok shutdown: %s", type(exc).__name__)
