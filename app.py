from fastapi import FastAPI, Query, Header, HTTPException, Response
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from fastapi import Request
import os, requests

# Import logiky z test.py
from test import (
    get_price,
    get_prices,
    get_prices_for_ean,
    upcitemdb_lookup,
    build_variants,
    google_cse_fetch,
    host_from_url,
    MARKET_CFG,
    MARKET_DOMAINS,
    GOOGLE_API_KEY,
    GOOGLE_CSE_CX,
)

app = FastAPI(title="Price Bridge", version="1.0")

# Block marketplaces/aggregators
BLOCKLIST = {"finn.no", "m.finn.no", "facebook.com", "m.facebook.com", "instagram.com", "proff.maxbo.no"}

# --- very simple in-memory rate limit (per IP per minute) ---
from time import time
_LIMIT_WINDOW = 60  # seconds
_LIMIT_REQ = 120    # requests per IP per minute
_rate: dict = {}

def allow(ip: str) -> bool:
    now = time()
    window = int(now // _LIMIT_WINDOW)
    key = (ip, window)
    _rate[key] = _rate.get(key, 0) + 1
    return _rate[key] <= _LIMIT_REQ

@app.get("/health")
def health():
    return {"ok": True, "ts": datetime.utcnow().isoformat()}


# --- Root and favicon endpoints ---
@app.get("/")
def root():
    return {"ok": True, "message": "Price Bridge API. See /docs", "ts": datetime.utcnow().isoformat()}

@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)

# --- Jednoduchý token auth ---
API_TOKEN = os.getenv("API_TOKEN", "moj-super-tajny-token")  # z ENV alebo default

def check_token(authorization: Optional[str]):
    if not API_TOKEN:
        return  # DEV mode: no auth
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing token")
    if authorization.split(" ", 1)[1] != API_TOKEN:
        raise HTTPException(status_code=403, detail="Bad token")

# --- Auto-register with WordPress (optional, set via ENV) ---
WP_URL = os.getenv("WP_URL")  # e.g. https://tvoja-wp-domena.sk
WP_SHARED_SECRET = os.getenv("WP_SHARED_SECRET", "pb-secret-123")
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL")  # e.g. https://xyz.ngrok-free.app or Render URL (no trailing slash)

def _register_with_wp():
    if not WP_URL or not PUBLIC_BASE_URL:
        return
    try:
        r = requests.post(
            f"{WP_URL.rstrip('/')}/wp-json/price-bridge/v1/register",
            json={
                "url": PUBLIC_BASE_URL.rstrip('/'),
                "token": API_TOKEN,
                "secret": WP_SHARED_SECRET,
            },
            timeout=10,
        )
        r.raise_for_status()
        print("[PriceBridge] Registered with WP:", r.text)
    except Exception as e:
        print("[PriceBridge] WP register failed:", e)

@app.on_event("startup")
def _on_startup():
    _register_with_wp()

# --- Schémy odpovedí ---
class PriceItem(BaseModel):
    retailer: str
    url: str
    price: Optional[str] = None
    price_regular: Optional[str] = None
    price_member: Optional[str] = None
    ts: str

class EanResult(BaseModel):
    items: List[PriceItem]

# --- Endpoints ---
@app.get("/price", response_model=PriceItem)
def endpoint_price(
        url: str = Query(..., description="Produktové URL"),
        authorization: Optional[str] = Header(None),
        request: Request = None
):
    check_token(authorization)
    if request and not allow(request.client.host):
        raise HTTPException(status_code=429, detail="Too many requests")
    p = get_prices(url)
    return {
        "retailer": host_from_url(url),
        "url": url,
        "price": p.get("price"),
        "price_regular": p.get("price_regular"),
        "price_member": p.get("price_member"),
        "ts": datetime.utcnow().isoformat(),
    }

@app.get("/ean", response_model=EanResult)
def endpoint_ean(
        ean: str = Query(..., description="EAN/UPC"),
        market: Optional[str] = Query(None, pattern=r"^(NO|SE|DK|DE|FR|UK|US|CZ|PL)?$"),
        limit: int = 10,
        per_domain: int = 1,
        sites: Optional[str] = None,
        only_no: int = 1,  # Default to .no domains only
        include_empty: int = 0,
        authorization: Optional[str] = Header(None),
        request: Request = None
):
    check_token(authorization)
    if request and not allow(request.client.host):
        raise HTTPException(status_code=429, detail="Too many requests")

    out: List[PriceItem] = []
    printed_links = set()
    seen_per_domain: dict = {}

    # 1) UPCitemdb (preferované, ale filtruj per-domain)
    offers = upcitemdb_lookup(ean)
    for o in offers[: max(1, limit)]:
        link = o.get("link")
        if not link:
            continue
        host = host_from_url(link)
        if host in BLOCKLIST:
            continue
        if only_no and not host.endswith(".no"):
            continue
        # removed TLD restriction to allow all markets
        if per_domain > 0 and seen_per_domain.get(host, 0) >= per_domain:
            continue
        printed_links.add(link)
        seen_per_domain[host] = seen_per_domain.get(host, 0) + 1
        out.append(PriceItem(
            retailer=host,
            url=link,
            price=o.get("price"),
            price_regular=None,
            price_member=None,
            ts=datetime.utcnow().isoformat(),
        ))
        if len(out) >= limit:
            return {"items": out}

    # 2) Google CSE (ak máš kľúče)
    if not (os.getenv("GOOGLE_API_KEY", GOOGLE_API_KEY) and os.getenv("GOOGLE_CSE_CX", GOOGLE_CSE_CX)):
        return {"items": out}

    hl = MARKET_CFG.get(market, {}).get("hl")
    gl = MARKET_CFG.get(market, {}).get("gl")
    default_sites = ""
    sites_arg = sites if sites else default_sites

    queries = build_variants(ean, keywords="", sites=sites_arg)
    api_key = os.getenv("GOOGLE_API_KEY", GOOGLE_API_KEY)
    cx = os.getenv("GOOGLE_CSE_CX", GOOGLE_CSE_CX)

    for q in queries:
        if len(out) >= limit:
            break
        for page in range(0, 2):  # max 20 výsledkov/query
            start = 1 + page * 10
            data = google_cse_fetch(api_key, cx, q, hl, gl, num=10, start=start)
            items = data.get("items") or []
            if not items:
                break
            for it in items:
                link = it.get("link")
                if not link or link in printed_links:
                    continue
                host = host_from_url(link)
                if host in BLOCKLIST:
                    continue
                if only_no and not host.endswith(".no"):
                    continue
                # removed TLD restriction to allow all markets
                if per_domain > 0 and seen_per_domain.get(host, 0) >= per_domain:
                    continue

                prices = get_prices_for_ean(link, ean)
                if not prices or not prices.get("price"):
                    if include_empty:
                        out.append(PriceItem(
                            retailer=host,
                            url=link,
                            price=None,
                            price_regular=None,
                            price_member=None,
                            ts=datetime.utcnow().isoformat(),
                        ))
                        printed_links.add(link)
                        seen_per_domain[host] = seen_per_domain.get(host, 0) + 1
                    continue

                out.append(PriceItem(
                    retailer=host,
                    url=link,
                    price=prices.get("price"),
                    price_regular=prices.get("price_regular"),
                    price_member=prices.get("price_member"),
                    ts=datetime.utcnow().isoformat(),
                ))
                printed_links.add(link)
                seen_per_domain[host] = seen_per_domain.get(host, 0) + 1
                if len(out) >= limit:
                    break


    return {"items": out}