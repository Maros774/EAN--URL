# test.py  —  Clean Python scraper: URL price extraction + UPCitemdb + Google CSE (ported from PHP)
# Usage:
#   python test.py --url <URL1> [URL2 ...]
#   python test.py --ean <EAN1> [EAN2 ...] [--limit 20] [--per-domain 1] [--market NO|SE] [--sites dom1,dom2]

import os
import re
import sys
import json
import html
import argparse
from typing import Optional, Tuple, List, Iterable, Dict
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', '')
GOOGLE_CSE_CX  = os.getenv('GOOGLE_CSE_CX',  '')

# UPCitemdb (trial)
UPCITEMDB_TRIAL_URL = 'https://api.upcitemdb.com/prod/trial/lookup'

HEADERS = {
    'User-Agent': 'PriceScraper/1.3 (+https://example.local)',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en,nb;q=0.9,sv;q=0.9,da;q=0.9,sk;q=0.8',
}

# Exclude marketplaces/aggregators we don't want
BLOCKLIST = {"finn.no", "m.finn.no", "facebook.com", "m.facebook.com", "instagram.com", "proff.maxbo.no"}

# Sites that REQUIRE Selenium (JavaScript-heavy, won't work with conventional fetch)
JS_HEAVY_SITES = {"maxbo.no", "proff.maxbo.no"}

# --- Regex building blocks (allow normal/NBSP/thin spaces) ---
SP = r"[\s\u00A0\u202F]*"
NUM = r"(?:\d{1,3}(?:[\.,\s\u00A0\u202F]\d{3})*(?:[\.,]\d{2})?|\d+)"
CODES = r"(?:EUR|USD|GBP|CHF|PLN|CZK|HUF|RON|NOK|SEK|DKK|ISK|BGN|RSD|TRY)"
SYMBOLS_BEFORE = r"(?:€|£|\$)"
AFTER_TOKENS = rf"(?:€|{CODES}|Kč|CZK|zł|PLN|Ft|HUF|lei|RON|CHF|NOK|SEK|DKK|ISK|BGN|RSD|TRY|kr)"

RE_BEFORE = re.compile(rf"(?:{SYMBOLS_BEFORE}|{CODES}){SP}{NUM}", re.I | re.U)
RE_AFTER  = re.compile(rf"{NUM}{SP}{AFTER_TOKENS}", re.I | re.U)


RE_JSONLD_SCRIPT = re.compile(r'<script[^>]+type\s*=\s*["\']application/ld\+json["\'][^>]*>(.*?)</script>', re.I | re.S)
RE_NEXT_DATA = re.compile(r'<script[^>]+id=["\']__NEXT_DATA__["\'][^>]*>(.*?)</script>', re.I | re.S)


RE_META_PRICE = [
    re.compile(r'<meta[^>]+itemprop="price"[^>]+content="([^"]+)"', re.I),
    re.compile(r'<meta[^>]+property="product:price:amount"[^>]+content="([^"]+)"', re.I),
    re.compile(r'<meta[^>]+property="og:price:amount"[^>]+content="([^"]+)"', re.I),
]

# --- Helpers for extracting prices from JSON/Next.js blobs ---
COMMON_PRICE_KEYS = {
    "price", "priceValue", "currentPrice", "salePrice", "salesPrice",
    "grossPrice", "unitPrice", "amount", "value", "cost", "total",
    # Norwegian/Scandinavian terms
    "pris", "belop", "beløp", "sum", "kost",
    # Member/loyalty pricing
    "memberPrice", "clubPrice", "loyaltyPrice", "medlemspris",
    # Regular pricing
    "regularPrice", "listPrice", "originalPrice", "normalpris",
    # Maxbo-specific patterns
    "displayPrice", "sellPrice", "retailPrice", "consumerPrice"
}

def _find_prices_in_json(obj, path: str = "") -> List[dict]:
    """Enhanced price finder that tracks context and price types"""
    found: List[dict] = []

    if isinstance(obj, dict):
        for k, v in obj.items():
            current_path = f"{path}.{k}" if path else k
            lk = str(k).lower()

            # Check if this key indicates a price
            if lk in {pk.lower() for pk in COMMON_PRICE_KEYS} and isinstance(v, (int, float, str)):
                price_type = "member" if any(term in lk for term in ["member", "medlem", "club", "loyalty"]) else \
                            "regular" if any(term in lk for term in ["regular", "list", "original", "normal"]) else \
                            "base"
                found.append({
                    "value": str(v),
                    "type": price_type,
                    "path": current_path,
                    "key": k
                })
            else:
                found.extend(_find_prices_in_json(v, current_path))

    elif isinstance(obj, list):
        for i, it in enumerate(obj):
            found.extend(_find_prices_in_json(it, f"{path}[{i}]" if path else f"[{i}]"))

    return found

def _extract_best_prices_from_json_results(results: List[dict]) -> dict:
    """Extract best prices from JSON search results"""
    prices = {"price": None, "price_regular": None, "price_member": None}

    # Group by type
    by_type = {"base": [], "regular": [], "member": []}
    for result in results:
        by_type[result["type"]].append(result["value"])

    # Get best from each category
    if by_type["base"]:
        prices["price"] = _best_from_candidates(by_type["base"])
    if by_type["regular"]:
        prices["price_regular"] = _best_from_candidates(by_type["regular"])
    if by_type["member"]:
        prices["price_member"] = _best_from_candidates(by_type["member"])

    return prices


def from_nextdata(html_text: str) -> Optional[str]:
    """Extract price from __NEXT_DATA__ - legacy function for compatibility"""
    enhanced_result = from_nextdata_enhanced(html_text)
    return enhanced_result.get("price") if enhanced_result else None

def from_nextdata_enhanced(html_text: str) -> Optional[dict]:
    """Enhanced __NEXT_DATA__ extraction with price type detection"""
    for m in RE_NEXT_DATA.finditer(html_text):
        blob = m.group(1).strip()
        if not blob:
            continue
        try:
            data = json.loads(blob)
        except Exception:
            continue

        price_results = _find_prices_in_json(data)
        if price_results:
            prices = _extract_best_prices_from_json_results(price_results)
            if any(prices.values()):
                return prices
    return None


def from_nextdata_raw(html_text: str) -> Optional[dict]:
    for m in RE_NEXT_DATA.finditer(html_text):
        blob = m.group(1).strip()
        if not blob:
            continue
        try:
            return json.loads(blob)
        except Exception:
            continue
    return None

# --- Relaxed JSON loader for slightly invalid JSON-LD blobs ---

def json_load_relaxed(blob: str):
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        # Remove trailing commas before } or ]
        cleaned = re.sub(r",\s*(\}|\])", r"\1", blob)
        # Unescape stray control chars if any remain problematic
        try:
            return json.loads(cleaned)
        except Exception:
            return None
    except Exception:
        return None

def deep_get(obj, path: List[str]):
    cur = obj
    for k in path:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return None
    return cur


# --- Hints for member vs regular pricing ---
MEMBER_HINTS = re.compile(r"(member|medlem|fordel|bonus|klubb|club)", re.I)
REGULAR_HINTS = re.compile(r"(ordin\u00E6r|veiledende|uten\s*medlemskap|non-?member|ikke\s*medlem|normalpris)", re.I)

RE_PRICE_BLOCKS = [
    re.compile(r'<(?:div|span)[^>]+(?:class|id)\s*=\s*"[^"]*(?:product-price__price|product-price|product_price|product__price|price__|price--|price-|prisbel[øo]p|kampanjepris|totalpris|pris|amount)[^"]*"[^>]*>(.{0,4000}?)</\s*(?:div|span)>', re.I | re.S),
    re.compile(r'<(?:div|span)[^>]+(?:class|id)\s*=\s*"[^"]*(?:price|pris)[^"]*"[^>]*>(.{0,8000}?)</\s*(?:div|span)>', re.I | re.S),
]

# -------- Utils --------

def _strip_tags(html_text: str) -> str:
    text = re.sub(r"(?is)<script.*?>.*?</script>", " ", html_text)
    text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    # Decode HTML entities including &nbsp;
    text = html.unescape(text)
    # Handle Norwegian thousand separators and spaces
    text = re.sub(r"(\d)\s+(\d{3})", r"\1\2", text)  # Remove spaces in numbers like "2 490"
    text = re.sub(r"[\s\u00A0\u202F]+", " ", text).strip()
    return text


def host_from_url(url: str) -> str:
    try:
        from urllib.parse import urlparse
        h = urlparse(url).netloc.lower()
    except Exception:
        return url
    return re.sub(r'^www\.', '', h)


def page_contains_ean(html_text: str, ean: str) -> bool:
    ean = re.sub(r"\D+", "", ean or "")
    if not ean:
        return False
    variants = {ean}
    if len(ean) == 13:
        variants.add(ean[1:])  # UPC-A varianta

    # 1) plain text (toleruje medzery a pomlčky)
    pats = [re.compile("".join([c + (r"[\s\-]?" if i < len(v)-1 else "") for i, c in enumerate(v)])) for v in variants]
    for rx in pats:
        if rx.search(html_text or ""):
            return True

    # 2) JSON-LD (gtin/ean polia)
    for sm in RE_JSONLD_SCRIPT.finditer(html_text or ""):
        blob = sm.group(1).strip()
        try:
            data = json.loads(blob)
        except Exception:
            continue
        stack = [data]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                for key in ("gtin", "gtin13", "gtin12", "ean", "ean13", "ean12", "barcode"):
                    if key in node:
                        digits = re.sub(r"\D+", "", str(node[key]))
                        if digits in variants:
                            return True
                stack.extend(node.values())
            elif isinstance(node, list):
                stack.extend(node)

    # 3) __NEXT_DATA__ (celý JSON strom)
    nd = from_nextdata_raw(html_text or "")
    if nd is not None:
        stack = [nd]
        while stack:
            node = stack.pop()
            if isinstance(node, (str, int, float)):
                digits = re.sub(r"\D+", "", str(node))
                if digits in variants:
                    return True
            elif isinstance(node, dict):
                stack.extend(node.values())
            elif isinstance(node, list):
                stack.extend(node)
    return False


def price_to_number(raw: str) -> Tuple[Optional[float], Optional[str]]:
    currency = None
    if re.search(r"€|EUR", raw, re.I): currency = "EUR"
    elif re.search(r"\$|USD", raw, re.I): currency = "USD"
    elif re.search(r"£|GBP", raw, re.I): currency = "GBP"
    elif re.search(r"\bNOK\b|\skr\b", raw, re.I): currency = "NOK" if "NOK" in raw.upper() else None
    elif re.search(r"\bSEK\b", raw, re.I): currency = "SEK"
    elif re.search(r"\bDKK\b", raw, re.I): currency = "DKK"
    elif re.search(r"\bCZK\b|Kč", raw, re.I): currency = "CZK"
    elif re.search(r"\bPLN\b|zł", raw, re.I): currency = "PLN"
    elif re.search(r"\bCHF\b", raw, re.I): currency = "CHF"
    elif re.search(r"\bHUF\b|Ft", raw, re.I): currency = "HUF"
    elif re.search(r"\bRON\b|lei", raw, re.I): currency = "RON"

    digits = re.sub(r"[^\d,\.]", "", raw)
    if "," in digits and "." in digits:
        digits = digits.replace(".", "").replace(",", ".")
    else:
        if re.search(r",\d{1,2}$", digits):
            digits = digits.replace(".", "").replace(",", ".")
        else:
            digits = digits.replace(",", "")
    try:
        val = float(digits)
    except ValueError:
        return None, currency
    return val, currency


# --- Normalization helpers (Norway) ---

def _fmt_number_no(val: float) -> str:
    # Show integer without decimals; otherwise show 2 decimals
    if float(val).is_integer():
        return f"{int(val)}"
    return f"{val:.2f}".rstrip('0').rstrip('.')


def normalize_price_no(raw: Optional[str]) -> Optional[str]:
    """Normalize a raw price string into Norwegian consumer format: 'kr <value>'.
    For Norwegian sites, be more aggressive about converting to kr format."""
    if not raw:
        return None
    val, cur = price_to_number(raw)
    if val is None:
        return None
    # For Norwegian sites, convert NOK and reasonable prices to kr format
    # Only keep foreign currency if it's clearly a major international currency with large amounts
    if cur in ('EUR', 'USD', 'GBP') and val > 100:
        return raw  # Keep major foreign currencies for expensive items
    # Convert NOK, unknown currency, and everything else to Norwegian kr format
    return f"kr {_fmt_number_no(val)}"


def _best_from_candidates(raws: Iterable[str]) -> Optional[str]:
    best_raw, best_val = None, None
    for r in raws:
        val, _cur = price_to_number(r)
        if val is None:
            continue
        if best_val is None or val < best_val:
            best_val, best_raw = val, str(r).strip()
    return best_raw


def extract_prices_from_text(text: str) -> List[str]:
    cands: List[str] = []
    for m in RE_BEFORE.finditer(text):
        cands.append(m.group(0).strip())
    for m in RE_AFTER.finditer(text):
        cands.append(m.group(0).strip())
    seen, uniq = set(), []
    for x in cands:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq

# -------- Price extraction from HTML --------

def from_jsonld(html_text: str) -> Optional[str]:
    for sm in RE_JSONLD_SCRIPT.finditer(html_text):
        blob = sm.group(1).strip()
        if not blob:
            continue
        try:
            data = json.loads(blob)
        except json.JSONDecodeError:
            try:
                data = json.loads(re.sub(r",\s*([}\]])", r"\1", blob))
            except Exception:
                continue
        nodes = data if isinstance(data, list) else [data]
        expanded: List[dict] = []
        for n in nodes:
            expanded.append(n)
            if isinstance(n, dict) and "@graph" in n and isinstance(n["@graph"], list):
                expanded.extend(n["@graph"])

        # Look for direct price fields in any node (Maxbo pattern)
        cands: List[str] = []
        def extract_prices_from_node(node):
            if isinstance(node, dict):
                # Direct price field (like Maxbo's "price":"2490.00")
                if 'price' in node and isinstance(node['price'], (str, int, float)):
                    price_val = str(node['price'])
                    if re.search(r'\d', price_val):
                        cands.append(f"kr {price_val}")

                # Traditional offers structure
                offers = node.get('offers')
                if offers:
                    offers_list = offers if isinstance(offers, list) else [offers]
                    for o in offers_list:
                        if isinstance(o, dict):
                            price = o.get('price') or o.get('lowPrice')
                            cur = o.get('priceCurrency')
                            if price is not None:
                                raw = f"{cur} {price}" if cur else str(price)
                                cands.append(raw)

                # Recurse into nested objects
                for v in node.values():
                    if isinstance(v, (dict, list)):
                        extract_prices_from_node(v)
            elif isinstance(node, list):
                for item in node:
                    extract_prices_from_node(item)

        extract_prices_from_node(data)

        if cands:
            best = _best_from_candidates(cands)
            if best:
                return best
    return None


def from_meta(html_text: str) -> Optional[str]:
    cur = None
    amt = None
    for rx in RE_META_PRICE:
        m = rx.search(html_text)
        if m:
            amt = m.group(1).strip()
    mcur = re.search(r'<meta[^>]+(priceCurrency|og:price:currency)[^>]+content="([^"]+)"', html_text, re.I)
    if mcur:
        cur = mcur.group(2).strip().upper()
    if amt:
        raw = f"{cur} {amt}" if cur else amt
        if extract_prices_from_text(raw):
            return raw
    return None


def from_price_blocks(html_text: str) -> Optional[str]:
    blocks: List[str] = []
    for rx in RE_PRICE_BLOCKS:
        for m in rx.finditer(html_text):
            blk = _strip_tags(m.group(1))
            if blk:
                blocks.append(blk)
    if not blocks:
        return None
    cands: List[str] = []
    for b in blocks:
        # Try normal price extraction first
        found_prices = extract_prices_from_text(b)
        if found_prices:
            cands.extend(found_prices)
        else:
            # For Norwegian sites, try to extract standalone numbers and add "kr"
            number_match = re.search(r'\b(\d{2,5}(?:[.,]\d{2})?)\b', b)
            if number_match:
                num = number_match.group(1)
                val, _ = price_to_number(num)
                if val and 10 <= val <= 100000:  # Reasonable price range
                    cands.append(f"kr {num}")

    if not cands:
        return None
    return _best_from_candidates(cands)


def from_visible_text(html_text: str) -> Optional[str]:
    vis = _strip_tags(html_text)
    cands = extract_prices_from_text(vis)
    if not cands:
        return None
    return _best_from_candidates(cands)

# --- Generic extractor for correct variant by EAN ---

# --- EAN + nearby-price helpers ---

def _digits_only(val: Optional[str]) -> str:
    return re.sub(r"\D+", "", str(val or ""))

PRICE_KEY_CANDIDATES = {
    "price", "currentPrice", "salePrice", "salesPrice", "grossPrice", "unitPrice",
    "amount", "value", "current", "now", "priceValue", "listPrice", "regularPrice",
    "memberPrice", "clubPrice"
}

NESTED_PRICE_PATHS = [
    ["price", "value"], ["price", "amount"], ["priceInfo", "current"],
    ["prices", "current"], ["prices", "now"], ["pricing", "price"],
]

def _find_price_nearby(node: dict) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Try to extract (base, regular, member) from node by common keys or nested paths."""
    def get_any(d: dict, keys: Iterable[str]) -> Optional[str]:
        for k in keys:
            if k in d and isinstance(d[k], (str, int, float)):
                return str(d[k])
        return None

    base = get_any(node, PRICE_KEY_CANDIDATES)
    for path in NESTED_PRICE_PATHS:
        cur = node
        ok = True
        for p in path:
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            else:
                ok = False
                break
        if ok and isinstance(cur, (str, int, float)) and not base:
            base = str(cur)
            break

    price_regular = get_any(node, {"regularPrice", "listPrice", "oldPrice", "originalPrice"})
    price_member = get_any(node, {"memberPrice", "clubPrice", "loyaltyPrice"})
    return base, price_regular, price_member


def _dict_has_matching_ean(d: dict, ean_digits: str) -> bool:
    # Matches when a field equals the EAN OR contains it as a contiguous digit substring.
    # This covers cases like URLs or variant IDs that embed the EAN.
    if not ean_digits:
        return False
    pat = re.compile(rf"(?<!\d){re.escape(ean_digits)}(?!\d)")
    for k, v in d.items():
        # 1) Prefer explicit EAN/GTIN/UPC keys
        if k.lower() in {"gtin", "gtin13", "gtin12", "ean", "ean13", "ean12", "barcode", "upc"}:
            if _digits_only(v) == ean_digits:
                return True
        # 2) Generic string/number fields may still contain the EAN (e.g., inside a URL or SKU)
        if isinstance(v, (str, int, float)):
            s = str(v)
            # Fast path: exact digits-only equality
            if _digits_only(s) == ean_digits:
                return True
            # Substring path: look for EAN as its own digit-run inside the string
            if isinstance(v, str) and pat.search(s):
                return True
    return False

def extract_price_for_matching_ean(html_text: str, expected_ean: str) -> Optional[dict]:
    ean_digits = _digits_only(expected_ean)
    if not ean_digits:
        return None

    # A) JSON-LD: match node by EAN and pull price from the same node OR its 'offers'
    for sm in RE_JSONLD_SCRIPT.finditer(html_text or ""):
        data = json_load_relaxed(sm.group(1))
        if data is None:
            continue
        nodes = data if isinstance(data, list) else [data]
        expanded: List[dict] = []
        for n in nodes:
            expanded.append(n)
            if isinstance(n, dict) and "@graph" in n and isinstance(n["@graph"], list):
                expanded.extend(n["@graph"])
        for n in expanded:
            if not isinstance(n, dict):
                continue
            # direct EAN match on node
            if _dict_has_matching_ean(n, ean_digits):
                base, pr_reg, pr_mem = _find_price_nearby(n)
                # also check offers attached to the node
                offers = n.get("offers")
                offers_list = offers if isinstance(offers, list) else ([offers] if isinstance(offers, dict) else [])
                for o in offers_list:
                    if not isinstance(o, dict):
                        continue
                    b2, r2, m2 = _find_price_nearby(o)
                    # Parent Product already matched the EAN → allow taking price from Offer
                    if b2 or r2 or m2:
                        base = base or b2
                        pr_reg = pr_reg or r2
                        pr_mem = pr_mem or m2
                if base or pr_reg or pr_mem:
                    return {
                        "price": base,
                        "price_regular": pr_reg,
                        "price_member": pr_mem,
                    }
    # B) __NEXT_DATA__: find dict containing the EAN and read price from that dict, then parent, then grandparent.
    nd = from_nextdata_raw(html_text or "")
    if nd is not None:
        stack: List[Tuple[object, Optional[dict], Optional[dict]]] = [(nd, None, None)]
        while stack:
            node, parent, gparent = stack.pop()
            if isinstance(node, dict):
                if _dict_has_matching_ean(node, ean_digits):
                    for src in (node, parent or {}, gparent or {}):
                        base, pr_reg, pr_mem = _find_price_nearby(src)
                        if base or pr_reg or pr_mem:
                            return {
                                "price": base,
                                "price_regular": pr_reg,
                                "price_member": pr_mem,
                            }
                for v in node.values():
                    # push child with parents tracked
                    if isinstance(v, (dict, list)):
                        stack.append((v, node, parent))
            elif isinstance(node, list):
                for it in node:
                    if isinstance(it, (dict, list)):
                        stack.append((it, parent, gparent))
    return None

# --- Per-site extractors ---

def extract_obsbygg(html_text: str, expected_ean: str) -> Optional[dict]:
    ean_digits = _digits_only(expected_ean)
    if not ean_digits:
        return None

    # Prefer __NEXT_DATA__ with parent/ancestor price lookup
    nd = from_nextdata_raw(html_text or "")
    if nd:
        stack: List[Tuple[object, Optional[dict], Optional[dict]]] = [(nd, None, None)]
        while stack:
            node, parent, gparent = stack.pop()
            if isinstance(node, dict):
                if _dict_has_matching_ean(node, ean_digits):
                    for src in (node, parent or {}, gparent or {}):
                        base, pr_reg, pr_mem = _find_price_nearby(src)
                        if base or pr_reg or pr_mem:
                            return {
                                "price": base,
                                "price_regular": pr_reg,
                                "price_member": pr_mem,
                            }
                for v in node.values():
                    if isinstance(v, (dict, list)):
                        stack.append((v, node, parent))
            elif isinstance(node, list):
                for it in node:
                    if isinstance(it, (dict, list)):
                        stack.append((it, parent, gparent))

    # Fallback to JSON-LD node/offer matching
    for sm in RE_JSONLD_SCRIPT.finditer(html_text or ""):
        data = json_load_relaxed(sm.group(1))
        if data is None:
            continue
        nodes = data if isinstance(data, list) else [data]
        expanded = []
        for n in nodes:
            expanded.append(n)
            if isinstance(n, dict) and "@graph" in n and isinstance(n["@graph"], list):
                expanded.extend(n["@graph"])
        for n in expanded:
            if not isinstance(n, dict):
                continue
            if _dict_has_matching_ean(n, ean_digits):
                base, pr_reg, pr_mem = _find_price_nearby(n)
                offers = n.get("offers")
                offers_list = offers if isinstance(offers, list) else ([offers] if isinstance(offers, dict) else [])
                for o in offers_list:
                    if isinstance(o, dict):
                        b2, r2, m2 = _find_price_nearby(o)
                        # Parent Product already matched EAN, allow taking price from Offer
                        if b2 or r2 or m2:
                            base = base or b2
                            pr_reg = pr_reg or r2
                            pr_mem = pr_mem or m2
                if base or pr_reg or pr_mem:
                    return {"price": base, "price_regular": pr_reg, "price_member": pr_mem}
    return None


# --- Maxbo.no / proff.maxbo.no extractor ---

# --- Maxbo.no / proff.maxbo.no extractor ---

def extract_maxbo(html_text: str, expected_ean: str) -> Optional[dict]:
    """Extractor for maxbo.no / proff.maxbo.no.
    1) Try __NEXT_DATA__ with EAN match on node/parents.
    2) Try JSON-LD Product.offers {priceCurrency, price} even if Offer lacks EAN (parent matched).
    3) Fallback: read visible price from product-price__price / product-price__amount / data-price.
    """
    ean_digits = _digits_only(expected_ean)

    # 1) Enhanced __NEXT_DATA__ extraction with better price detection
    enhanced_prices = from_nextdata_enhanced(html_text or "")
    if enhanced_prices and any(enhanced_prices.values()):
        # For Maxbo, be less strict about EAN validation since it often works without explicit EAN
        return enhanced_prices

    # 1b) Fallback: traditional __NEXT_DATA__ with parent/ancestor price lookup
    nd = from_nextdata_raw(html_text or "")
    if nd:
        stack: List[Tuple[object, Optional[dict], Optional[dict]]] = [(nd, None, None)]
        while stack:
            node, parent, gparent = stack.pop()
            if isinstance(node, dict):
                if ean_digits and _dict_has_matching_ean(node, ean_digits):
                    for src in (node, parent or {}, gparent or {}):
                        base, pr_reg, pr_mem = _find_price_nearby(src)
                        if base or pr_reg or pr_mem:
                            return {
                                "price": base,
                                "price_regular": pr_reg,
                                "price_member": pr_mem,
                            }
                for v in node.values():
                    if isinstance(v, (dict, list)):
                        stack.append((v, node, parent))
            elif isinstance(node, list):
                for it in node:
                    if isinstance(it, (dict, list)):
                        stack.append((it, parent, gparent))

    # 2) JSON-LD Product.offers (as seen on Maxbo screenshot)
    for sm in RE_JSONLD_SCRIPT.finditer(html_text or ""):
        data = json_load_relaxed(sm.group(1))
        if data is None:
            continue
        nodes = data if isinstance(data, list) else [data]
        expanded: List[dict] = []
        for n in nodes:
            expanded.append(n)
            if isinstance(n, dict) and "@graph" in n and isinstance(n["@graph"], list):
                expanded.extend(n["@graph"])
        for n in expanded:
            if not isinstance(n, dict):
                continue
            # Prefer Product that matches the EAN; if not present, we will still accept the first Product with offers
            if ean_digits and not _dict_has_matching_ean(n, ean_digits):
                # try non-strict fallback later
                pass
            offers = n.get("offers")
            offers_list = offers if isinstance(offers, list) else ([offers] if isinstance(offers, dict) else [])
            base = pr_reg = pr_mem = None
            for o in offers_list:
                if not isinstance(o, dict):
                    continue
                b2, r2, m2 = _find_price_nearby(o)
                if b2 or r2 or m2:
                    base = base or b2
                    pr_reg = pr_reg or r2
                    pr_mem = pr_mem or m2
            if base or pr_reg or pr_mem:
                return {"price": base, "price_regular": pr_reg, "price_member": pr_mem}
        # Secondary pass: accept first Product.offers price even without explicit EAN on node (Maxbo case)
        for n in expanded:
            if isinstance(n, dict) and n.get('@type') in ('Product', ['Product']):
                offers = n.get('offers')
                offers_list = offers if isinstance(offers, list) else ([offers] if isinstance(offers, dict) else [])
                base = pr_reg = pr_mem = None
                for o in offers_list:
                    if isinstance(o, dict):
                        b2, r2, m2 = _find_price_nearby(o)
                        if b2 or r2 or m2:
                            base = base or b2
                            pr_reg = pr_reg or r2
                            pr_mem = pr_mem or m2
                if base or pr_reg or pr_mem:
                    return {"price": base, "price_regular": pr_reg, "price_member": pr_mem}

    # 3) Enhanced HTML block extraction for Maxbo
    maxbo_price_patterns = [
        # Primary price selectors
        r'<div[^>]+class="[^"]*product-price__price[^"]*"[^>]*>(.*?)</div>',
        r'<div[^>]+class="[^"]*product-price__amount[^"]*"[^>]*>(.*?)</div>',
        r'<span[^>]+class="[^"]*price[^"]*"[^>]*>(.*?)</span>',

        # Member vs regular price selectors
        r'<div[^>]+class="[^"]*member-?price[^"]*"[^>]*>(.*?)</div>',
        r'<div[^>]+class="[^"]*regular-?price[^"]*"[^>]*>(.*?)</div>',
        r'<div[^>]+class="[^"]*normal-?price[^"]*"[^>]*>(.*?)</div>',

        # React/JS rendered price containers
        r'<div[^>]+data-testid="[^"]*price[^"]*"[^>]*>(.*?)</div>',
        r'<div[^>]+data-cy="[^"]*price[^"]*"[^>]*>(.*?)</div>',

        # Generic price containers with common Norwegian terms
        r'<div[^>]+class="[^"]*pris[^"]*"[^>]*>(.*?)</div>',
        r'<span[^>]+class="[^"]*pris[^"]*"[^>]*>(.*?)</span>',
    ]

    price_results = {"price": None, "price_regular": None, "price_member": None}
    all_found_prices = []

    for pattern in maxbo_price_patterns:
        for m in re.finditer(pattern, html_text or "", re.I | re.S):
            raw_txt = _strip_tags(m.group(1))
            cands = extract_prices_from_text(raw_txt)
            if cands:
                all_found_prices.extend(cands)

                # Classify by context
                context = m.group(0).lower()
                best = _best_from_candidates(cands)
                if best:
                    if 'member' in context or 'medlem' in context:
                        price_results["price_member"] = price_results["price_member"] or best
                    elif 'regular' in context or 'normal' in context or 'ordinær' in context:
                        price_results["price_regular"] = price_results["price_regular"] or best
                    else:
                        price_results["price"] = price_results["price"] or best

    # Try data attributes
    for attr_pattern in [r'data-price\s*=\s*"([^"]+)"', r'data-amount\s*=\s*"([^"]+)"', r'data-cost\s*=\s*"([^"]+)"']:
        for m in re.finditer(attr_pattern, html_text or "", re.I):
            if m.group(1):
                all_found_prices.append(m.group(1))

    # If we found any prices but didn't classify them, use the best one as main price
    if all_found_prices and not any(price_results.values()):
        price_results["price"] = _best_from_candidates(all_found_prices)

    # Return if we found anything
    if any(price_results.values()):
        return price_results

    # 4) Final fallback: try aggressive text mining for Norwegian price patterns
    return extract_maxbo_fallback_prices(html_text or "")


def extract_maxbo_fallback_prices(html_text: str) -> Optional[dict]:
    """Aggressive fallback price extraction for Maxbo when other methods fail"""
    if not html_text:
        return None

    # Remove scripts and styles to focus on visible content
    cleaned = re.sub(r'(?is)<script.*?>.*?</script>', ' ', html_text)
    cleaned = re.sub(r'(?is)<style.*?>.*?</style>', ' ', cleaned)

    all_prices = []

    # Look for Norwegian currency patterns in the entire page
    norwegian_patterns = [
        r'\b(\d+(?:[.,]\d{3})*(?:[.,]\d{2})?)\s*kr\b',  # "1234 kr" or "1,234.50 kr"
        r'\bkr\s*(\d+(?:[.,]\d{3})*(?:[.,]\d{2})?)\b',  # "kr 1234"
        r'\b(\d+(?:[.,]\d{3})*(?:[.,]\d{2})?)\s*NOK\b', # "1234 NOK"
        r'\bNOK\s*(\d+(?:[.,]\d{3})*(?:[.,]\d{2})?)\b', # "NOK 1234"
    ]

    for pattern in norwegian_patterns:
        for match in re.finditer(pattern, cleaned, re.I):
            price_text = match.group(0)
            if extract_prices_from_text(price_text):
                all_prices.append(price_text)

    # Also look for standalone numbers that might be prices (in reasonable price ranges)
    # Be much more conservative - only accept numbers that look like real prices
    number_pattern = r'\b(\d{2,}[.,]?\d{3}(?:[.,]\d{2})?|\d{3,}(?:[.,]\d{2})?)\b'
    for match in re.finditer(number_pattern, cleaned):
        try:
            # Convert to float to check if it's a reasonable price
            num_str = match.group(1).replace(',', '.')
            num_val = float(num_str)
            if 10 <= num_val <= 50000:  # Reasonable price range, exclude zero and tiny values
                all_prices.append(f"kr {num_str}")
        except ValueError:
            continue

    if all_prices:
        best_price = _best_from_candidates(all_prices)
        if best_price:
            return {"price": best_price, "price_regular": None, "price_member": None}

    return None

# --- Member/regular price detection ---
def detect_prices(html_text: str) -> dict:
    """Return dict with potential base/regular/member prices detected from HTML."""
    # Gather candidates from price blocks
    candidates: List[Tuple[str, str]] = []  # (context, raw)
    for rx in RE_PRICE_BLOCKS:
        for m in rx.finditer(html_text):
            blk = _strip_tags(m.group(1))
            if not blk:
                continue
            for c in extract_prices_from_text(blk):
                candidates.append((blk, c))

    price_regular = None
    price_member = None
    for ctx, raw in candidates:
        if price_member is None and MEMBER_HINTS.search(ctx):
            price_member = raw
        elif price_regular is None and REGULAR_HINTS.search(ctx):
            price_regular = raw

    # Enhanced extraction order for Norwegian sites
    base = None

    # 1) Try Next.js/React state first (covers modern sites)
    base = from_nextdata(html_text)

    # 2) Try traditional structured methods first (more reliable)
    if not base:
        base = from_jsonld(html_text) or from_meta(html_text)

    # 3) Try price blocks (CSS selectors - reliable)
    if not base:
        base = from_price_blocks(html_text)

    # 4) Filter candidates by currency preference (medium reliability)
    if not base and candidates:
        # Prefer Norwegian currency
        norwegian_candidates = [raw for _ctx, raw in candidates if any(term in raw.lower() for term in ['kr', 'nok'])]
        foreign_candidates = [raw for _ctx, raw in candidates if not any(term in raw.lower() for term in ['kr', 'nok'])]

        if norwegian_candidates:
            base = _best_from_candidates(norwegian_candidates)
        elif foreign_candidates:
            base = _best_from_candidates(foreign_candidates)

    # 5) Enhanced Norwegian extraction (aggressive, lower reliability)
    if not base:
        base = extract_norwegian_prices_enhanced(html_text)

    # 6) Visible text as final fallback
    if not base:
        base = from_visible_text(html_text)

    return {"price": base, "price_regular": price_regular, "price_member": price_member}


def extract_norwegian_prices_enhanced(html_text: str) -> Optional[str]:
    """Enhanced Norwegian price extraction for modern e-commerce sites."""
    if not html_text:
        return None

    # Norwegian-specific patterns for modern e-commerce sites
    norwegian_patterns = [
        # React/Vue/Angular component data attributes
        r'data-price["\s]*:[\s]*["\']([^"\']+)["\']',
        r'"price"[:\s]*"([^"]+)"',
        r'"currentPrice"[:\s]*"([^"]+)"',
        r'"displayPrice"[:\s]*"([^"]+)"',
        r'"sellPrice"[:\s]*"([^"]+)"',
        r'"amount"[:\s]*(\d+(?:\.\d+)?)',

        # Norwegian specific classes and IDs
        r'class="[^"]*pris[^"]*"[^>]*>([^<]+)</[^>]*>',
        r'id="[^"]*pris[^"]*"[^>]*>([^<]+)</[^>]*>',
        r'data-testid="[^"]*price[^"]*"[^>]*>([^<]+)</[^>]*>',

        # Modern price containers with Norwegian currency
        r'<span[^>]*class="[^"]*price[^"]*"[^>]*>([^<]*\d+[^<]*kr[^<]*)</span>',
        r'<div[^>]*class="[^"]*price[^"]*"[^>]*>([^<]*\d+[^<]*kr[^<]*)</div>',
        r'<p[^>]*class="[^"]*price[^"]*"[^>]*>([^<]*\d+[^<]*kr[^<]*)</p>',

        # Direct Norwegian currency patterns with word boundaries
        r'\b(\d+(?:[.,\s]\d{3})*(?:[.,]\d{2})?\s*kr)\b',
        r'\b(kr\s*\d+(?:[.,\s]\d{3})*(?:[.,]\d{2})?)\b',
        r'\b(\d+(?:[.,\s]\d{3})*(?:[.,]\d{2})?\s*NOK)\b',

        # JavaScript variable assignments (common in Norwegian sites)
        r'price["\s]*:["\s]*(\d+(?:\.\d+)?)',
        r'pris["\s]*:["\s]*(\d+(?:\.\d+)?)',
        r'amount["\s]*:["\s]*(\d+(?:\.\d+)?)',
    ]

    all_matches = []
    for pattern in norwegian_patterns:
        for match in re.finditer(pattern, html_text, re.I):
            price_text = match.group(1).strip()

            # Skip if empty or too long
            if not price_text or len(price_text) > 30:
                continue

            # Clean and validate
            if re.search(r'\d', price_text):
                # Try to extract with existing function first
                prices_found = extract_prices_from_text(price_text)
                if prices_found:
                    all_matches.extend(prices_found)
                else:
                    # Handle direct number matches
                    num_match = re.search(r'(\d+(?:[.,]\d+)?)', price_text)
                    if num_match:
                        num_part = num_match.group(1)
                        # Convert to Norwegian format
                        all_matches.append(f"kr {num_part}")

    if all_matches:
        # Filter reasonable prices and prefer Norwegian currency
        reasonable_prices = []
        for price in all_matches:
            val, cur = price_to_number(price)
            if val and 10 <= val <= 100000:  # Reasonable price range for Norwegian products
                reasonable_prices.append(price)

        if reasonable_prices:
            return _best_from_candidates(reasonable_prices)

    return None


def is_js_heavy_site(host: str) -> bool:
    """Check if a host requires Selenium due to JavaScript-heavy rendering"""
    return any(host.endswith(site) for site in JS_HEAVY_SITES)


def extract_prices_from_html(html_text: str, local_host: str, expected_ean: Optional[str] = None, debug: bool = False) -> dict:
    """
    Unified price extraction logic using all available extractors.

    Strategy:
    1. Try per-site extractors (obsbygg, maxbo, etc.)
    2. Try generic JSON-LD/__NEXT_DATA__ extraction (with optional EAN matching)
    3. Fallback to generic price detection
    """
    if debug:
        print(f"[debug] host={local_host} -> trying per-site extractors")

    # 1) Per-site extractor(s)
    if local_host.endswith("obsbygg.no") and expected_ean:
        if debug:
            print("[debug] obsbygg extractor: start")
        out = extract_obsbygg(html_text, expected_ean)
        if out:
            if local_host.endswith('.no'):
                out["price"] = normalize_price_no(out.get("price")) or out.get("price")
                out["price_regular"] = normalize_price_no(out.get("price_regular")) or out.get("price_regular")
                out["price_member"] = normalize_price_no(out.get("price_member")) or out.get("price_member")
            if debug:
                print(f"[debug] obsbygg extractor: OK price={out.get('price')} regular={out.get('price_regular')} member={out.get('price_member')}")
            return out
        elif debug:
            print("[debug] obsbygg extractor: no match")

    if (local_host.endswith("maxbo.no") or local_host.endswith("proff.maxbo.no")) and expected_ean:
        if debug:
            print("[debug] maxbo extractor: start")
        out = extract_maxbo(html_text, expected_ean)
        if out:
            if local_host.endswith('.no'):
                out["price"] = normalize_price_no(out.get("price")) or out.get("price")
                out["price_regular"] = normalize_price_no(out.get("price_regular")) or out.get("price_regular")
                out["price_member"] = normalize_price_no(out.get("price_member")) or out.get("price_member")
            if debug:
                print(f"[debug] maxbo extractor: OK price={out.get('price')} regular={out.get('price_regular')} member={out.get('price_member')}")
            return out
        elif debug:
            print("[debug] maxbo extractor: no match")

    # 2) Generic: match variant by EAN from JSON-LD / __NEXT_DATA__ (if EAN provided)
    if expected_ean:
        if debug:
            print("[debug] generic JSON-LD/__NEXT_DATA__: start")
        out = extract_price_for_matching_ean(html_text, expected_ean)
        if out:
            if local_host.endswith('.no'):
                out["price"] = normalize_price_no(out.get("price")) or out.get("price")
                out["price_regular"] = normalize_price_no(out.get("price_regular")) or out.get("price_regular")
                out["price_member"] = normalize_price_no(out.get("price_member")) or out.get("price_member")
            if debug:
                print(f"[debug] generic JSON-LD/__NEXT_DATA__: OK price={out.get('price')} regular={out.get('price_regular')} member={out.get('price_member')}")
            return out
        elif debug:
            print("[debug] generic JSON-LD/__NEXT_DATA__: no match")

        # 3) Smart fallback with EAN validation
        ean_found = page_contains_ean(html_text, expected_ean)
        if debug:
            print(f"[debug] page_contains_ean: {ean_found}")

    # 4) Generic price detection as final fallback
    if debug:
        print("[debug] fallback detect_prices: start")
    d = detect_prices(html_text)
    if local_host.endswith('.no'):
        d["price"] = normalize_price_no(d.get("price")) or d.get("price")
        d["price_regular"] = normalize_price_no(d.get("price_regular")) or d.get("price_regular")
        d["price_member"] = normalize_price_no(d.get("price_member")) or d.get("price_member")

    if expected_ean:
        # If EAN not found but we got prices, be more cautious (lower confidence)
        if not page_contains_ean(html_text, expected_ean) and any(d.values()):
            if debug:
                print("[debug] fallback detect_prices: OK (no EAN validation, lower confidence)")
        elif debug:
            print(f"[debug] fallback detect_prices: {'OK' if (d.get('price') or d.get('price_regular') or d.get('price_member')) else 'no price found'}")
    elif debug:
        print(f"[debug] fallback detect_prices: {'OK' if (d.get('price') or d.get('price_regular') or d.get('price_member')) else 'no price found'}")

    return d


def get_prices(url: str, debug: bool = False) -> dict:
    """
    Extract prices from a URL with intelligent Selenium strategy.

    Strategy:
    - For JS-heavy sites (Maxbo): Use Selenium FIRST (required)
    - For other sites: Try conventional fetch first, then Selenium as fallback
    """
    host = host_from_url(url)

    # Determine wait element based on known sites
    wait_element = None
    if host.endswith("maxbo.no") or host.endswith("proff.maxbo.no"):
        wait_element = '.product-price__price, .product-price__amount, [data-testid*="price"]'

    # For JS-heavy sites: Use Selenium FIRST (they require JavaScript)
    if is_js_heavy_site(host):
        if debug:
            print(f"[get_prices] JS-heavy site detected, using Selenium first: {url}")

        html_text_selenium = selenium_fetch(url, timeout=20, wait_for_element=wait_element)
        if html_text_selenium:
            result = extract_prices_from_html(html_text_selenium, host, expected_ean=None, debug=debug)
            if result and (result.get("price") or result.get("price_regular") or result.get("price_member")):
                if debug:
                    print(f"[get_prices] Selenium fetch succeeded with price")
                return result
            elif debug:
                print(f"[get_prices] Selenium fetch got HTML but no price found")
        else:
            if debug:
                print(f"[get_prices] Selenium fetch failed")

        # If Selenium failed, don't try conventional fetch (it won't work for JS-heavy sites)
        return {}

    # For regular sites: Try conventional fetch first
    if debug:
        print(f"[get_prices] trying conventional fetch: {url}")
    html_text = fetch(url)

    if html_text:
        result = extract_prices_from_html(html_text, host, expected_ean=None, debug=debug)
        # If we found a price, return it
        if result and (result.get("price") or result.get("price_regular") or result.get("price_member")):
            if debug:
                print(f"[get_prices] conventional fetch succeeded with price")
            return result
        elif debug:
            print(f"[get_prices] conventional fetch got HTML but no price found")
    else:
        if debug:
            print(f"[get_prices] conventional fetch failed")

    # Selenium fallback for regular sites: if conventional fetch failed OR didn't find price
    if debug:
        print(f"[get_prices] trying Selenium fallback: {url}")

    html_text_selenium = selenium_fetch(url, timeout=20, wait_for_element=wait_element)

    if not html_text_selenium:
        if debug:
            print(f"[get_prices] Selenium fetch also failed")
        # Return whatever we got from conventional fetch (might be empty)
        return extract_prices_from_html(html_text, host, expected_ean=None, debug=debug) if html_text else {}

    # Extract prices from Selenium result
    result_selenium = extract_prices_from_html(html_text_selenium, host, expected_ean=None, debug=debug)
    if debug:
        has_price = result_selenium and (result_selenium.get("price") or result_selenium.get("price_regular") or result_selenium.get("price_member"))
        print(f"[get_prices] Selenium fetch {'succeeded with price' if has_price else 'got HTML but no price'}")

    return result_selenium


# --- URL variant adjustment (when URL embeds a different EAN/UPC) ---

def adjust_url_for_ean(url: str, ean: str) -> List[str]:
    """
    If the URL contains a 12/13-digit run (variant id / embedded EAN) that doesn't match
    the requested EAN/UPC, generate candidate URLs where that run is replaced by our EAN.
    Returns a list of candidate URLs (most likely first), including the original URL last.
    """
    if not url or not ean:
        return [url]
    e13 = re.sub(r"\D+", "", ean)
    if not e13:
        return [url]
    e12 = e13[1:] if len(e13) == 13 else e13

    candidates: List[str] = []

    # 1) Generic: replace the first 12/13-digit run in path or query
    def repl_first_run(u: str) -> str:
        def _repl(m: re.Match) -> str:
            digits = m.group(1)
            if digits == e13 or digits == e12:
                return digits  # already matching
            return e13 if len(digits) == 13 else e12
        return re.sub(r"(?<!\d)(\d{12,13})(?!\d)", _repl, u, count=1)

    u1 = repl_first_run(url)
    if u1 and u1 != url:
        candidates.append(u1)

    # 2) Special case: query param like v=Something-<digits>
    m = re.search(r"([?&]v=[^&#]*?)(\d{12,13})(?!\d)", url, re.I)
    if m:
        new_digits = e13 if len(m.group(2)) == 13 else e12
        u2 = url[:m.start(2)] + new_digits + url[m.end(2):]
        if u2 != url and u2 not in candidates:
            candidates.append(u2)

    # 2b) Obs Bygg: ensure explicit variant parameter `v=ObsBygg-<EAN13>`
    host = host_from_url(url)
    if host.endswith("obsbygg.no") and e13:
        # If URL already has a `v=` param but with a different digits-run, replace it.
        m2 = re.search(r"([?&]v=)([^&#]*)", url, re.I)
        desired = f"ObsBygg-{e13}"
        if m2:
            cur_v = m2.group(2)
            # If it doesn't already contain our exact EAN13, replace the digits-run.
            if e13 not in cur_v:
                u3 = url[:m2.start(2)] + desired + url[m2.end(2):]
                if u3 != url and u3 not in candidates:
                    candidates.insert(0, u3)  # highest priority
        else:
            sep = '&' if ('?' in url) else '?'
            u3 = f"{url}{sep}v={desired}"
            if u3 not in candidates:
                candidates.insert(0, u3)  # highest priority

    # 3) Return original last (deduped)
    if url not in candidates:
        candidates.append(url)

    return candidates


def get_prices_for_ean(url: str, expected_ean: str, debug: bool = False) -> dict:
    """
    Extract prices for a specific EAN from a URL.
    Uses the unified extraction logic with Selenium fallback.
    """
    host = host_from_url(url)

    # Try adjusted candidate URLs if the URL embeds a different variant id/EAN
    candidates = adjust_url_for_ean(url, expected_ean)

    def attempt(one_url: str) -> dict:
        local_host = host_from_url(one_url)

        # Determine wait element based on known sites
        wait_element = None
        if local_host.endswith("maxbo.no") or local_host.endswith("proff.maxbo.no"):
            wait_element = '.product-price__price, .product-price__amount, [data-testid*="price"]'

        # STRATEGY:
        # - For JS-heavy sites (Maxbo): Use Selenium FIRST (required)
        # - For other sites: Try conventional fetch first, then Selenium as fallback

        # For JS-heavy sites: Use Selenium FIRST (they require JavaScript)
        if is_js_heavy_site(local_host):
            if debug:
                print(f"[debug] JS-heavy site detected, using Selenium first: {one_url}")

            html_text_selenium = selenium_fetch(one_url, timeout=20, wait_for_element=wait_element)
            if html_text_selenium:
                result = extract_prices_from_html(html_text_selenium, local_host, expected_ean=expected_ean, debug=debug)
                if result and (result.get("price") or result.get("price_regular") or result.get("price_member")):
                    if debug:
                        print(f"[debug] Selenium fetch succeeded with price")
                    return result
                elif debug:
                    print(f"[debug] Selenium fetch got HTML but no price found")
            else:
                if debug:
                    print(f"[debug] Selenium fetch failed: {one_url}")

            # If Selenium failed, don't try conventional fetch (it won't work for JS-heavy sites)
            return {}

        # For regular sites: Try conventional fetch first
        if debug:
            print(f"[debug] trying conventional fetch: {one_url}")
        html_text = fetch(one_url)

        if not html_text:
            if debug:
                print(f"[debug] conventional fetch failed: {one_url}")
        else:
            # Try extracting prices from conventional fetch using unified logic
            result = extract_prices_from_html(html_text, local_host, expected_ean=expected_ean, debug=debug)
            if result and (result.get("price") or result.get("price_regular") or result.get("price_member")):
                if debug:
                    print(f"[debug] conventional fetch succeeded with price")
                return result
            elif debug:
                print(f"[debug] conventional fetch got HTML but no price found")

        # Selenium fallback for regular sites: if conventional fetch failed OR didn't find price
        if debug:
            print(f"[debug] trying Selenium fallback: {one_url}")

        html_text_selenium = selenium_fetch(one_url, timeout=20, wait_for_element=wait_element)

        if not html_text_selenium:
            if debug:
                print(f"[debug] Selenium fetch also failed: {one_url}")
            # Return whatever we got from conventional fetch (might be empty)
            return extract_prices_from_html(html_text, local_host, expected_ean=expected_ean, debug=debug) if html_text else {}

        # Try extracting from Selenium result using unified logic
        result_selenium = extract_prices_from_html(html_text_selenium, local_host, expected_ean=expected_ean, debug=debug)
        if debug:
            has_price = result_selenium and (result_selenium.get("price") or result_selenium.get("price_regular") or result_selenium.get("price_member"))
            print(f"[debug] Selenium fetch {'succeeded with price' if has_price else 'got HTML but no price'}")

        return result_selenium

    # Iterate through candidates and return the first successful extraction
    for cu in candidates:
        if debug:
            print(f"[debug] candidate URL: {cu}")
        res = attempt(cu)
        if res and (res.get("price") or res.get("price_regular") or res.get("price_member")):
            if debug:
                print("[debug] candidate succeeded")
            return res
        elif debug:
            print("[debug] candidate had no usable price")

    # If none succeeded, last attempt on the original URL (for completeness)
    return attempt(url)

def fetch(url: str, timeout: int = 10, max_bytes: int = 300_000) -> Optional[str]:
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True, stream=True)
        r.raise_for_status()
        content = r.content[:max_bytes]
        return content.decode(r.encoding or 'utf-8', errors='replace')
    except Exception:
        return None


def selenium_fetch(url: str, timeout: int = 15, wait_for_element: str = None) -> Optional[str]:
    """
    Fetch page content using Selenium WebDriver for JavaScript-heavy sites.
    Optimized for low-memory environments like Render (0.5GB RAM).
    """
    driver = None
    try:
        # Configure Chrome options for minimal memory usage
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--disable-extensions')
        chrome_options.add_argument('--disable-plugins')
        chrome_options.add_argument('--disable-images')
        chrome_options.add_argument('--memory-pressure-off')
        chrome_options.add_argument('--max_old_space_size=256')
        chrome_options.add_argument('--window-size=1024,768')
        chrome_options.add_argument('--disable-logging')
        chrome_options.add_argument('--disable-background-timer-throttling')
        chrome_options.add_argument('--disable-renderer-backgrounding')
        chrome_options.add_argument('--disable-features=TranslateUI')
        chrome_options.add_argument('--disable-ipc-flooding-protection')

        # Create driver with minimal resource usage
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        driver.set_page_load_timeout(timeout)

        # Load the page
        driver.get(url)

        # Wait for specific element if provided (e.g., price elements)
        if wait_for_element:
            try:
                WebDriverWait(driver, min(10, timeout)).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, wait_for_element))
                )
            except:
                pass  # Continue even if specific element not found
        else:
            # Default wait for page to load
            time.sleep(3)

        # Get page source
        html_content = driver.page_source
        return html_content

    except Exception as e:
        print(f"[selenium_fetch] Error fetching {url}: {e}")
        return None
    finally:
        if driver:
            try:
                driver.quit()
            except:
                pass


def get_price(url: str) -> Optional[str]:
    d = get_prices(url)
    return d.get("price") if d else None

# -------- UPCitemdb --------

def upcitemdb_lookup(ean: str) -> List[dict]:
    try:
        resp = requests.get(UPCITEMDB_TRIAL_URL, params={'upc': ean}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []
    items = data.get('items') or []
    if not items:
        return []
    offers = items[0].get('offers') or []
    out: List[dict] = []
    for o in offers:
        link = o.get('link') or o.get('url')
        price = o.get('price')
        currency = o.get('currency')
        if not link or price in (None, ''):
            continue
        price_str = str(price)
        if currency and not re.search(r"\b" + re.escape(currency) + r"\b|[€$£]", price_str, re.I):
            price_str = f"{currency} {price_str}"
        out.append({'link': link, 'price': price_str})
    return out

# -------- Google CSE (ported) --------
MARKET_CFG = {
    None: {'hl': None, 'gl': None},
    'NO': {'hl': 'nb', 'gl': 'no'},
    'SE': {'hl': 'sv', 'gl': 'se'},
    'DK': {'hl': 'da', 'gl': 'dk'},
    'DE': {'hl': 'de', 'gl': 'de'},
    'FR': {'hl': 'fr', 'gl': 'fr'},
    'UK': {'hl': 'en', 'gl': 'gb'},
    'US': {'hl': 'en', 'gl': 'us'},
    'CZ': {'hl': 'cs', 'gl': 'cz'},
    'PL': {'hl': 'pl', 'gl': 'pl'},
}

MARKET_DOMAINS = {
    'NO': ['elkjop.no','komplett.no','power.no','pricerunner.no','prisjakt.no','proshop.no','cdon.no','dustinhome.no'],
    'SE': ['elgiganten.se','netonnet.se','complett.se','proshop.se','pricerunner.se','prisjakt.nu','dustin.se','webhallen.com','cdon.se'],
    'DK': ['elgiganten.dk','proshop.dk','computersalg.dk','av-cables.dk','merlin.dk','pricerunner.dk'],
    'DE': ['amazon.de','otto.de','mediamarkt.de','saturn.de','idealo.de','geizhals.de'],
    'FR': ['amazon.fr','fnac.com','cdiscount.com','rueducommerce.fr','ldlc.com'],
    'UK': ['amazon.co.uk','currys.co.uk','argos.co.uk','johnlewis.com','very.co.uk'],
    'US': ['amazon.com','bestbuy.com','walmart.com','target.com','newegg.com'],
    'CZ': ['alza.cz','czc.cz','datart.cz','mall.cz'],
    'PL': ['allegro.pl','ceneo.pl','morele.net','x-kom.pl'],
}

def build_variants(ean: str, keywords: str = '', sites: str = '') -> List[str]:
    ean = re.sub(r"\D+", "", ean)
    if not ean:
        return []
    vars_: List[str] = [ean]
    if len(ean) == 13:
        vars_.append(ean[1:])  # 12-digit variant
    quoted = [f'"{v}"' for v in vars_]
    suffix_parts: List[str] = []
    if keywords:
        suffix_parts.append(keywords)
    if sites:
        domains = [d.strip() for d in re.split(r"[\|,\s]+", sites) if d.strip()]
        if domains:
            suffix_parts.append(" ".join([f"site:{d}" for d in domains]))
    suffix = (" " + " ".join(suffix_parts)) if suffix_parts else ""
    out: List[str] = []
    for base in list(dict.fromkeys(vars_ + quoted)):
        out.append((base + suffix).strip())
    return out


def google_cse_fetch(api_key: str, cx: str, query: str, hl: Optional[str], gl: Optional[str], num: int = 10, start: int = 1) -> Dict:
    params = {
        'key': api_key,
        'cx': cx,
        'q': query,
        'num': max(1, min(num, 10)),
        'start': max(1, start),
        'filter': '0',
        'safe': 'off',
    }
    if hl: params['hl'] = hl
    if gl: params['gl'] = gl
    url = 'https://www.googleapis.com/customsearch/v1'
    try:
        r = requests.get(url, params=params, timeout=12)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {'items': [], 'error': f'HTTP/JSON error: {e.__class__.__name__}'}

# -------- CLI --------

def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Scraper/lookup pre cenu: --url alebo --ean (CSE + UPCitemdb)')
    p.add_argument('--url', dest='urls', nargs='*', help='Produktové URL na scrapovanie')
    p.add_argument('--ean', dest='eans', nargs='*', help='EAN/UPC kódy na lookup (CSE + UPCitemdb)')
    p.add_argument('--limit', type=int, default=20, help='Limit výsledkov na jeden EAN (CSE/UPCitemdb)')
    p.add_argument('--per-domain', type=int, default=1, help='Max výsledkov z jednej domény (CSE)')
    p.add_argument('--market', choices=['NO','SE','DK','DE','FR','UK','US','CZ','PL'], default=None, help='Lokalizácia CSE (hl/gl)')
    p.add_argument('--sites', default='', help='Obmedz CSE na tieto domény (oddelené čiarkou/medzerou)')
    p.add_argument('--tld-no', action='store_true', help='Po vyhľadávaní ponechaj len domény končiace na .no')
    p.add_argument('--include-empty', action='store_true', help='Zobraz aj výsledky bez detegovanej ceny')
    return p.parse_args(argv)


def run_urls(urls: Iterable[str]) -> None:
    for url in urls:
        if not url:
            continue
        price = get_price(url)
        print(f"{url}\t{(price or 'nenájdené')}")


def run_eans(eans: Iterable[str], limit: int, per_domain: int, market: Optional[str], sites: str, tld_no: bool=False, include_empty: bool=False) -> None:
    for ean in eans:
        if not ean:
            continue
        any_printed = False
        # 1) UPCitemdb offers first
        offers = upcitemdb_lookup(ean)
        for o in offers[:max(1, limit)]:
            print(f"{o['link']}\t{o['price']}")
            any_printed = True

        # 2) Google CSE (requires keys)
        if not GOOGLE_API_KEY or not GOOGLE_CSE_CX:
            if not any_printed:
                print(f"{ean}\tno-offers")
            continue

        hl = MARKET_CFG.get(market, {}).get('hl')
        gl = MARKET_CFG.get(market, {}).get('gl')
        # CLI: neobmedzuj domény automaticky; použijeme celý web, pokiaľ používateľ nedá --sites
        default_sites = ''
        sites_arg = sites if sites else default_sites

        queries = build_variants(ean, keywords='', sites=sites_arg)
        seen_per_domain: Dict[str, int] = {}
        printed_links: set = set()
        printed_count = 0
        for q in queries:
            if printed_count >= limit:
                break
            for page in range(0, 2):  # up to 20 results/query
                start = 1 + page * 10
                data = google_cse_fetch(GOOGLE_API_KEY, GOOGLE_CSE_CX, q, hl, gl, num=10, start=start)
                items = data.get('items') or []
                if not items:
                    break
                for it in items:
                    link = it.get('link')
                    if not link:
                        continue
                    host = host_from_url(link)
                    if host in BLOCKLIST:
                        continue
                    if per_domain > 0 and seen_per_domain.get(host, 0) >= per_domain:
                        continue
                    if link in printed_links:
                        continue
                    if tld_no and not host.endswith('.no'):
                        continue

                    prices = get_prices_for_ean(link, ean, debug=include_empty)
                    price = prices.get('price') if prices else None
                    if price is None:
                        if include_empty:
                            print(f"{link}\tnenájdené (pozri [debug] log vyššie)")
                            printed_links.add(link)
                            seen_per_domain[host] = seen_per_domain.get(host, 0) + 1
                            printed_count += 1
                        continue
                    print(f"{link}\t{price}")
                    printed_links.add(link)
                    seen_per_domain[host] = seen_per_domain.get(host, 0) + 1
                    printed_count += 1
                    if printed_count >= limit:
                        break
                if printed_count >= limit:
                    break
        if not any_printed and printed_count == 0:
            print(f"{ean}\tno-results")


def main():
    args = parse_args(sys.argv[1:])
    if args.urls:
        run_urls(args.urls)
    if args.eans:
        run_eans(args.eans, args.limit, args.per_domain, args.market, args.sites, args.tld_no, args.include_empty)
    if not args.urls and not args.eans:
        print(
            'Použitie:\n'
            '  python test.py --url <URL1> [URL2 ...]\n'
            '  python test.py --ean <EAN1> [EAN2 ...] [--limit N] [--per-domain 1] [--market NO|SE] [--sites dom1,dom2]'
        )

if __name__ == '__main__':
    main()