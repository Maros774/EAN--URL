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

# === CONFIG === (taken from your PHP)
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', '')
GOOGLE_CSE_CX  = os.getenv('GOOGLE_CSE_CX',  '')

# UPCitemdb (trial)
UPCITEMDB_TRIAL_URL = 'https://api.upcitemdb.com/prod/trial/lookup'

HEADERS = {
    'User-Agent': 'PriceScraper/1.3 (+https://example.local)',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en,nb;q=0.9,sv;q=0.9,da;q=0.9,sk;q=0.8',
}

# --- Regex building blocks (allow normal/NBSP/thin spaces) ---
SP = r"[\s\u00A0\u202F]*"
NUM = r"(?:\d{1,3}(?:[\.,\s\u00A0\u202F]\d{3})*(?:[\.,]\d{2})?|\d+)"
CODES = r"(?:EUR|USD|GBP|CHF|PLN|CZK|HUF|RON|NOK|SEK|DKK|ISK|BGN|RSD|TRY)"
SYMBOLS_BEFORE = r"(?:€|£|\$)"
AFTER_TOKENS = rf"(?:€|{CODES}|Kč|CZK|zł|PLN|Ft|HUF|lei|RON|CHF|NOK|SEK|DKK|ISK|BGN|RSD|TRY|kr)"

RE_BEFORE = re.compile(rf"(?:{SYMBOLS_BEFORE}|{CODES}){SP}{NUM}", re.I | re.U)
RE_AFTER  = re.compile(rf"{NUM}{SP}{AFTER_TOKENS}", re.I | re.U)

RE_JSONLD_SCRIPT = re.compile(r'<script[^>]+type\s*=\s*["\']application/ld\+json["\'][^>]*>(.*?)</script>', re.I | re.S)

RE_META_PRICE = [
    re.compile(r'<meta[^>]+itemprop="price"[^>]+content="([^"]+)"', re.I),
    re.compile(r'<meta[^>]+property="product:price:amount"[^>]+content="([^"]+)"', re.I),
    re.compile(r'<meta[^>]+property="og:price:amount"[^>]+content="([^"]+)"', re.I),
]

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
    text = html.unescape(text)
    text = re.sub(r"[\s\u00A0\u202F]+", " ", text).strip()
    return text


def host_from_url(url: str) -> str:
    try:
        from urllib.parse import urlparse
        h = urlparse(url).netloc.lower()
    except Exception:
        return url
    return re.sub(r'^www\.', '', h)


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
        offers: List[dict] = []
        for n in expanded:
            if not isinstance(n, dict):
                continue
            o = n.get('offers')
            if o:
                offers.extend(o if isinstance(o, list) else [o])
        cands: List[str] = []
        for o in offers:
            if not isinstance(o, dict):
                continue
            price = o.get('price') or o.get('lowPrice')
            cur = o.get('priceCurrency')
            if price is None:
                continue
            raw = f"{cur} {price}" if cur else str(price)
            cands.append(raw)
        if cands:
            best = _best_from_candidates(cands)
            if best and extract_prices_from_text(best):
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
        cands.extend(extract_prices_from_text(b))
    if not cands:
        return None
    return _best_from_candidates(cands)


def from_visible_text(html_text: str) -> Optional[str]:
    vis = _strip_tags(html_text)
    cands = extract_prices_from_text(vis)
    if not cands:
        return None
    return _best_from_candidates(cands)

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

    # Fallback chain using existing detectors
    base = None
    if candidates:
        base = _best_from_candidates([raw for _ctx, raw in candidates])
    if not base:
        base = from_jsonld(html_text) or from_meta(html_text) or from_price_blocks(html_text) or from_visible_text(html_text)

    return {"price": base, "price_regular": price_regular, "price_member": price_member}


def get_prices(url: str) -> dict:
    html_text = fetch(url)
    if not html_text:
        return {}
    return detect_prices(html_text)


def fetch(url: str, timeout: int = 10, max_bytes: int = 300_000) -> Optional[str]:
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True, stream=True)
        r.raise_for_status()
        content = r.content[:max_bytes]
        return content.decode(r.encoding or 'utf-8', errors='replace')
    except Exception:
        return None


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
    'UK': {'hl': 'en', 'gl': 'uk'},
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
    return p.parse_args(argv)


def run_urls(urls: Iterable[str]) -> None:
    for url in urls:
        if not url:
            continue
        price = get_price(url)
        print(f"{url}\t{(price or 'nenájdené')}")


def run_eans(eans: Iterable[str], limit: int, per_domain: int, market: Optional[str], sites: str) -> None:
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
        default_sites = ' '.join(MARKET_DOMAINS.get(market, [])) if market else ''
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
                    if per_domain > 0 and seen_per_domain.get(host, 0) >= per_domain:
                        continue
                    if link in printed_links:
                        continue
                    price = get_price(link)
                    print(f"{link}\t{(price or 'nenájdené')}")
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
        run_eans(args.eans, args.limit, args.per_domain, args.market, args.sites)
    if not args.urls and not args.eans:
        print(
            'Použitie:\n'
            '  python test.py --url <URL1> [URL2 ...]\n'
            '  python test.py --ean <EAN1> [EAN2 ...] [--limit N] [--per-domain 1] [--market NO|SE] [--sites dom1,dom2]'
        )

if __name__ == '__main__':
    main()