import os, sys, json, time, importlib, types
from datetime import timedelta
from collections import defaultdict

BASE_MOD_NAME = "config.rmsc03_base" 
_base = importlib.import_module(BASE_MOD_NAME)

# -------------------- markers --------------------
_cfg_path = os.environ.get("MM_MVP_CFG", "")
try:
    _root = os.path.dirname(_cfg_path) if _cfg_path else os.getcwd()
    with open(os.path.join(_root, "wrapper_imported.txt"), "w", encoding="utf-8") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S") + " import reached\n")
    print("[rmsc03_mm] module import reached", flush=True)
except Exception as _e:
    print(f"[rmsc03_mm] import marker failed: {_e}", file=sys.stderr, flush=True)

def _write_seen(summary: dict):
    try:
        root = os.path.dirname(_cfg_path) if _cfg_path else os.getcwd()
        with open(os.path.join(root, "wrapper_seen.txt"), "w", encoding="utf-8") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")
    except Exception:
        pass

# -------------------- fallback parse_arguments --------------------
def _fallback_parse_arguments():
    import argparse
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("-t","--ticker", dest="ticker", default="ABM")
    p.add_argument("-d","--historical-date", dest="historical_date", default="20200603")
    p.add_argument("-s","--seed", dest="seed", type=int, default=1)
    p.add_argument("-l","--log-label", dest="log_label", default="rmsc03_mm")
    args,_ = p.parse_known_args()
    return args

parse_arguments = getattr(_base, "parse_arguments", _fallback_parse_arguments)

# -------------------- helpers --------------------
def _agent_sig(a):
    nm = getattr(a, "name", None)
    tp = getattr(a, "type", None)
    cls = a.__class__.__name__ if hasattr(a, "__class__") else None
    return nm, tp, cls

def _find_exchange(agents):
    for a in agents or []:
        cls = a.__class__.__name__
        if "ExchangeAgent" in cls or getattr(a, "is_exchange", False):
            return a
    return None

def _find_mm_agent(agents):
    for a in agents or []:
        nm, tp, cls = _agent_sig(a)
        hay = " ".join(str(x).upper() for x in (nm, tp, cls) if x)
        if ("MARKET" in hay and "MAKER" in hay) or ("POV" in hay):
            return a
    return None

ATTR_CANDIDATES = {
    "spread_bps":        ["spread_bps","spreadBps","spread","half_spread_bps"],
    "quote_size":        ["quote_size","order_size","size","min_order_size"],
    "inv_aversion":      ["inv_aversion","inventory_risk_aversion","risk_aversion","kappa"],
    "skew_gain":         ["skew_gain","inventory_skew_gain","alpha","skew"],
    "cancel_thresh_bps": ["cancel_thresh_bps","cancel_threshold_bps","cancel_thresh","cancel_threshold"],
    "max_inventory":     ["max_inventory","inventory_limit","max_inv"],
}

def _apply_mm_params(mm_agent, mm_params):
    applied = []
    if mm_agent is None or not mm_params:
        return applied
    for key, cands in ATTR_CANDIDATES.items():
        if key not in mm_params: 
            continue
        val = mm_params[key]
        for attr in cands:
            if hasattr(mm_agent, attr):
                try:
                    setattr(mm_agent, attr, val)
                    applied.append((attr, val))
                    break
                except Exception:
                    pass
    return applied

def _thin_agents_list(agents, limits):
    if not isinstance(agents, list) or not limits:
        return agents, 0
    keep = set()
    for a in agents:
        nm, tp, cls = _agent_sig(a)
        hay = f"{nm} {tp} {cls}".upper()
        if "EXCHANGE" in hay or ("MARKET" in hay and "MAKER" in hay) or "POV" in hay:
            keep.add(id(a))
    counts = defaultdict(int)
    out = []
    dropped = 0
    for a in agents:
        if id(a) in keep:
            out.append(a); 
            continue
        cls = a.__class__.__name__
        lim = limits.get(cls)
        if lim is None:
            out.append(a); 
            continue
        if counts[cls] < max(0, int(lim)):
            counts[cls] += 1
            out.append(a)
        else:
            dropped += 1
    return out, dropped

def _disable_booklog_exchange(exchange):
    """Flip any likely logging flags on the exchange agent."""
    if not exchange: return []
    changed = []
    forced = {
        "book_freq": 0,
        "log_orders": False,
        "book_log": False,
        "write_to_disk": False,
        "log_depth": 0,
        "log_order_book": False,
        "orderbook_logging": False,
        "order_book_logging": False,
        "log_period": 0,
        "wide_book": False,
        "log_to_file": False,
    }
    for name, val in forced.items():
        if hasattr(exchange, name):
            try:
                setattr(exchange, name, val)
                changed.append(f"exchange.{name}")
            except Exception:
                pass
    # generic sweep
    for name in dir(exchange):
        lname = name.lower()
        if any(tok in lname for tok in ("book","orderbook","log","depth","lob","archive","dump","csv")):
            try:
                cur = getattr(exchange, name)
                if isinstance(cur, bool) and cur:
                    setattr(exchange, name, False); changed.append(f"exchange.{name}")
                elif isinstance(cur, int) and cur > 0:
                    setattr(exchange, name, 0); changed.append(f"exchange.{name}")
            except Exception:
                pass
    return sorted(set(changed))

def _disable_booklog_module(base_mod):
    """Flip module-level toggles and monkey-patch likely archivers/loggers to no-ops."""
    changed = []

    # flip obvious module-level booleans/ints
    for name in dir(base_mod):
        lname = name.lower()
        if any(tok in lname for tok in ("book","orderbook","log","archive","dump","depth","csv")):
            try:
                cur = getattr(base_mod, name)
                if isinstance(cur, bool) and cur:
                    setattr(base_mod, name, False); changed.append(f"base.{name}")
                elif isinstance(cur, int) and cur > 0:
                    setattr(base_mod, name, 0); changed.append(f"base.{name}")
            except Exception:
                pass

    # monkey-patch functions that look like archivers
    def _noop(*a, **k): 
        return None

    patch_name_tokens = (
        "orderbook", "order_book", "booklog", "book_log", "log_order",
        "archive", "dump", "write_book", "write_orderbook", "write_order_book",
        "orderbook_to_csv", "log_orderbook_csv", "to_csv", "process_orderbook_log",
    )
    for name in dir(base_mod):
        obj = getattr(base_mod, name, None)
        if isinstance(obj, (types.FunctionType, types.MethodType)):
            lname = name.lower()
            if any(tok in lname for tok in patch_name_tokens):
                try:
                    setattr(base_mod, name, _noop)
                    changed.append(f"patched.{name}()")
                except Exception:
                    pass

    # env hint (for any code that checks it)
    os.environ["MM_DISABLE_BOOKLOG"] = "1"
    os.environ["ABIDES_DISABLE_BOOKLOG"] = "1"
    return sorted(set(changed))

def _load_payload():
    if not _cfg_path:
        return {}
    try:
        with open(_cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[rmsc03_mm] WARNING: failed to read MM_MVP_CFG: {e}", flush=True)
        return {}

def _apply_module_level():
    payload = _load_payload()
    if not payload:
        return None

    summary = {
        "session_seconds": int(payload.get("session_seconds", 0)),
        "session_minutes": int(payload.get("session_minutes", 30)),
        "thin_agents": payload.get("thin_agents") or {},
        "thinned_dropped": 0,
        "mm_attrs_applied": [],
        "booklog_changes": [],
        "mode": "module_level",
    }

    marketOpen  = getattr(_base, "marketOpen",  None)
    marketClose = getattr(_base, "marketClose", None)
    agents      = getattr(_base, "agents",      None)

    # shorten session
    try:
        secs = int(payload.get("session_seconds", 0))
        if marketOpen and (marketClose is not None):
            if secs > 0:
                marketClose = marketOpen + timedelta(seconds=secs)
            else:
                minutes = int(payload.get("session_minutes", 30))
                marketClose = marketOpen + timedelta(minutes=minutes)
            globals()["marketOpen"]  = marketOpen
            globals()["marketClose"] = marketClose
    except Exception:
        pass

    # thin agents
    if isinstance(agents, list):
        thin = payload.get("thin_agents") or {}
        if isinstance(thin, dict):
            thin = {str(k): int(v) for k, v in thin.items()}
        agents2, dropped = _thin_agents_list(agents, thin)
        globals()["agents"] = agents2
        summary["thinned_dropped"] = dropped
        mm_applied = _apply_mm_params(_find_mm_agent(agents2), payload.get("mm_params", {}) or {})
        summary["mm_attrs_applied"] = mm_applied
        summary["booklog_changes"] += _disable_booklog_exchange(_find_exchange(agents2))

    # module-level booklog off + patch archivers
    if payload.get("disable_booklog", False):
        summary["booklog_changes"] += _disable_booklog_module(_base)

    print(f"[rmsc03_mm] applied (module-level): {summary}", flush=True)
    _write_seen(summary)
    return summary

_module_apply_summary = _apply_module_level()

def _apply_to_cfg_dict(cfg: dict):
    """Mutate a cfg dict (builder style) using the same rules."""
    payload = _load_payload()
    if not payload or not isinstance(cfg, dict):
        return cfg, None
    summary = {
        "session_seconds": int(payload.get("session_seconds", 0)),
        "session_minutes": int(payload.get("session_minutes", 30)),
        "thin_agents": payload.get("thin_agents") or {},
        "thinned_dropped": 0,
        "mm_attrs_applied": [],
        "booklog_changes": [],
        "mode": "builder",
    }

    if "marketOpen" in cfg and "marketClose" in cfg:
        try:
            secs = int(payload.get("session_seconds", 0))
            if secs > 0:
                cfg["marketClose"] = cfg["marketOpen"] + timedelta(seconds=secs)
            else:
                cfg["marketClose"] = cfg["marketOpen"] + timedelta(minutes=summary["session_minutes"])
        except Exception:
            pass

    if "agents" in cfg and isinstance(cfg["agents"], list):
        thin = payload.get("thin_agents") or {}
        if isinstance(thin, dict):
            thin = {str(k): int(v) for k, v in thin.items()}
        cfg["agents"], dropped = _thin_agents_list(cfg["agents"], thin)
        summary["thinned_dropped"] = dropped
        summary["mm_attrs_applied"] = _apply_mm_params(_find_mm_agent(cfg["agents"]), payload.get("mm_params", {}) or {})
        summary["booklog_changes"] += _disable_booklog_exchange(_find_exchange(cfg["agents"]))

    if payload.get("disable_booklog", False):
        summary["booklog_changes"] += _disable_booklog_module(_base)

    print(f"[rmsc03_mm] applied (builder): {summary}", flush=True)
    _write_seen(summary)
    return cfg, summary

def _base_build(args):
    for name in ("build_config","build_environment","build","build_market_environment",
                 "rmsc03","make_config","get_config","generate_config"):
        fn = getattr(_base, name, None)
        if callable(fn):
            print(f"[rmsc03_mm] using {BASE_MOD_NAME}.{name}", flush=True)
            cfg = fn(args)
            if isinstance(cfg, dict):
                cfg, _ = _apply_to_cfg_dict(cfg)
            return cfg
    return None  

def build_config(args):             return _base_build(args)
def build_environment(args):        return _base_build(args)
def build(args):                    return _base_build(args)
def build_market_environment(args): return _base_build(args)
def rmsc03(args):                   return _base_build(args)
def make_config(args):              return _base_build(args)
def get_config(args):               return _base_build(args)
def generate_config(args):          return _base_build(args)

# -------------------- delegation for unknown symbols --------------------
def __getattr__(name):
    if name in globals():
        return globals()[name]
    return getattr(_base, name)
