from flask import Flask, render_template, Response, request, jsonify, make_response
from simulation_engine import SimulationEngine
import atexit
import base64
import cv2
import difflib
import json
import numpy as np
import os
import re
import subprocess
import sys
import time
import datetime
import math
import html
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_background_log_streams = []


def _load_env_file(path):
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except OSError:
        pass


for _env_name in (".env", ".env.local"):
    _load_env_file(os.path.join(PROJECT_ROOT, _env_name))


def _ensure_windows_background_streams():
    if os.name != "nt":
        return
    stdout_missing = sys.stdout is None
    stderr_missing = sys.stderr is None
    if not stdout_missing and not stderr_missing:
        return
    if stdout_missing:
        stdout_handle = open(os.path.join(PROJECT_ROOT, "app_stdout.log"), "a", encoding="utf-8", buffering=1)
        sys.stdout = stdout_handle
        _background_log_streams.append(stdout_handle)
    if stderr_missing:
        stderr_handle = open(os.path.join(PROJECT_ROOT, "app_stderr.log"), "a", encoding="utf-8", buffering=1)
        sys.stderr = stderr_handle
        _background_log_streams.append(stderr_handle)


_ensure_windows_background_streams()

app = Flask(__name__)
ASSISTANT_REMOTE_TIMEOUT_SEC = float(os.getenv("ASSISTANT_REMOTE_TIMEOUT_SEC", "12"))
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
SERVER_LOCK_PATH = os.path.join(PROJECT_ROOT, ".dashboard_server.lock")
TABLE_Z = 0.7
MAIN_ROBOT_BASE = [0.02, -0.02, TABLE_Z]
DOUBLE_LEFT_BASE = list(MAIN_ROBOT_BASE)
DOUBLE_RIGHT_BASE = list(MAIN_ROBOT_BASE)
DOUBLE_LEFT_OFFSET = [8.0, 0.0, 0.0]
DOUBLE_RIGHT_OFFSET = [16.0, 0.0, 0.0]
_server_lock_handle = None
SIM_GUI_SETTING = (os.getenv("SIM_GUI") or "").strip().lower()
# Keep the browser-embedded dashboard as the default experience.
# Separate PyBullet GUI should open only when explicitly requested.
MAIN_SIM_GUI_ENABLED = SIM_GUI_SETTING in {"1", "true", "yes", "on"}


def acquire_single_instance_lock():
    global _server_lock_handle
    lock_file = open(SERVER_LOCK_PATH, "a+", encoding="utf-8")
    try:
        lock_file.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write(str(os.getpid()))
        lock_file.flush()
        _server_lock_handle = lock_file
        atexit.register(release_single_instance_lock)
        return True
    except OSError:
        lock_file.close()
        return False


def release_single_instance_lock():
    global _server_lock_handle
    if _server_lock_handle is None:
        return
    try:
        _server_lock_handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(_server_lock_handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(_server_lock_handle.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass
    try:
        _server_lock_handle.close()
    finally:
        _server_lock_handle = None

sim = SimulationEngine(robot_base=MAIN_ROBOT_BASE,
                       robot_color=[0.92, 0.94, 0.99, 1.0],
                       enable_gui=MAIN_SIM_GUI_ENABLED)
sim_double_a = SimulationEngine(world_offset=DOUBLE_LEFT_OFFSET,
                                robot_base=DOUBLE_LEFT_BASE,
                                render_profile="full",
                                robot_color=[0.90, 0.90, 0.95, 1.0],
                                enable_gui=False)
sim_double_b = SimulationEngine(world_offset=DOUBLE_RIGHT_OFFSET,
                                robot_base=DOUBLE_RIGHT_BASE,
                                render_profile="full",
                                robot_color=[0.25, 0.65, 1.0, 1.0],
                                enable_gui=False)

@app.after_request
def add_no_cache(response):
    if 'text/html' in response.content_type:
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma']        = 'no-cache'
        response.headers['Expires']       = '0'
    return response

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/color_sort')
def color_sort():
    return render_template('color_sort.html')

@app.route('/congruent_surface')
def congruent_surface():
    return render_template('congruent_html.html')

@app.route('/double')
def double_robot():
    sim_double_a.send_command("RESET")
    sim_double_b.send_command("RESET")
    time.sleep(0.35)
    return render_template('double_robot.html')

def gen_frames_for(sim_ref):
    while True:
        frame = sim_ref.get_frame()
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.05)


def build_status_payload(sim_ref):
    summary = sim_ref.get_runtime_summary()
    conj = sim_ref.get_conjugate_data()
    grasp = sim_ref.get_grasp_quality_data()
    return {
        "state": sim_ref.get_status(),
        "task_stats": sim_ref.get_task_progress(),
        "summary": summary,
        "conjugate": conj,
        "grasp_quality": grasp,
        "active_task": summary.get("active_task"),
        "active_mode": summary.get("mode"),
        "focused_surface": summary.get("focused_surface"),
        "current_surface": summary.get("current_surface"),
        "current_color": summary.get("current_color"),
        "target_zone": summary.get("target_zone"),
        "current_shape": summary.get("current_shape"),
        "insertion_force_peak": summary.get("insertion_force_peak"),
    }


def build_double_dashboard_payload(sim_ref):
    payload = build_status_payload(sim_ref)
    payload.update({
        "vision": sim_ref.get_vision_data(),
        "slip": sim_ref.get_slip_data(),
        "force_history": sim_ref.get_force_history(),
        "color_sort": sim_ref.get_color_sort_status(),
        "telemetry": sim_ref.get_full_telemetry(),
    })
    return payload

def gen_frames():
    return gen_frames_for(sim)


def get_groq_model():
    model = (os.getenv("GROQ_MODEL") or DEFAULT_GROQ_MODEL).strip()
    return model or DEFAULT_GROQ_MODEL


def get_assistant_provider_catalog():
    groq_key = (os.getenv("GROQ_API_KEY") or "").strip()
    return {
        "groq": {
            "label": "ARIA AI (Groq Free)",
            "configured": bool(groq_key),
            "mode": "remote",
            "free": True,
            "notes": (
                "ARIA AI is connected via Groq Free Tier - Llama 3.3 70B. No cost, high quality."
                if groq_key else
                "Groq API key not found. Add GROQ_API_KEY in .env to enable AI assistant."
            ),
        },
    }


ASSISTANT_HINGLISH_MARKERS = {
    "acha", "achha", "abhi", "agar", "apna", "apne", "aap", "bata", "batao",
    "bolo", "chahta", "chahiye", "dikha", "dikhao", "ekdum", "gir", "giraya",
    "hai", "hain", "ho", "hoga", "hole", "insert", "jaws", "ka", "kai",
    "kaise", "karo", "karna", "karni", "kya", "kyu", "kyun", "mai", "main",
    "matlab", "mujhe", "nahi", "normal", "or", "peg", "pegs", "puch", "pucho",
    "pura", "robot", "sahi", "samjha", "samjhao", "sirf", "tight", "toh",
    "uska", "user", "wala", "wali",
}

SIMULATION_QUERY_MARKERS = {
    "align", "assistant", "cell", "cells", "color sort", "conveyor", "dashboard",
    "double robot", "drift", "drop", "feed", "force", "grasp", "gripper",
    "hole", "insert", "insertion", "jaw", "left cell", "peg", "physics",
    "pose", "queue", "rack", "regrasp", "right cell", "robot", "simulation",
    "slip", "sort", "surface", "task", "telemetry", "torque", "vision", "workbench",
}

LIVE_SIM_INTENT_MARKERS = {
    "current simulation", "dashboard status", "is running", "kar raha",
    "kya chal raha", "kya ho raha", "kya kar raha", "live status",
    "robot doing", "scene explain", "simulation status", "telemetry",
    "this simulation", "this workbench", "what is happening",
}

WEB_QUERY_MARKERS = {
    "aaj", "abhi internet", "browse", "google", "internet", "internet se",
    "latest", "news", "new update", "price", "recent", "search", "today",
    "trending", "updated", "weather", "web",
}


def _strip_html(text):
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", text or "", flags=re.I | re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _decode_duckduckgo_url(url):
    url = html.unescape(url or "").strip()
    if not url:
        return ""
    if url.startswith("//"):
        url = "https:" + url
    parsed = urllib.parse.urlparse(url)
    qs = urllib.parse.parse_qs(parsed.query)
    if "uddg" in qs and qs["uddg"]:
        return qs["uddg"][0]
    if url.startswith("/"):
        return urllib.parse.urljoin("https://duckduckgo.com", url)
    return url


def _extract_duckduckgo_results(raw_html, limit=5):
    results = []
    blocks = re.findall(
        r'<div[^>]+class="[^"]*result[^"]*"[^>]*>(.*?)</div>\s*</div>',
        raw_html or "",
        flags=re.I | re.S,
    )
    if not blocks:
        blocks = re.findall(
            r'<a[^>]+class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>(.{0,900})',
            raw_html or "",
            flags=re.I | re.S,
        )
        for href, title_html, tail in blocks:
            snippet_match = re.search(
                r'class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</',
                tail,
                flags=re.I | re.S,
            )
            title = _strip_html(title_html)
            snippet = _strip_html(snippet_match.group(1) if snippet_match else "")
            url = _decode_duckduckgo_url(href)
            if title and url:
                results.append({"title": title[:180], "url": url[:600], "snippet": snippet[:500]})
            if len(results) >= limit:
                return results
        return results

    for block in blocks:
        link_match = re.search(
            r'<a[^>]+class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
            block,
            flags=re.I | re.S,
        )
        if not link_match:
            continue
        snippet_match = re.search(
            r'class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</',
            block,
            flags=re.I | re.S,
        )
        title = _strip_html(link_match.group(2))
        snippet = _strip_html(snippet_match.group(1) if snippet_match else "")
        url = _decode_duckduckgo_url(link_match.group(1))
        if title and url and not any(item["url"] == url for item in results):
            results.append({"title": title[:180], "url": url[:600], "snippet": snippet[:500]})
        if len(results) >= limit:
            break
    return results


def _extract_bing_results(raw_html, limit=5):
    results = []
    parts = re.split(r'<li[^>]+class="[^"]*b_algo[^"]*"[^>]*>', raw_html or "", flags=re.I | re.S)
    for segment in parts[1:]:
        block = re.split(r'<li[^>]+class="[^"]*b_algo[^"]*"[^>]*>|</ol>', segment, maxsplit=1, flags=re.I | re.S)[0]
        link_match = re.search(r'<h2[^>]*>.*?<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', block, flags=re.I | re.S)
        if not link_match:
            continue
        snippet_match = re.search(r'<p[^>]*>(.*?)</p>', block, flags=re.I | re.S)
        title = _strip_html(link_match.group(2))
        url = html.unescape(link_match.group(1)).strip()
        snippet = _strip_html(snippet_match.group(1) if snippet_match else "")
        if title and url and not any(item["url"] == url for item in results):
            results.append({"title": title[:180], "url": url[:600], "snippet": snippet[:500]})
        if len(results) >= limit:
            break
    return results


def _fetch_search_html(url):
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=8) as response:
        return response.read().decode("utf-8", errors="ignore")


def _search_bing_rss(clean_query, limit=5, news=False):
    base_url = "https://www.bing.com/news/search?" if news else "https://www.bing.com/search?"
    url = base_url + urllib.parse.urlencode({"q": clean_query, "format": "rss"})
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}, method="GET")
    with urllib.request.urlopen(req, timeout=8) as response:
        raw = response.read().decode("utf-8", errors="ignore")
    root = ET.fromstring(raw)
    results = []
    for item in root.findall("./channel/item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        description = _strip_html(item.findtext("description") or "")
        if title and link:
            results.append({"title": title[:180], "url": link[:600], "snippet": description[:500]})
        if len(results) >= limit:
            break
    return results


def _search_web_for_assistant(query, limit=5):
    clean_query = re.sub(r"\s+", " ", (query or "").strip())
    if not clean_query:
        return {"query": "", "results": [], "error": "empty_query"}
    errors = []
    wants_news = any(word in clean_query.lower() for word in ("latest", "news", "recent", "today", "aaj"))
    try:
        results = _search_bing_rss(clean_query, limit, news=wants_news)
        if results:
            return {"query": clean_query, "source": "bing_news_rss" if wants_news else "bing_rss", "results": results, "error": ""}
    except Exception as exc:
        errors.append("bing_rss: " + str(exc)[:160])
    try:
        ddg_url = "https://duckduckgo.com/html/?" + urllib.parse.urlencode({"q": clean_query})
        raw = _fetch_search_html(ddg_url)
        results = _extract_duckduckgo_results(raw, limit)
        if results:
            return {"query": clean_query, "source": "duckduckgo", "results": results, "error": ""}
    except Exception as exc:
        errors.append("duckduckgo: " + str(exc)[:160])
    try:
        bing_url = "https://www.bing.com/search?" + urllib.parse.urlencode({"q": clean_query})
        raw = _fetch_search_html(bing_url)
        results = _extract_bing_results(raw, limit)
        if results:
            return {"query": clean_query, "source": "bing", "results": results, "error": ""}
    except Exception as exc:
        errors.append("bing: " + str(exc)[:160])
    return {"query": clean_query, "results": [], "error": "; ".join(errors)[:240]}


def _assistant_search_queries(message):
    base = re.sub(r"\s+", " ", (message or "").strip())
    if not base:
        return []
    cleaned = re.sub(
        r"\b(internet|internet se|web|search|google|batao|bata|please|pls|karo|kar|se)\b",
        " ",
        base,
        flags=re.I,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if cleaned and cleaned.lower() != base.lower():
        return [cleaned, base]
    queries = [base]
    return queries


def _search_web_with_retries(message, limit=5):
    errors = []
    for query in _assistant_search_queries(message):
        result = _search_web_for_assistant(query, limit)
        if result.get("results"):
            return result
        if result.get("error"):
            errors.append(f"{query}: {result['error']}")
    return {"query": message, "results": [], "error": "; ".join(errors)[:240]}


def _message_needs_web(message):
    lower = (message or "").strip().lower()
    if not lower:
        return False
    if any(marker in lower for marker in WEB_QUERY_MARKERS):
        return True
    if re.search(r"\b(20[2-9][0-9]|19[8-9][0-9])\b", lower) and any(
        word in lower for word in ("latest", "current", "today", "now", "news", "update")
    ):
        return True
    if re.search(r"\b(who|which|what)\s+(is|are)\s+(the\s+)?(current|latest)\b", lower):
        return True
    if re.search(r"\b(who|which)\s+(is|are)\b", lower) and any(
        title in lower for title in ("ceo", "president", "prime minister", "pm ", "chief minister")
    ):
        return True
    if "live" in lower and any(word in lower for word in ("score", "price", "weather", "news", "result")):
        return True
    if re.search(r"\b(who|what|when|where|why|how|which)\b", lower):
        return True
    if re.search(r"\b(explain|define|meaning|compare|best|top|tell me|give me)\b", lower):
        return True
    return False


def _message_explicitly_requests_web(message):
    lower = (message or "").strip().lower()
    if not lower:
        return False
    if any(marker in lower for marker in WEB_QUERY_MARKERS):
        return True
    if re.search(r"\b(current|latest|today|now|recent|updated)\b", lower):
        return True
    if re.search(r"\b(who|which)\s+(is|are)\b", lower) and any(
        title in lower for title in ("ceo", "president", "prime minister", "pm ", "chief minister")
    ):
        return True
    return False


def _build_web_fallback(message, style, web_context):
    results = (web_context or {}).get("results") or []
    if not results:
        return _assistant_text(
            style,
            "I tried to use internet search for this, but no web results came back. Please try again or ask with a more specific keyword.",
            "Maine iske liye internet search try kiya, lekin web results nahi mile. Thoda specific keyword ke saath dobara poochho.",
        )
    lines = [_assistant_text(
        style,
        "I found these web results. Please open the sources for the latest exact details:",
        "Internet search se ye results mile. Latest exact details ke liye sources open kar lena:",
    )]
    for idx, item in enumerate(results[:5], 1):
        snippet = item.get("snippet") or "No short preview available."
        lines.append(f"{idx}. {item.get('title') or 'Result'}")
        lines.append(f"   {snippet}")
        lines.append(f"   {item.get('url')}")
    return "\n".join(lines)


def _assistant_reply_style(message, history=None):
    chunks = []
    for item in history or []:
        if isinstance(item, dict) and (item.get("role") or "").strip().lower() == "user":
            text = (item.get("content") or "").strip()
            if text:
                chunks.append(text)
    if message:
        chunks.append(message)
    text = " ".join(chunks[-4:]).strip()
    lower = text.lower()
    if re.search(r"[\u0900-\u097f]", text):
        return "hinglish"
    if "hinglish" in lower:
        return "hinglish"
    if "hindi" in lower:
        return "hinglish"
    if "reply in english" in lower or "answer in english" in lower:
        return "english"
    return "english"


def _assistant_text(style, english, hinglish):
    return hinglish if style == "hinglish" else english


def _assistant_local_now():
    return datetime.datetime.now().astimezone()


def _assistant_day_period(now=None):
    hour = (now or _assistant_local_now()).hour
    if 5 <= hour < 12:
        return "morning"
    if 12 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 21:
        return "evening"
    return "night"


def _assistant_time_text(now=None):
    now = now or _assistant_local_now()
    zone = now.tzname() or "local time"
    return {
        "time": now.strftime("%I:%M %p").lstrip("0"),
        "date": now.strftime("%A, %B %d, %Y"),
        "zone": zone,
        "period": _assistant_day_period(now),
        "iso": now.isoformat(timespec="seconds"),
    }


def _detect_wish(message):
    lower = (message or "").strip().lower()
    words = re.findall(r"[a-z]+", lower)
    if not words:
        return None

    if len(words) <= 3 and words[0] in {"hi", "hello", "hey", "hii", "hiii"}:
        return {"period": None, "canonical": "Hello", "typed": words[0], "typo": words[0] not in {"hi", "hello", "hey"}}

    if len(words) <= 3 and words[0] in {"gm", "gdmrng"}:
        return {"period": "morning", "canonical": "Good morning", "typed": words[0], "typo": True}
    if len(words) <= 3 and words[0] in {"gn", "gdnyt", "gdnite"}:
        return {"period": "night", "canonical": "Good night", "typed": words[0], "typo": True}

    periods = {"morning", "afternoon", "evening", "night"}
    for index, word in enumerate(words[:-1]):
        if word not in {"good", "gud", "gd"}:
            continue
        typed_period = words[index + 1]
        match = difflib.get_close_matches(typed_period, periods, n=1, cutoff=0.68)
        if not match:
            continue
        period = match[0]
        typed = f"{word} {typed_period}"
        canonical = "Good " + period
        return {
            "period": period,
            "canonical": canonical,
            "typed": typed,
            "typo": typed != canonical.lower(),
        }

    if len(words) <= 3:
        match = difflib.get_close_matches(words[0], periods, n=1, cutoff=0.82)
        if match:
            period = match[0]
            return {"period": period, "canonical": "Good " + period, "typed": words[0], "typo": words[0] != period}
    return None


def _message_asks_time(message):
    lower = (message or "").strip().lower()
    if not lower:
        return False
    if any(phrase in lower for phrase in ("time complexity", "runtime complexity", "compile time", "run time")):
        return False
    if re.search(r"\b(what|current|tell|show|give)\b.{0,30}\btime\b", lower):
        return True
    if re.search(r"\btime\b.{0,30}\b(now|right now|please|pls|bata|batao|kya|kitna)\b", lower):
        return True
    if re.search(r"\b(kitna|kitne|kya)\s+baj", lower):
        return True
    if any(word in lower for word in ("samay", "waqt")):
        return True
    return False


def _message_is_mostly_wish_or_time(message, wish, asks_time):
    lower = (message or "").strip().lower()
    if asks_time:
        return True
    if not wish:
        return False
    cleaned = lower
    for phrase in (
        wish.get("typed") or "",
        wish.get("canonical", "").lower(),
        "good", "gud", "gd", "morning", "afternoon", "evening", "night",
        "gm", "gn", "hi", "hello", "hey", "hii", "hiii",
    ):
        if phrase:
            cleaned = re.sub(r"\b" + re.escape(phrase) + r"\b", " ", cleaned)
    cleaned = re.sub(r"[^a-z0-9]+", " ", cleaned).strip()
    return not cleaned or cleaned in {"sir", "dear", "bro", "friend", "aria", "ai"}


def _build_wish_or_time_reply(message):
    wish = _detect_wish(message)
    asks_time = _message_asks_time(message)
    if not _message_is_mostly_wish_or_time(message, wish, asks_time):
        return None

    now_info = _assistant_time_text()
    current_period = now_info["period"]
    current_wish = "Good " + current_period
    parts = []

    if wish:
        requested_period = wish.get("period")
        if requested_period and requested_period != current_period:
            parts.append(f"{current_wish}!")
            parts.append(
                f"Small correction: it is {current_period} here right now, so \"{current_wish}\" fits better than \"{wish['canonical']}\"."
            )
        elif requested_period:
            parts.append(f"{wish['canonical']}!")
            if wish.get("typo"):
                parts.append(f"Tiny correction: the usual spelling is \"{wish['canonical']}\".")
        else:
            parts.append("Hello!")
            if wish.get("typo"):
                parts.append("Tiny correction: \"Hi\" or \"Hello\" is the cleaner spelling.")

        if requested_period == "night" and requested_period == current_period:
            parts.append("Rest well, and take care.")
        else:
            parts.append("Hope you are doing well.")

    if asks_time:
        parts.append(
            f"The current local time is {now_info['time']} ({now_info['zone']}) on {now_info['date']}."
        )

    if wish and not asks_time:
        parts.append("How can I help you today?")

    return " ".join(parts).strip()


def _offline_general_answer(message, style):
    lower = (message or "").strip().lower()
    words = set(re.findall(r"[a-z0-9+#.']+", lower))
    if not lower:
        return None
    if "newton" in lower and ("second law" in lower or "2nd law" in lower):
        return _assistant_text(
            style,
            "Newton's second law says `F = m * a`. In simple words, more mass needs more force for the same acceleration, and the same force accelerates a lighter object more.",
            "Newton ka second law kehta hai `F = m * a`. Simple matlab: same acceleration ke liye heavy object ko zyada force chahiye, aur same force light object ko zyada accelerate karegi.",
        )
    if any(word in words for word in ["python", "function", "class", "loop", "list", "dictionary", "dict", "tuple", "code", "programming"]):
        return _assistant_text(
            style,
            "I can still help offline with Python and coding basics. Send the exact code, error, or concept and I will explain it step by step.",
            "Main offline mode me bhi Python aur coding basics me help kar sakta hoon. Exact code, error, ya concept bhejo, main usko step by step samjhaunga.",
        )
    if {"ai", "ml", "neural", "llm", "model", "training", "inference"} & words or "machine learning" in lower:
        return _assistant_text(
            style,
            "AI and ML in one line: a model learns patterns from data, then uses those patterns to predict or generate outputs. If you want, I can explain any part of that in simpler words.",
            "AI aur ML ka short version: model data se patterns seekhta hai, phir un patterns ko use karke prediction ya output banata hai. Chaaho to main isko aur simple tareeke se samjha sakta hoon.",
        )
    if any(word in words for word in ["math", "derivative", "integral", "matrix", "probability", "calculus", "algebra", "equation"]):
        return _assistant_text(
            style,
            "I can still explain math offline. Send the exact formula, problem statement, or topic name and I will break it into plain steps.",
            "Main math bhi offline explain kar sakta hoon. Exact formula, problem statement, ya topic name bhejo, main usko plain steps me tod kar samjhaunga.",
        )
    if any(word in words for word in ["physics", "force", "torque", "energy", "momentum", "velocity", "acceleration"]):
        return _assistant_text(
            style,
            "I can still explain core physics offline. Send the exact topic or question and I will answer with formula, intuition, and a small example.",
            "Main core physics offline bhi samjha sakta hoon. Exact topic ya question bhejo, main formula, intuition, aur chhota example ke saath answer dunga.",
        )
    return _assistant_text(
        style,
        "The cloud assistant is unavailable right now, so I am answering in offline mode. Ask a more specific question and I will still help directly instead of repeating a generic line.",
        "Cloud assistant abhi unavailable hai, isliye main offline mode me answer kar raha hoon. Thoda specific question poochho, main generic repeat karne ke bajay direct help dunga.",
    )


def _collect_assistant_sim_snapshot(sim_ref):
    return {
        "summary": sim_ref.get_runtime_summary(),
        "conjugate": sim_ref.get_conjugate_data(),
        "grasp_quality": sim_ref.get_grasp_quality_data(),
        "vision": sim_ref.get_vision_data(),
        "slip": sim_ref.get_slip_data(),
        "task_progress": sim_ref.get_task_progress(),
        "color_sort": sim_ref.get_color_sort_status(),
    }


def get_assistant_sim_context(page_context=None):
    page_ctx = page_context or {}
    page_name = ((page_ctx.get("page_name") or page_ctx.get("page") or "").strip().lower())
    if "double robot" in page_name:
        return {
            "page_mode": "double_robot",
            "robot1": _collect_assistant_sim_snapshot(sim_double_a),
            "robot2": _collect_assistant_sim_snapshot(sim_double_b),
        }
    sim_ctx = _collect_assistant_sim_snapshot(sim)
    if "color" in page_name and "sort" in page_name:
        sim_ctx["page_mode"] = "color_sort"
    elif "congruent" in page_name:
        sim_ctx["page_mode"] = "congruent_surface"
    else:
        sim_ctx["page_mode"] = "main"
    return sim_ctx


def build_assistant_system_prompt(page_context=None, user_message=None, history=None, web_context=None, include_sim_context=False):
    sim_ctx = get_assistant_sim_context(page_context) if include_sim_context else {}
    page_ctx = page_context or {}
    page_name = (page_ctx.get("page_name") or page_ctx.get("page") or "").strip()
    lower_page = page_name.lower()
    reply_style = _assistant_reply_style(user_message, history)
    current_time = _assistant_time_text()
    page_rules = [
        "Answer the user's actual question directly before adding extra detail.",
        "Do not repeat the same sentence, paragraph, or advice unless the user explicitly asks for repetition.",
        "Keep continuity with the recent chat history and avoid restarting the explanation from scratch.",
        "Behave like a full general-purpose AI assistant first, not like a simulation status bot.",
        "Do not mention the dashboard, robot, telemetry, workbench, or live simulation unless the user's message clearly asks about them.",
        "When the user greets or wishes you, respond warmly in English before continuing.",
        "If the user's greeting is wrong for the current local time, politely correct it in English and use the right greeting.",
        "If the user's greeting has a spelling mistake, politely provide the cleaner English wording.",
        "If the user asks for the time, answer with CURRENT_LOCAL_TIME directly.",
        "If you are unsure, say what is uncertain instead of bluffing.",
    ]
    if "double robot" in lower_page:
        page_rules.append(
            "On the Double Robot Workbench, clearly distinguish Robot 1 in the left cell from Robot 2 in the right cell."
        )
    elif "color" in lower_page and "sort" in lower_page:
        page_rules.append(
            "On the Color Sorting Workbench, use queue, target zone, color, pick state, and insertion state when relevant."
        )
    elif "congruent" in lower_page:
        page_rules.append(
            "On the Congruent Surface Workbench, use surface geometry, curvature, conjugate grasp, and insertion telemetry when relevant."
        )
    return (
        "You are ARIA, the AI assistant embedded inside the Gripper Simulation Software dashboard. "
        "You are a strong general-purpose assistant and also a simulation-aware robotics copilot. "
        "\n\nPrimary behavior: "
        "answer clearly, accurately, and naturally. Lead with the direct answer. Expand only as much as the question needs. "
        "Avoid repetitive phrasing, repeated summaries, and copy-paste style responses. "
        "\n\nLanguage behavior: "
        "reply in English by default, with a friendly, polite, human tone. "
        "Only switch to Hindi or Hinglish if the user explicitly asks for that language. "
        "\n\nGreeting and time behavior: "
        "answer wishes warmly. If someone says the wrong greeting for the current local time, gently correct it in English. "
        "If someone misspells a greeting, give the correct wording without sounding harsh. "
        "If someone asks the time, use CURRENT_LOCAL_TIME and do not guess. "
        "\n\nSimulation behavior: "
        "use LIVE_SIM_CONTEXT only when SIM_CONTEXT_ALLOWED=true and the user asks about the live dashboard, robot, telemetry, scene, or simulation status. "
        "If SIM_CONTEXT_ALLOWED=false, ignore LIVE_SIM_CONTEXT entirely. "
        "Mention exact state, force, grasp score, target, queue, slip, pose, or task only when relevant. "
        "If the robot is idle, say it plainly and suggest the next useful action. "
        "If something is failing, diagnose the likely reason from the available telemetry instead of giving generic advice. "
        "\n\nGeneral behavior: "
        "you can answer coding, math, physics, engineering, AI, writing, and general knowledge questions as well. "
        "For normal questions, answer the topic directly and do not add simulation details. "
        "For current, latest, news, price, weather, or explicit internet-search questions, use WEB_CONTEXT when provided. "
        "When WEB_CONTEXT has useful results, ground the answer in those results and include a short 'Sources:' section with 2-4 source titles or URLs. "
        "Avoid odd or irrelevant wording from source snippets unless it is truly important to the answer. "
        "If web search was requested but WEB_CONTEXT has no results, say the search did not return enough evidence instead of guessing. "
        "\n\nFormatting behavior: "
        "keep answers readable. Use short paragraphs by default. Use lists only when the content is naturally list-shaped. "
        "Use code blocks for code. "
        "\n\nCurrent simulation architecture: "
        "PyBullet-based Panda robot arm with conjugate grasping, peg handling, peg-in-hole tasks, color sorting, double robot workbench, and congruent surface analysis. "
        + (" ".join(page_rules) + " " if page_rules else "")
        + "\nREPLY_STYLE=" + reply_style
        + "\nCURRENT_LOCAL_TIME=" + json.dumps(current_time, ensure_ascii=True)
        + "\nSIM_CONTEXT_ALLOWED=" + json.dumps(bool(include_sim_context), ensure_ascii=True)
        + "\nWEB_CONTEXT=" + json.dumps(web_context or {}, ensure_ascii=True)
        + "\nLIVE_SIM_CONTEXT=" + json.dumps(sim_ctx, ensure_ascii=True)
        + "\nPAGE_CONTEXT=" + json.dumps(page_ctx, ensure_ascii=True)
    )

    return (
        "You are ARIA - an advanced, deeply knowledgeable AI assistant embedded in the "
        "Gripper Simulation Software dashboard. You are like a senior robotics engineer, "
        "physicist, and software expert combined. You must behave EXACTLY like ChatGPT or "
        "Gemini: give rich, detailed, well-structured, accurate, and helpful answers. "
        "NEVER give short, one-line, vague, or repetitive answers. "
        "NEVER repeat the same answer twice. Every response must be fresh, detailed, context-aware. "
        "\n\nYou can answer ANY topic: robotics, physics, mathematics, Python coding, AI/ML, "
        "engineering, general knowledge, history, science, and more. "
        "You are NOT just a simulation status bot. You are a FULL general-purpose AI. "
        "\n\nFor simulation questions, use the LIVE_SIM_CONTEXT data below to give accurate, "
        "specific, real-time answers. Examples: "
        "If asked about grasp quality, report the actual percentage and explain what it means. "
        "If asked about force, report the exact Newton value and whether it is safe. "
        "If the robot is IDLE, say so and suggest the next action. "
        "If pegs are dropping, diagnose the specific cause from telemetry. "
        "\n\nFor general questions (coding, math, science, etc.), give complete, expert-level "
        "answers with examples, formulas, code snippets, or step-by-step explanations as needed. "
        "\n\nLanguage rules: "
        "Match the user's language 100%. If user writes Hinglish (Hindi+English in Latin script), "
        "reply in natural flowing Hinglish. If user writes English, reply in English. "
        "If user switches language mid-conversation, switch immediately. "
        "\n\nFormatting rules: "
        "Use numbered lists for steps. Use bullet points for comparisons. "
        "Use bold headings when explaining multiple concepts. "
        "Use code blocks for code. Show mathematical formulas clearly. "
        "Lead with the direct answer, then give detail. "
        "Never start with 'I' as the first word. Never use filler phrases like 'As an AI...'. "
        "\n\nSimulation Architecture Context (always know this): "
        "This is a PyBullet-based robot simulation. Franka Panda robot arm uses conjugate grasping. "
        "Surfaces: Normal (flat, square peg), Inclined (15/30/45 deg, cylinder peg), "
        "Sculpture/Dome (curved, triangle peg). "
        "Color Sort: orange/blue/green pegs sorted by vision system onto conveyor. "
        "Double Robot: two independent Panda arms working on separate tasks simultaneously. "
        "Congruent Workbench: curvature analysis with Gaussian/Mean curvature metrics. "
        "Physics: CCD collision detection, 400000N grip force constraints, FSM state machine. "
        + (" ".join(page_rules) + " " if page_rules else "")
        + "\nREPLY_STYLE=" + reply_style
        + "\nLIVE_SIM_CONTEXT=" + json.dumps(sim_ctx, ensure_ascii=True)
        + "\nPAGE_CONTEXT=" + json.dumps(page_ctx, ensure_ascii=True)
    )


def normalize_assistant_history(history, max_items=12):
    cleaned = []
    for item in history or []:
        if not isinstance(item, dict):
            continue
        role = (item.get("role") or "").strip().lower()
        if role not in {"user", "assistant"}:
            continue
        content = (item.get("content") or "").strip()
        if not content:
            continue
        cleaned.append({"role": role, "content": content[:4000]})
    return cleaned[-max_items:]


def _message_targets_live_sim(message, page_context=None):
    lower = (message or "").strip().lower()
    if not lower:
        return False
    has_sim_marker = any(marker in lower for marker in SIMULATION_QUERY_MARKERS)
    has_live_intent = any(marker in lower for marker in LIVE_SIM_INTENT_MARKERS)
    if has_sim_marker and has_live_intent:
        return True
    return False


def _assistant_reply_signature(text):
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def _cleanup_assistant_reply(text):
    chunks = []
    seen = set()
    for raw in str(text or "").replace("\r\n", "\n").split("\n"):
        line = raw.rstrip()
        if not line.strip():
            if chunks and chunks[-1] != "":
                chunks.append("")
            continue
        sig = _assistant_reply_signature(line)
        if sig and sig in seen:
            continue
        if sig:
            seen.add(sig)
        chunks.append(line)
    return "\n".join(chunks).strip()


def _is_repetitive_assistant_reply(reply, history):
    candidate = _assistant_reply_signature(reply)
    if not candidate:
        return True
    recent = [
        _assistant_reply_signature(item.get("content"))
        for item in history or []
        if isinstance(item, dict) and (item.get("role") or "").strip().lower() == "assistant"
    ]
    for prev in recent[-3:]:
        if not prev:
            continue
        if candidate == prev:
            return True
        if difflib.SequenceMatcher(None, candidate, prev).ratio() >= 0.92:
            return True
    return False


def _read_remote_json(req, timeout_sec=ASSISTANT_REMOTE_TIMEOUT_SEC):
    with urllib.request.urlopen(req, timeout=timeout_sec) as response:
        raw = response.read().decode("utf-8")
        return json.loads(raw)


def _post_json(url, payload, headers=None, timeout_sec=ASSISTANT_REMOTE_TIMEOUT_SEC):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers or {"Content-Type": "application/json"},
        method="POST",
    )
    return _read_remote_json(req, timeout_sec=timeout_sec)


def default_assistant_provider(catalog=None):
    return "groq"


def resolve_provider_api_key(provider, explicit_key=None):
    key = (explicit_key or "").strip()
    if key:
        return key
    if provider == "groq":
        return (os.getenv("GROQ_API_KEY") or "").strip()
    return ""


def resolve_provider_label(provider, explicit_key=None):
    if provider == "groq":
        return "Groq"
    return "ARIA AI"


def _extract_chat_completion_text(payload):
    return (
        payload.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
        .strip()
    ) or "No response text returned."


def forward_assistant_chat(provider, message, attachments, page_context=None,
                           api_key_override=None, history=None, web_context=None,
                           include_sim_context=False):
    prompt_messages = [{
        "role": "system",
        "content": build_assistant_system_prompt(
            page_context,
            message,
            history,
            web_context,
            include_sim_context=include_sim_context,
        ),
    }]
    prompt_messages.extend(normalize_assistant_history(history))
    prompt_messages.append(
        {
            "role": "user",
            "content": message
            + "\nATTACHMENTS="
            + json.dumps(attachments or [], ensure_ascii=True)
            + "\nWEB_SEARCH_REQUESTED="
            + json.dumps(bool(web_context), ensure_ascii=True),
        }
    )

    if provider == "groq":
        api_key = resolve_provider_api_key(provider, api_key_override)
        if not api_key:
            raise ValueError("Groq API key is missing. Add GROQ_API_KEY to your .env file.")
        helper_path = os.path.join(PROJECT_ROOT, "groq_chat_helper.py")
        python_exe = os.path.join(PROJECT_ROOT, "venv", "Scripts", "python.exe")
        if not os.path.exists(python_exe):
            python_exe = sys.executable
        helper_payload = {
            "api_key": api_key,
            "model": get_groq_model(),
            "messages": prompt_messages,
            "temperature": 0.45,
            "max_tokens": 1500,
            "top_p": 0.95,
            "frequency_penalty": 0.55,
            "presence_penalty": 0.15,
            "timeout": 20.0,
        }
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        completed = subprocess.run(
            [python_exe, helper_path],
            input=json.dumps(helper_payload, ensure_ascii=True),
            text=True,
            capture_output=True,
            timeout=28,
            cwd=PROJECT_ROOT,
            creationflags=creationflags,
        )
        raw = (completed.stdout or "").strip()
        try:
            helper_result = json.loads(raw)
        except json.JSONDecodeError:
            helper_result = {"error": raw or (completed.stderr or "").strip()}
        if completed.returncode != 0 or helper_result.get("error"):
            raise RuntimeError(helper_result.get("error") or "Groq helper failed.")
        return (helper_result.get("reply") or "").strip() or "No response text returned."

    raise ValueError(f"Only Groq is enabled for assistant replies. Requested provider: {provider}")


def answer_locally(message, attachments, page_context=None, history=None):
    sim_ctx = get_assistant_sim_context(page_context)
    page_ctx = page_context or {}
    style = _assistant_reply_style(message, history)
    lower = (message or "").lower()
    lines = []
    specific_offline = _offline_general_answer(message, style)
    if (
        not _message_targets_live_sim(message, page_context)
        and specific_offline
        and "cloud assistant" not in specific_offline.lower()
    ):
        return specific_offline

    if lower in {"hi", "hello", "hey", "namaste", "hii"}:
        lines.append(_assistant_text(
            style,
            "Hello. ARIA AI is ready on this dashboard.",
            "Hello. ARIA AI is dashboard par ready hai.",
        ))
        lines.append(_assistant_text(
            style,
            "Ask anything normally. I will use live simulation context only when it helps the answer.",
            "Normal tareeke se kuch bhi poochho. Main live simulation context tabhi use karunga jab answer me useful hoga.",
        ))

    if any(word in lower for word in ["language", "hinglish", "english"]):
        lines.append(_assistant_text(
            style,
            "I can reply in English or Hinglish and I will match the user's wording.",
            "Main English ya Hinglish dono me reply kar sakta hoon, aur user jis style me poochega main wahi style match karunga.",
        ))

    if sim_ctx.get("page_mode") == "double_robot":
        robot_rows = [
            ("Robot 1 / Left", sim_ctx.get("robot1") or {}),
            ("Robot 2 / Right", sim_ctx.get("robot2") or {}),
        ]
        if any(word in lower for word in ["status", "state", "current", "abhi", "robot 1", "robot1", "robot 2", "robot2", "left", "right"]):
            lines.append(_assistant_text(
                style,
                "Current double-robot status:",
                "Abhi double-robot status:",
            ))
            for label, snap in robot_rows:
                summary = snap.get("summary") or {}
                progress = snap.get("task_progress") or {}
                target = (
                    summary.get("target_zone")
                    or summary.get("current_surface")
                    or summary.get("focused_surface")
                    or "none"
                )
                lines.append(
                    f"- {label}: {summary.get('state', 'IDLE')} | task {summary.get('active_task') or 'none'} | target {target} | progress {progress.get('completed', 0)}/{progress.get('total', 0)}"
                )
        if any(word in lower for word in ["grasp", "pick", "peg", "insert", "hole", "drop", "slip", "jaw", "tight"]):
            lines.append(_assistant_text(
                style,
                "Latest grasp and insertion health:",
                "Latest grasp aur insertion health:",
            ))
            for label, snap in robot_rows:
                grasp = snap.get("grasp_quality") or {}
                slip = snap.get("slip") or {}
                score = round((grasp.get("quality_score") or 0.0) * 100)
                drift = slip.get("drift_mm")
                drift_text = f"{drift:.1f} mm" if drift is not None else "n/a"
                lines.append(
                    f"- {label}: quality {score}% | bilateral {'yes' if grasp.get('has_bilateral') else 'no'} | slip {'yes' if slip.get('slipped') else 'no'} | drift {drift_text}"
                )
    else:
        summary = sim_ctx["summary"]
        grasp = sim_ctx["grasp_quality"]
        slip = sim_ctx["slip"]
        progress = sim_ctx["task_progress"]
        vision = sim_ctx["vision"]
        color_sort = sim_ctx.get("color_sort") or {}

        if any(word in lower for word in ["status", "state", "current", "abhi"]):
            lines.append(_assistant_text(
                style,
                "Current system status:",
                "Abhi system status:",
            ))
            lines.append(f"- State: {summary.get('state', 'IDLE')}")
            lines.append(f"- Active task: {summary.get('active_task') or 'none'}")
            lines.append(
                f"- Target: {summary.get('target_zone') or summary.get('current_surface') or summary.get('focused_surface') or 'none'}"
            )
            lines.append(
                f"- Progress: {progress.get('completed', 0)}/{progress.get('total', 0)} completed, {progress.get('failed', 0)} failed"
            )

        if any(word in lower for word in ["grasp", "pick", "peg", "insert", "hole", "drop", "slip", "jaw", "tight"]):
            lines.append(_assistant_text(
                style,
                "Grasp and insertion health:",
                "Grasp aur insertion health:",
            ))
            lines.append(f"- Quality score: {round((grasp.get('quality_score') or 0.0) * 100)}%")
            lines.append(f"- Bilateral contact: {'yes' if grasp.get('has_bilateral') else 'no'}")
            lines.append(f"- Slip detected: {'yes' if slip.get('slipped') else 'no'}")
            drift = slip.get("drift_mm")
            if drift is not None:
                lines.append(f"- Drift: {drift:.1f} mm")

        if any(word in lower for word in ["vision", "camera", "detect", "color", "sort"]):
            lines.append(_assistant_text(
                style,
                "Vision and sort status:",
                "Vision aur sort status:",
            ))
            lines.append(f"- Color: {vision.get('color') or color_sort.get('current_color') or 'none'}")
            lines.append(f"- Shape: {vision.get('shape') or summary.get('current_shape') or 'none'}")
            if color_sort.get("active"):
                zone = color_sort.get("target_zone") or {}
                lines.append(
                    f"- Sort target: {zone.get('label') or 'none'} | queue remaining {color_sort.get('queue_remaining', 0)}"
                )
            conf = vision.get("confidence")
            if conf is not None:
                lines.append(f"- Confidence: {round(conf * 100)}%")

    if any(word in lower for word in ["free", "provider", "model", "cloud", "local"]):
        lines.append(_assistant_text(
            style,
            "Assistant availability:",
            "Assistant availability:",
        ))
        lines.append(_assistant_text(
            style,
            "- This dashboard is configured to use Groq cloud AI for normal assistant replies.",
            "- Ye dashboard normal assistant replies ke liye Groq cloud AI use karta hai.",
        ))
        lines.append(_assistant_text(
            style,
            "- If Groq is temporarily unavailable, I can still answer from live simulation telemetry on this page.",
            "- Agar Groq temporary unavailable ho, to main is page ke live simulation telemetry se bhi answer de sakta hoon.",
        ))
        lines.append(_assistant_text(
            style,
            "- Reports and charts in this dashboard are generated from local simulation telemetry.",
            "- Is dashboard ke reports aur charts local simulation telemetry se bante hain.",
        ))
        lines.append(_assistant_text(
            style,
            "- Video in this dashboard is simulation recording, not cloud generative video.",
            "- Is dashboard ka video simulation recording hai, cloud generative video nahi.",
        ))

    if any(word in lower for word in ["help", "guide", "how", "kaise", "use", "start", "workflow", "workbench"]):
        page_name = page_ctx.get("page_name") or page_ctx.get("page") or "current page"
        lines.append(_assistant_text(
            style,
            f"Workbench guidance for {page_name}:",
            f"{page_name} ke liye quick guidance:",
        ))
        if page_ctx.get("hint"):
            lines.append(f"- {page_ctx['hint']}")
        if page_ctx.get("instructions"):
            lines.append(f"- {page_ctx['instructions']}")

    if attachments:
        lines.append(_assistant_text(
            style,
            f"Attachment summary: {len(attachments)} item(s) attached.",
            f"Attachment summary: {len(attachments)} item attached hai.",
        ))
        for item in attachments[:3]:
            label = item.get("name") or item.get("kind") or "attachment"
            lines.append(f"- {label}")

    if not lines:
        if _message_targets_live_sim(message, page_context):
            summary = sim_ctx.get("summary") or {}
            state = summary.get("state", "IDLE")
            task = summary.get("active_task") or summary.get("preview_task") or "none"
            target = summary.get("target_zone") or summary.get("current_surface") or summary.get("focused_surface") or "none"
            lines.append(_assistant_text(
                style,
                f"Live dashboard summary: state {state}, task {task}, target {target}.",
                f"Live dashboard summary: state {state}, task {task}, target {target}.",
            ))
            hint = summary.get("state_hint")
            if hint:
                lines.append(hint)
            return "\n".join(lines)
        offline_reply = _offline_general_answer(message, style)
        if offline_reply:
            lines.append(offline_reply)
        else:
            lines.append(_assistant_text(
                style,
                "I can help with live status, peg-in-hole performance, surfaces, color sorting, telemetry, reports, snapshots, and simulation workflow questions.",
                "Main live status, peg-in-hole performance, surfaces, color sorting, telemetry, reports, snapshots aur simulation workflow questions me help kar sakta hoon.",
            ))
            lines.append(_assistant_text(
                style,
                "For general questions, I will use the Groq-backed assistant when it is reachable.",
                "General questions ke liye main Groq-backed assistant use karunga jab connection reachable hoga.",
            ))
    return "\n".join(lines)


def build_local_assistant_fallback(message, attachments, page_context=None, history=None):
    return answer_locally(message, attachments, page_context, history)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed_snapshot')
def video_feed_snapshot():
    frame = sim.get_frame()
    if not frame:
        return Response(status=204)
    return Response(frame, mimetype='image/jpeg')


@app.route('/assistant/config')
def assistant_config():
    catalog = get_assistant_provider_catalog()
    return jsonify(
        {
            "providers": catalog,
            "default_provider": default_assistant_provider(catalog),
            "source_disclosed": False,
            "listening_supported": False,
            "speaking_supported": True,
            "camera_supported": False,
            "file_upload_supported": True,
            "figure_supported": False,
            "video_supported": True,
            "notes": [
                "The assistant is configured to use Groq cloud AI in this local app build.",
                "Current/latest questions can include live web-search context before the AI answers.",
                "Simulation telemetry is used only when the user asks about the robot, scene, or dashboard status.",
                "If Groq is temporarily unavailable, assistant replies can still fall back to a local direct answer.",
                "Video in this dashboard is local simulation recording.",
            ],
        }
    )


@app.route('/assistant/chat', methods=['POST'])
def assistant_chat():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    attachments = data.get("attachments") or []
    page_context = data.get("page_context") or {}
    api_key = (data.get("api_key") or "").strip()
    force_local = bool(data.get("force_local"))
    history = normalize_assistant_history(data.get("history"))

    if not message:
        return jsonify({"error": "missing_message"}), 400

    direct_reply = _build_wish_or_time_reply(message)
    if direct_reply:
        return jsonify(
            {
                "provider": "local",
                "provider_label": "ARIA AI",
                "reply": direct_reply,
                "mode": "local_direct",
            }
        )

    catalog = get_assistant_provider_catalog()
    provider = default_assistant_provider(catalog)
    prefer_live_local = _message_targets_live_sim(message, page_context)
    web_context = None
    if not prefer_live_local and _message_needs_web(message):
        web_context = _search_web_with_retries(message)
        if not web_context.get("results"):
            app.logger.warning("Assistant web search returned no results: %s", web_context.get("error") or "no error")
            if not _message_explicitly_requests_web(message):
                web_context = None

    if force_local or prefer_live_local:
        reply = build_local_assistant_fallback(
            message,
            attachments,
            page_context,
            history,
        )
        reply = _cleanup_assistant_reply(reply)
        return jsonify(
            {
                "provider": provider,
                "provider_label": "ARIA AI",
                "reply": reply,
                "mode": "live_local" if prefer_live_local and not force_local else "fallback_local",
            }
        )

    provider_is_configured = catalog.get(provider, {}).get("configured", False)
    if provider == "groq":
        provider_is_configured = provider_is_configured or bool(
            resolve_provider_api_key(provider, api_key)
        )

    if not provider_is_configured:
        reply = (
            _build_web_fallback(message, _assistant_reply_style(message, history), web_context)
            if web_context else
            build_local_assistant_fallback(
                message,
                attachments,
                page_context,
                history,
            )
        )
        reply = _cleanup_assistant_reply(reply)
        return jsonify(
            {
                "provider": provider,
                "provider_label": "ARIA AI",
                "reply": reply,
                "mode": "fallback_local",
            }
        )

    try:
        reply = forward_assistant_chat(
            provider,
            message,
            attachments,
            page_context,
            api_key_override=api_key,
            history=history,
            web_context=web_context,
            include_sim_context=prefer_live_local,
        )
        reply = _cleanup_assistant_reply(reply)
        if _is_repetitive_assistant_reply(reply, history):
            reply = _cleanup_assistant_reply(
                build_local_assistant_fallback(
                    message,
                    attachments,
                    page_context,
                    history,
                )
            )
        provider_label = resolve_provider_label(provider, api_key)
        return jsonify(
            {
                "provider": provider,
                "provider_label": provider_label,
                "reply": reply,
                "mode": "remote",
            }
        )
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="ignore")
        try:
            err_payload = json.loads(raw)
        except json.JSONDecodeError:
            err_payload = {"error": raw or exc.reason}
        app.logger.warning("Assistant remote HTTP error for %s: %s", provider, err_payload)
        reply = (
            _build_web_fallback(message, _assistant_reply_style(message, history), web_context)
            if web_context else
            build_local_assistant_fallback(
                message,
                attachments,
                page_context,
                history,
            )
        )
        reply = _cleanup_assistant_reply(reply)
        return jsonify(
            {
                "provider": provider,
                "provider_label": "ARIA AI",
                "reply": reply,
                "mode": "fallback_local",
            }
        )
    except Exception as exc:
        app.logger.warning("Assistant remote error for %s: %s", provider, exc)
        reply = (
            _build_web_fallback(message, _assistant_reply_style(message, history), web_context)
            if web_context else
            build_local_assistant_fallback(
                message,
                attachments,
                page_context,
                history,
            )
        )
        reply = _cleanup_assistant_reply(reply)
        return jsonify(
            {
                "provider": provider,
                "provider_label": "ARIA AI",
                "reply": reply,
                "mode": "fallback_local",
            }
        )

@app.route('/double_feed/<rid>')
def double_feed(rid):
    sim_ref = sim_double_a if rid == '1' else sim_double_b
    def gen_double_frames():
        while True:
            frame = sim_ref.get_single_view_frame()
            if not frame:
                time.sleep(0.03)
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.05)
    return Response(gen_double_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def _decode_stream_frame(frame_bytes):
    if not frame_bytes:
        return None
    try:
        arr = np.frombuffer(frame_bytes, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _make_double_overview_frame(frame_a, frame_b):
    def prepare(img, label):
        if img is None:
            img = np.zeros((540, 860, 3), dtype=np.uint8)
            cv2.putText(img, f"{label} loading...", (34, 68),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (180, 220, 255), 2, cv2.LINE_AA)
        else:
            img = cv2.resize(img, (860, 540))
        cv2.rectangle(img, (0, 0), (img.shape[1], 56), (7, 14, 24), -1)
        cv2.putText(img, label, (22, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 212, 255), 2, cv2.LINE_AA)
        return img

    left = prepare(frame_a, "Robot 1 - Left Cell")
    right = prepare(frame_b, "Robot 2 - Right Cell")
    gutter = np.zeros((left.shape[0], 18, 3), dtype=np.uint8)
    combined = np.hstack([left, gutter, right])
    canvas = np.zeros((combined.shape[0] + 72, combined.shape[1], 3), dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], canvas.shape[0]), (4, 9, 15), -1)
    cv2.putText(canvas, "Full Dual Robot Cell View", (26, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 136), 2, cv2.LINE_AA)
    cv2.putText(canvas,
                "Both robots visible together for one clear full-workbench overview",
                (26, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (190, 208, 224), 1, cv2.LINE_AA)
    canvas[72:, :] = combined
    ok, enc = cv2.imencode('.jpg', canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return enc.tobytes() if ok else None


@app.route('/double_feed/full')
def double_full_feed():
    def gen_double_full_frames():
        while True:
            frame_a = _decode_stream_frame(sim_double_a.get_single_view_frame())
            frame_b = _decode_stream_frame(sim_double_b.get_single_view_frame())
            merged = _make_double_overview_frame(frame_a, frame_b)
            if not merged:
                time.sleep(0.05)
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + merged + b'\r\n')
            time.sleep(0.06)
    return Response(gen_double_full_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/double_command', methods=['POST'])
def double_command():
    data = request.get_json(silent=True) or {}
    rid = str(data.get("robot") or "").strip()
    cmd = (data.get("cmd") or "").strip()
    if not cmd:
        return jsonify({"status": "error", "error": "missing_cmd"}), 400
    sim_ref = sim_double_a if rid == '1' else sim_double_b
    result = sim_ref.send_command(cmd) or {}
    return jsonify({
        "status": "ok" if result.get("accepted", True) else "busy",
        "robot": rid,
        "cmd": cmd,
        **result,
    })

@app.route('/double_assign', methods=['POST'])
def double_assign():
    data = request.get_json(silent=True) or {}
    r1 = (data.get("robot1") or "").strip()
    r2 = (data.get("robot2") or "").strip()
    res1 = None
    res2 = None
    if r1:
        res1 = sim_double_a.send_command(r1)
    if r2:
        res2 = sim_double_b.send_command(r2)
    ok = all((item or {}).get("accepted", True) for item in [res1, res2] if item is not None)
    return jsonify({
        "status": "ok" if ok else "partial",
        "robot1": r1,
        "robot2": r2,
        "robot1_result": res1,
        "robot2_result": res2,
    })

@app.route('/double_preview', methods=['POST'])
def double_preview():
    data = request.get_json(silent=True) or {}
    r1 = (data.get("robot1") or "").strip()
    r2 = (data.get("robot2") or "").strip()
    return jsonify({
        "status": "ok",
        "robot1": sim_double_a.set_preview_task(r1),
        "robot2": sim_double_b.set_preview_task(r2),
    })

@app.route('/double_reset', methods=['POST'])
def double_reset():
    sim_double_a.send_command("RESET")
    sim_double_b.send_command("RESET")
    return jsonify({"status": "ok"})

@app.route('/double_speech_events')
def double_speech_events():
    r1_speech = request.args.get('r1_speech_seq', 0, type=int)
    r1_sound = request.args.get('r1_sound_seq', 0, type=int)
    r2_speech = request.args.get('r2_speech_seq', 0, type=int)
    r2_sound = request.args.get('r2_sound_seq', 0, type=int)
    return jsonify({
        "robot1": sim_double_a.get_speech_events(r1_speech, r1_sound),
        "robot2": sim_double_b.get_speech_events(r2_speech, r2_sound),
    })

@app.route('/double_status')
def double_status():
    return jsonify({
        "robot1": build_status_payload(sim_double_a),
        "robot2": build_status_payload(sim_double_b),
    })


@app.route('/double_dashboard_data')
def double_dashboard_data():
    robot1 = build_double_dashboard_payload(sim_double_a)
    robot2 = build_double_dashboard_payload(sim_double_b)
    aggregate = {
        "states": [robot1.get("state"), robot2.get("state")],
        "any_active": any(state and state != "IDLE" for state in [robot1.get("state"), robot2.get("state")]),
        "active_count": sum(1 for state in [robot1.get("state"), robot2.get("state")] if state and state != "IDLE"),
        "assigned_tasks": [
            robot1.get("summary", {}).get("active_task") or robot1.get("summary", {}).get("preview_task"),
            robot2.get("summary", {}).get("active_task") or robot2.get("summary", {}).get("preview_task"),
        ],
    }
    return jsonify({
        "robot1": robot1,
        "robot2": robot2,
        "aggregate": aggregate,
    })

@app.route('/command/<cmd>', methods=['POST'])
def send_command(cmd):
    result = sim.send_command(cmd) or {}
    return jsonify({
        "status": "success" if result.get("accepted", True) else "busy",
        "cmd": cmd,
        **result,
    })


@app.route('/reset', methods=['POST'])
def reset_command():
    sim.send_command("RESET")
    return jsonify({"status": "success", "cmd": "RESET"})

@app.route('/status')
def get_status():
    state     = sim.get_status()
    peg_shape = getattr(sim, 'current_peg_shape', None)
    conj      = sim.get_conjugate_data()
    summary   = sim.get_runtime_summary()
    analysis  = {
        "state":            state,
        "peg_shape":        peg_shape,
        "conjugate_active": conj["active"],
        "conj_shape":       conj["shape"],
        "conj_contact":     conj["contact"],
        "conj_force":       conj["force"],
        "conj_width":       conj["width"],
        "conj_angle":       conj["angle"],
        "conj_closure":     conj["closure"],
        "curvature_K":      conj.get("curvature_K"),
        "curvature_H":      conj.get("curvature_H"),
        "surface_type":     conj.get("surface_type"),
        "force_history":    sim._force_history,
        "grasp_quality":    sim._grasp_quality,
        "task_stats":       sim._task_stats,
        "summary":          summary,
        "active_task":      summary.get("active_task"),
        "active_mode":      summary.get("mode"),
        "focused_surface":  summary.get("focused_surface"),
        "current_surface":  summary.get("current_surface"),
        "current_color":    summary.get("current_color"),
        "target_zone":      summary.get("target_zone"),
        "current_shape":    summary.get("current_shape"),
        "insertion_force_peak": summary.get("insertion_force_peak"),
    }
    return jsonify(analysis)

@app.route('/conjugate_status')
def conjugate_status():
    state     = sim.get_status()
    peg_shape = getattr(sim, 'current_peg_shape', None)
    return jsonify({
        "state":            state,
        "conjugate_active": state not in ("IDLE",) and peg_shape is not None,
        "peg_shape":        peg_shape,
    })

@app.route('/telemetry')
def full_telemetry():
    return jsonify(sim.get_full_telemetry())

@app.route('/vision_status')
def vision_status():
    return jsonify(sim.get_vision_data())

@app.route('/grasp_quality')
def grasp_quality():
    return jsonify(sim.get_grasp_quality_data())

@app.route('/task_progress')
def task_progress():
    return jsonify(sim.get_task_progress())

@app.route('/force_history')
def force_history():
    return jsonify(sim.get_force_history())

@app.route('/slip_status')
def slip_status():
    return jsonify(sim.get_slip_data())

@app.route('/color_sort_status')
def color_sort_status():
    return jsonify(sim.get_color_sort_status())

def gen_color_sort_frames():
    while True:
        frame = sim.get_color_sort_frame()
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.05)

@app.route('/color_sort_feed')
def color_sort_feed():
    return Response(gen_color_sort_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/speech_events')
def speech_events():
    since_speech = request.args.get('speech_seq', 0, type=int)
    since_sound  = request.args.get('sound_seq',  0, type=int)
    return jsonify(sim.get_speech_events(since_speech, since_sound))

@app.route('/task_snapshots')
def task_snapshots():
    """Returns all completed task snapshots with base64 jpg + telemetry data."""
    return jsonify(sim.get_completed_snapshots())

@app.route('/download_report')
def download_report():
    """Generate downloadable report - format: json, txt, pdf, doc, html."""
    fmt   = request.args.get('format', 'json').lower()
    snaps = sim.get_completed_snapshots()
    logs  = sim.get_sim_data_log()

    def _clean_snaps(snaps_in):
        clean = []
        for s in snaps_in:
            sc = {k: v for k, v in s.items() if k != 'jpg_b64'}
            clean.append(sc)
        return clean

    def _safe_div(a, b):
        return (a / b) if b else 0.0

    def _mean(vals):
        return sum(vals) / len(vals) if vals else 0.0

    def _median(vals):
        if not vals:
            return 0.0
        svals = sorted(vals)
        mid = len(svals) // 2
        if len(svals) % 2:
            return float(svals[mid])
        return (float(svals[mid - 1]) + float(svals[mid])) / 2.0

    def _percentile(vals, pct):
        if not vals:
            return 0.0
        if len(vals) == 1:
            return float(vals[0])
        svals = sorted(float(v) for v in vals)
        pos = max(0.0, min(len(svals) - 1, (len(svals) - 1) * (pct / 100.0)))
        low = int(math.floor(pos))
        high = int(math.ceil(pos))
        if low == high:
            return svals[low]
        frac = pos - low
        return svals[low] + (svals[high] - svals[low]) * frac

    def _histogram(vals, bins=6, decimals=1):
        if not vals:
            return [], []
        arr = [float(v) for v in vals]
        vmin = min(arr)
        vmax = max(arr)
        if abs(vmax - vmin) < 1e-9:
            return [f"{round(vmin, decimals)}"], [len(arr)]
        step = (vmax - vmin) / float(bins)
        counts = [0] * bins
        labels = []
        for i in range(bins):
            start = vmin + step * i
            end = vmax if i == bins - 1 else vmin + step * (i + 1)
            labels.append(f"{round(start, decimals)}-{round(end, decimals)}")
        for value in arr:
            idx = min(bins - 1, int((value - vmin) / step))
            counts[idx] += 1
        return labels, counts

    def _count_by(items, key_fn):
        out = {}
        for it in items:
            k = key_fn(it)
            out[k] = out.get(k, 0) + 1
        return out

    def _state_durations(evts):
        states = {}
        seq = [e for e in evts if e.get("event") == "state_change"]
        for i in range(len(seq) - 1):
            st = (seq[i].get("data") or {}).get("state") or seq[i].get("state", "UNKNOWN")
            dt = max(0.0, float(seq[i + 1].get("ts", 0)) - float(seq[i].get("ts", 0)))
            states[st] = states.get(st, 0.0) + dt
        return states

    def _sentiment(metrics):
        total = metrics["task_stats"]["total"]
        success = metrics["task_stats"]["completed"]
        failed = metrics["task_stats"]["failed"]
        success_rate = _safe_div(success, total)
        gq = metrics["kpis"]["avg_grasp_quality"]
        slip_rate = _safe_div(metrics["events"]["slip"], total)
        drop_rate = _safe_div(metrics["events"]["drops"], total)
        avg_force = metrics["kpis"]["avg_peak_force"]
        force_penalty = min(1.0, avg_force / 1200.0) if avg_force > 0 else 0.0

        score = (0.45 * success_rate +
                 0.25 * gq +
                 0.15 * (1.0 - slip_rate) +
                 0.15 * (1.0 - drop_rate) -
                 0.10 * force_penalty)
        score = max(0.0, min(1.0, score))
        if score >= 0.70:
            label = "Positive"
        elif score >= 0.50:
            label = "Neutral"
        else:
            label = "Negative"

        notes = []
        if failed > 0:
            notes.append("Failures detected; review recovery and grasp quality.")
        if slip_rate > 0.10:
            notes.append("Slip rate elevated; check grip force and alignment.")
        if drop_rate > 0.05:
            notes.append("Drop rate non-trivial; refine approach and contact timing.")
        if avg_force > 900:
            notes.append("Insertion force high; increase clearance or alignment steps.")
        if not notes:
            notes.append("System stable with consistent insertion behavior.")
        return {"score": round(score, 3), "label": label, "notes": notes}

    # Task list
    tasks = [s for s in snaps if s.get("event") in ("insertion_complete", "sort_complete")]
    stats = sim.get_task_progress()
    total = int(stats.get("total", len(tasks)))
    completed = int(stats.get("completed", len(tasks)))
    failed = int(stats.get("failed", max(0, total - completed)))
    in_prog = max(0, total - completed - failed)

    peak_forces = [float(t["peak_force_N"]) for t in tasks if t.get("peak_force_N") is not None]
    gq_scores = []
    for t in tasks:
        gq = t.get("grasp_quality") or {}
        if gq.get("quality_score") is not None:
            gq_scores.append(float(gq.get("quality_score")))
    sorted_tasks = sorted(tasks, key=lambda t: float(t.get("ts", 0.0)))
    task_records = []
    prev_task_ts = None
    for idx, task in enumerate(sorted_tasks, start=1):
        task_ts = float(task.get("ts", 0.0) or 0.0)
        cycle_gap = (task_ts - prev_task_ts) if prev_task_ts else None
        prev_task_ts = task_ts
        gq = task.get("grasp_quality") or {}
        task_records.append(
            {
                "task_id": idx,
                "event": task.get("event", ""),
                "surface": task.get("surface") or task.get("zone") or "UNKNOWN",
                "shape": task.get("shape") or "UNKNOWN",
                "color": task.get("color") or "",
                "peak_force_N": round(float(task.get("peak_force_N") or 0.0), 2),
                "grasp_quality": round(float(gq.get("quality_score") or 0.0), 3),
                "bilateral": bool(gq.get("has_bilateral")),
                "timestamp_s": round(task_ts, 3),
                "cycle_gap_s": round(cycle_gap, 3) if cycle_gap is not None else None,
            }
        )

    counts_surface = _count_by(tasks, lambda t: t.get("surface") or t.get("zone") or "UNKNOWN")
    counts_shape   = _count_by(tasks, lambda t: t.get("shape") or "UNKNOWN")
    counts_color   = _count_by(tasks, lambda t: t.get("color") or "UNKNOWN")
    state_times    = _state_durations(logs)
    slip_count     = len([e for e in logs if e.get("event") == "slip_detected"])
    drop_count     = len([e for e in logs if e.get("event") == "peg_drop"])

    metrics = {
        "meta": {
            "project": "Robot Peg-in-Hole Simulation",
            "author":  "Shashwat Mishra",
        },
        "task_stats": {"total": total, "completed": completed, "failed": failed, "in_progress": in_prog},
        "kpis": {
            "success_rate": round(_safe_div(completed, total), 3),
            "avg_peak_force": round(_mean(peak_forces), 2) if peak_forces else 0.0,
            "median_peak_force": round(_median(peak_forces), 2) if peak_forces else 0.0,
            "p95_peak_force": round(_percentile(peak_forces, 95), 2) if peak_forces else 0.0,
            "max_peak_force": round(max(peak_forces), 2) if peak_forces else 0.0,
            "avg_grasp_quality": round(_mean(gq_scores), 3) if gq_scores else 0.0,
            "median_grasp_quality": round(_median(gq_scores), 3) if gq_scores else 0.0,
            "slip_rate": round(_safe_div(slip_count, total), 3),
            "drop_rate": round(_safe_div(drop_count, total), 3),
        },
        "counts": {
            "by_surface": counts_surface,
            "by_shape": counts_shape,
            "by_color": counts_color,
        },
        "events": {"slip": slip_count, "drops": drop_count},
        "state_durations_s": state_times,
        "series": {"peak_forces": peak_forces},
        "task_records": task_records,
    }
    force_hist_labels, force_hist_counts = _histogram(peak_forces, bins=6, decimals=1)
    quality_hist_labels, quality_hist_counts = _histogram(
        [q * 100.0 for q in gq_scores], bins=5, decimals=0
    )
    completion_timeline = list(range(1, len(task_records) + 1))
    metrics["distributions"] = {
        "peak_force_histogram": {
            "labels": force_hist_labels,
            "counts": force_hist_counts,
        },
        "grasp_quality_histogram_pct": {
            "labels": quality_hist_labels,
            "counts": quality_hist_counts,
        },
        "completion_timeline": {
            "labels": [str(i) for i in completion_timeline],
            "counts": completion_timeline,
        },
    }
    sorted_surface = sorted(counts_surface.items(), key=lambda kv: (-kv[1], kv[0]))
    sorted_shape = sorted(counts_shape.items(), key=lambda kv: (-kv[1], kv[0]))
    sorted_color = sorted(counts_color.items(), key=lambda kv: (-kv[1], kv[0]))
    sorted_states = sorted(state_times.items(), key=lambda kv: (-kv[1], kv[0]))

    # JSON report
    if fmt == 'json':
        import json
        data = json.dumps({
            "project":   metrics["meta"]["project"],
            "author":    metrics["meta"]["author"],
            "metrics":   metrics,
            "task_records": task_records,
            "snapshots": _clean_snaps(snaps),
            "event_log": logs[-200:]
        }, indent=2)
        resp = make_response(data)
        resp.headers['Content-Type']        = 'application/json'
        resp.headers['Content-Disposition'] = 'attachment; filename=sim_report.json'
        return resp

    # Text report
    import datetime
    lines = [
        "=" * 82,
        "ROBOT PEG-IN-HOLE SIMULATION REPORT",
        "Author  : Shashwat Mishra",
        "Date    : " + str(datetime.datetime.now()),
        "=" * 82,
        "",
        "Measured telemetry only. No inferred sentiment or synthetic role summary is included.",
        "",
        "Outcome Summary:",
        f"  Total Tasks       : {total}",
        f"  Completed         : {completed}",
        f"  Failed            : {failed}",
        f"  In Progress       : {in_prog}",
        f"  Success Rate      : {metrics['kpis']['success_rate']:.1%}",
        "",
        "Force and Quality Summary:",
        f"  Average Peak Force: {metrics['kpis']['avg_peak_force']:.2f} N",
        f"  Median Peak Force : {metrics['kpis']['median_peak_force']:.2f} N",
        f"  P95 Peak Force    : {metrics['kpis']['p95_peak_force']:.2f} N",
        f"  Max Peak Force    : {metrics['kpis']['max_peak_force']:.2f} N",
        f"  Avg Grasp Quality : {metrics['kpis']['avg_grasp_quality']:.1%}",
        f"  Median Grasp Qual.: {metrics['kpis']['median_grasp_quality']:.1%}",
        f"  Slip Rate         : {metrics['kpis']['slip_rate']:.1%}",
        f"  Drop Rate         : {metrics['kpis']['drop_rate']:.1%}",
    ]
    lines.append("")

    lines.append("Counts by Surface:")
    for k, v in sorted_surface:
        lines.append(f"  {k}: {v}")
    lines.append("Counts by Shape:")
    for k, v in sorted_shape:
        lines.append(f"  {k}: {v}")
    lines.append("Counts by Color:")
    for k, v in sorted_color:
        lines.append(f"  {k}: {v}")

    lines += ["", "Peak Force Histogram:", "-" * 48]
    for label, count in zip(force_hist_labels, force_hist_counts):
        lines.append(f"  {label:18s} {count}")
    lines += ["", "Grasp Quality Histogram (%):", "-" * 48]
    for label, count in zip(quality_hist_labels, quality_hist_counts):
        lines.append(f"  {label:18s} {count}")

    lines += ["", "Task Details:", "-" * 40]
    for row in task_records:
        lines.append(f"Task {row['task_id']}  [{row['event']}]")
        lines.append(f"  Surface  : {row['surface']}")
        lines.append(f"  Shape    : {row['shape']}")
        lines.append(f"  Color    : {row['color'] or '-'}")
        lines.append(f"  Peak F   : {row['peak_force_N']:.2f} N")
        lines.append(f"  Grasp Q  : {row['grasp_quality']:.3f}")
        lines.append(f"  Bilateral: {'YES' if row['bilateral'] else 'NO'}")
        lines.append(f"  Cycle Gap: {row['cycle_gap_s'] if row['cycle_gap_s'] is not None else '-'}")
        lines.append("")

    lines += ["", "State Durations (s):", "-" * 40]
    for st, dur in sorted_states:
        lines.append(f"  {st:20s} {dur:8.2f}")

    lines += ["", "Event Log (last 60):", "-" * 40]
    for entry in logs[-60:]:
        lines.append(f"  [{entry.get('event','?')}] state={entry.get('state','?')}")

    text = "\n".join(lines)

    # HTML/SVG charts
    def _svg_bar(labels, values, title):
        if not labels:
            return "<div>No data</div>"
        w, h, m = 420, 180, 24
        maxv = max(values) if values else 1
        bw = (w - 2 * m) / max(1, len(values))
        svg = [f"<svg width='{w}' height='{h}' viewBox='0 0 {w} {h}'>"]
        svg.append(f"<rect x='0' y='0' width='{w}' height='{h}' fill='#0b1420' rx='8'/>")
        svg.append(f"<text x='{m}' y='18' fill='#8bd6ff' font-size='12' font-weight='700'>{title}</text>")
        for i, (lab, val) in enumerate(zip(labels, values)):
            bh = (h - 2 * m - 20) * (val / maxv)
            x  = m + i * bw + 8
            y  = h - m - bh
            svg.append(f"<rect x='{x:.1f}' y='{y:.1f}' width='{bw-16:.1f}' height='{bh:.1f}' fill='#00d4ff' rx='4'/>")
            svg.append(f"<text x='{x:.1f}' y='{h-6}' fill='#6a7b92' font-size='9'>{lab[:10]}</text>")
        svg.append("</svg>")
        return "".join(svg)

    def _svg_line(values, title):
        w, h, m = 420, 180, 24
        if len(values) < 2:
            return "<div>No line data</div>"
        maxv = max(values) if values else 1
        minv = min(values) if values else 0
        rng = max(1e-6, maxv - minv)
        pts = []
        for i, v in enumerate(values):
            x = m + (w - 2 * m) * (i / (len(values) - 1))
            y = h - m - (h - 2 * m - 10) * ((v - minv) / rng)
            pts.append(f"{x:.1f},{y:.1f}")
        svg = [f"<svg width='{w}' height='{h}' viewBox='0 0 {w} {h}'>"]
        svg.append(f"<rect x='0' y='0' width='{w}' height='{h}' fill='#0b1420' rx='8'/>")
        svg.append(f"<text x='{m}' y='18' fill='#8bd6ff' font-size='12' font-weight='700'>{title}</text>")
        svg.append(f"<polyline fill='none' stroke='#00ff88' stroke-width='2' points='{' '.join(pts)}'/>")
        svg.append("</svg>")
        return "".join(svg)

    def _svg_pie(labels, values, title):
        if not values or sum(values) <= 0:
            return "<div>No pie data</div>"
        w, h = 220, 180
        cx, cy, r = 90, 95, 60
        total = sum(values)
        colors = ["#00d4ff", "#00ff88", "#ff9f43", "#cc66ff", "#ff4b2b", "#8888ff"]
        svg = [f"<svg width='{w}' height='{h}' viewBox='0 0 {w} {h}'>"]
        svg.append(f"<rect x='0' y='0' width='{w}' height='{h}' fill='#0b1420' rx='8'/>")
        svg.append(f"<text x='12' y='18' fill='#8bd6ff' font-size='12' font-weight='700'>{title}</text>")
        start = 0.0
        for i, v in enumerate(values):
            if v <= 0:
                continue
            ang = v / total * 360.0
            end = start + ang
            large = 1 if ang > 180 else 0
            x1 = cx + r * math.cos(math.radians(start))
            y1 = cy + r * math.sin(math.radians(start))
            x2 = cx + r * math.cos(math.radians(end))
            y2 = cy + r * math.sin(math.radians(end))
            path = (f"M {cx} {cy} L {x1:.1f} {y1:.1f} "
                    f"A {r} {r} 0 {large} 1 {x2:.1f} {y2:.1f} Z")
            svg.append(f"<path d='{path}' fill='{colors[i % len(colors)]}'/>")
            start = end
        # Legend
        ly = 32
        for i, lab in enumerate(labels):
            svg.append(f"<rect x='160' y='{ly-8}' width='8' height='8' fill='{colors[i % len(colors)]}'/>")
            svg.append(f"<text x='172' y='{ly-1}' fill='#8aa0b8' font-size='9'>{lab[:10]}</text>")
            ly += 12
        svg.append("</svg>")
        return "".join(svg)

    def _html_escape(value):
        text = "" if value is None else str(value)
        return (text
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;"))

    def _html_table(headers, rows):
        header_html = "".join([f"<th>{_html_escape(h)}</th>" for h in headers])
        if rows:
            body_html = "".join([
                "<tr>" + "".join([f"<td>{_html_escape(cell)}</td>" for cell in row]) + "</tr>"
                for row in rows
            ])
        else:
            body_html = f"<tr><td colspan='{len(headers)}'>No data</td></tr>"
        return f"<table><tr>{header_html}</tr>{body_html}</table>"

    def _chart_card(svg, caption):
        return (
            "<div class='card chart-card'>"
            + svg
            + f"<div class='chart-caption'>{_html_escape(caption)}</div></div>"
        )

    surf_labels = [label for label, _ in sorted_surface]
    surf_vals   = [value for _, value in sorted_surface]
    pie_labels  = ["Completed", "Failed", "In Progress"]
    pie_vals    = [completed, failed, in_prog]
    outcome_rows = [
        ["Total Tasks", total],
        ["Completed", completed],
        ["Failed", failed],
        ["In Progress", in_prog],
        ["Success Rate", f"{metrics['kpis']['success_rate']:.1%}"],
    ]
    force_rows = [
        ["Average Peak Force (N)", f"{metrics['kpis']['avg_peak_force']:.2f}"],
        ["Median Peak Force (N)", f"{metrics['kpis']['median_peak_force']:.2f}"],
        ["P95 Peak Force (N)", f"{metrics['kpis']['p95_peak_force']:.2f}"],
        ["Max Peak Force (N)", f"{metrics['kpis']['max_peak_force']:.2f}"],
        ["Average Grasp Quality", f"{metrics['kpis']['avg_grasp_quality']:.1%}"],
        ["Median Grasp Quality", f"{metrics['kpis']['median_grasp_quality']:.1%}"],
        ["Slip Rate", f"{metrics['kpis']['slip_rate']:.1%}"],
        ["Drop Rate", f"{metrics['kpis']['drop_rate']:.1%}"],
    ]
    surface_rows = [[k, v] for k, v in sorted_surface]
    shape_rows = [[k, v] for k, v in sorted_shape]
    color_rows = [[k, v] for k, v in sorted_color]
    state_rows = [[k, f"{v:.2f}"] for k, v in sorted_states]
    force_hist_rows = [[label, count] for label, count in zip(force_hist_labels, force_hist_counts)]
    quality_hist_rows = [[label, count] for label, count in zip(quality_hist_labels, quality_hist_counts)]
    task_record_rows = [[
        row["task_id"],
        row["event"],
        row["surface"],
        row["shape"],
        row["color"] or "-",
        f"{row['peak_force_N']:.2f}",
        f"{row['grasp_quality']:.3f}",
        "YES" if row["bilateral"] else "NO",
        row["cycle_gap_s"] if row["cycle_gap_s"] is not None else "-",
    ] for row in task_records]
    log_rows = [[
        entry.get("event", "?"),
        entry.get("state", "?"),
        f"{float(entry.get('ts', 0.0)):.3f}",
    ] for entry in logs[-30:]]

    generated_on = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html = render_template(
        "sim_report.html",
        generated_on=generated_on,
        total=total,
        completed=completed,
        failed=failed,
        in_progress=in_prog,
        success_rate=f"{metrics['kpis']['success_rate']:.1%}",
        median_peak_force=f"{metrics['kpis']['median_peak_force']:.1f}",
        p95_peak_force=f"{metrics['kpis']['p95_peak_force']:.1f}",
        slip_rate=f"{metrics['kpis']['slip_rate']:.1%}",
        drop_rate=f"{metrics['kpis']['drop_rate']:.1%}",
        outcome_table=_html_table(['Metric', 'Value'], outcome_rows),
        force_table=_html_table(['Metric', 'Value'], force_rows),
        surface_table=_html_table(['Surface', 'Count'], surface_rows),
        shape_table=_html_table(['Shape', 'Count'], shape_rows),
        color_table=_html_table(['Color', 'Count'], color_rows),
        force_hist_table=_html_table(['Peak Force Bin (N)', 'Count'], force_hist_rows),
        quality_hist_table=_html_table(['Grasp Quality Bin (%)', 'Count'], quality_hist_rows),
        state_table=_html_table(['State', 'Seconds'], state_rows),
        task_table=_html_table(
            ['ID', 'Event', 'Surface / Zone', 'Shape', 'Color', 'Peak Force (N)', 'Grasp Q', 'Bilateral', 'Cycle Gap (s)'],
            task_record_rows,
        ),
        log_table=_html_table(['Event', 'State', 'Timestamp (s)'], log_rows),
        chart_cards=[
            _chart_card(_svg_bar(surf_labels, surf_vals, 'Tasks by Surface'), 'Bar chart showing how many completed tasks ran on each target surface.'),
            _chart_card(_svg_pie(pie_labels, pie_vals, 'Task Outcomes'), 'Pie chart showing the completed, failed, and in-progress task mix.'),
            _chart_card(_svg_line(peak_forces, 'Peak Force per Task'), 'Line chart of measured peak insertion force for each completed task.'),
            _chart_card(_svg_line(completion_timeline, 'Task Completion Timeline'), 'Monotonic timeline of completed tasks in chronological order.'),
            _chart_card(_svg_bar(force_hist_labels, force_hist_counts, 'Peak Force Histogram'), 'Histogram showing how often peak force fell into each bin.'),
            _chart_card(_svg_bar(quality_hist_labels, quality_hist_counts, 'Grasp Quality Histogram (%)'), 'Histogram showing the distribution of grasp quality scores in percent.'),
            _chart_card(
                _svg_bar(
                    [k for k, _ in sorted(state_times.items(), key=lambda kv: -kv[1])[:6]],
                    [round(v, 2) for _, v in sorted(state_times.items(), key=lambda kv: -kv[1])[:6]],
                    'Top Stage Durations (s)',
                ),
                'Bar chart of the states that consumed the most total simulation time.',
            ),
        ],
    )

    if fmt in ('doc', 'html'):
        resp = make_response(html)
        if fmt == 'doc':
            resp.headers['Content-Type']        = 'application/msword; charset=utf-8'
            resp.headers['Content-Disposition'] = 'attachment; filename=sim_report.doc'
        else:
            resp.headers['Content-Type']        = 'text/html; charset=utf-8'
            resp.headers['Content-Disposition'] = 'attachment; filename=sim_report.html'
        return resp

    if fmt == 'pdf':
        import textwrap

        def _pdf_escape(s):
            return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

        def _pdf_text_block(x, y, lines_in, size=10, leading=13):
            out = ["BT", f"/F1 {size} Tf", f"{x} {y} Td"]
            for ln in lines_in:
                out.append(f"({_pdf_escape(ln)}) Tj")
                out.append(f"0 -{leading} Td")
            out.append("ET")
            return "\n".join(out)

        def _wrap_cell(value, width):
            text = "" if value is None else str(value)
            text = text.replace("\n", " ")
            return textwrap.wrap(text, width=width) or [""]

        def _table_lines(title, headers, rows, widths):
            sep = "+-" + "-+-".join(["-" * width for width in widths]) + "-+"

            def _format_row(parts):
                return "| " + " | ".join([
                    str(parts[idx])[:widths[idx]].ljust(widths[idx])
                    for idx in range(len(widths))
                ]) + " |"

            lines_local = [title, sep, _format_row(headers), sep]
            if not rows:
                lines_local.append(_format_row(["No data"] + [""] * (len(widths) - 1)))
                lines_local.append(sep)
                return lines_local
            for row in rows:
                wrapped = [_wrap_cell(row[idx], widths[idx]) for idx in range(len(widths))]
                row_height = max([len(cell) for cell in wrapped])
                for line_idx in range(row_height):
                    parts = [
                        wrapped[col_idx][line_idx] if line_idx < len(wrapped[col_idx]) else ""
                        for col_idx in range(len(widths))
                    ]
                    lines_local.append(_format_row(parts))
                lines_local.append(sep)
            return lines_local

        def _bar_lines(title, items, label_width=18, bar_width=24):
            lines_local = [title]
            if not items:
                lines_local.append("No data")
                return lines_local
            max_value = max([value for _, value in items]) or 1
            for label, value in items:
                value_num = float(value)
                filled = int(round((value_num / max_value) * bar_width)) if max_value else 0
                if value_num > 0 and filled == 0:
                    filled = 1
                bar = "#" * filled
                lines_local.append(
                    f"{str(label)[:label_width].ljust(label_width)} | {bar.ljust(bar_width)} | {value}"
                )
            return lines_local

        pdf_blocks = []
        pdf_blocks.append([
            "ROBOT PEG-IN-HOLE SIMULATION REPORT",
            f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "Measured telemetry only. No inferred sentiment or synthetic role summary is included.",
        ])
        pdf_blocks.append(_table_lines(
            "Outcome Summary",
            ["Metric", "Value"],
            outcome_rows,
            [24, 18],
        ))
        pdf_blocks.append(_table_lines(
            "Force and Quality Summary",
            ["Metric", "Value"],
            force_rows,
            [24, 18],
        ))
        pdf_blocks.append(_bar_lines("Tasks by Surface", surface_rows))
        pdf_blocks.append(_bar_lines("Task Outcomes", [
            ("Completed", completed),
            ("Failed", failed),
            ("In Progress", in_prog),
        ], label_width=14, bar_width=28))
        pdf_blocks.append(_bar_lines(
            "Peak Force Histogram",
            list(zip(force_hist_labels, force_hist_counts)),
            label_width=18,
            bar_width=22,
        ))
        pdf_blocks.append(_bar_lines(
            "Grasp Quality Histogram (%)",
            list(zip(quality_hist_labels, quality_hist_counts)),
            label_width=18,
            bar_width=22,
        ))
        pdf_blocks.append(_table_lines(
            "Counts by Shape",
            ["Shape", "Count"],
            shape_rows,
            [22, 10],
        ))
        pdf_blocks.append(_table_lines(
            "Counts by Color",
            ["Color", "Count"],
            color_rows,
            [22, 10],
        ))
        pdf_blocks.append(_table_lines(
            "State Durations (s)",
            ["State", "Seconds"],
            state_rows,
            [26, 12],
        ))
        for chunk_start in range(0, len(task_records), 10):
            chunk = task_records[chunk_start:chunk_start + 10]
            pdf_blocks.append(_table_lines(
                f"Task Records {chunk_start + 1}-{chunk_start + len(chunk)}",
                ["ID", "Event", "Surface", "Shape", "Color"],
                [[
                    row["task_id"],
                    row["event"],
                    row["surface"],
                    row["shape"],
                    row["color"] or "-",
                ] for row in chunk],
                [3, 12, 16, 10, 10],
            ))
            pdf_blocks.append(_table_lines(
                f"Task Metrics {chunk_start + 1}-{chunk_start + len(chunk)}",
                ["ID", "Peak N", "GQ", "Bilat", "Time s", "Gap s"],
                [[
                    row["task_id"],
                    f"{row['peak_force_N']:.2f}",
                    f"{row['grasp_quality']:.3f}",
                    "YES" if row["bilateral"] else "NO",
                    f"{row['timestamp_s']:.3f}",
                    row["cycle_gap_s"] if row["cycle_gap_s"] is not None else "-",
                ] for row in chunk],
                [3, 8, 6, 5, 10, 8],
            ))
        pdf_blocks.append(_table_lines(
            "Task Log (last 20)",
            ["Event", "State", "Timestamp s"],
            [[
                entry.get("event", "?"),
                entry.get("state", "?"),
                f"{float(entry.get('ts', 0.0)):.3f}",
            ] for entry in logs[-20:]],
            [24, 18, 12],
        ))

        pages = []
        current_page = []
        max_lines_per_page = 56

        def _append_block(block_lines):
            nonlocal current_page
            if not block_lines:
                return
            extra_gap = 1 if current_page else 0
            if current_page and len(current_page) + len(block_lines) + extra_gap > max_lines_per_page:
                pages.append(current_page)
                current_page = []
                extra_gap = 0
            if extra_gap:
                current_page.append("")
            current_page.extend(block_lines)

        for block in pdf_blocks:
            _append_block(block)
        if current_page:
            pages.append(current_page)
        if not pages:
            pages = [["No telemetry data available."]]

        page_streams = []
        total_pages = len(pages)
        for page_index, page_lines in enumerate(pages, start=1):
            content = [
                _pdf_text_block(40, 770, [
                    "Robot Peg-in-Hole Simulation Report",
                    f"Page {page_index} of {total_pages}",
                ], size=11, leading=13),
                _pdf_text_block(40, 740, page_lines, size=8, leading=10),
            ]
            page_streams.append("\n".join(content).encode("latin1", errors="ignore"))

        object_map = {
            1: "<< /Type /Catalog /Pages 2 0 R >>\n",
            3: "<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>\n",
        }
        kids = []
        next_object_id = 4
        for stream in page_streams:
            page_object_id = next_object_id
            content_object_id = next_object_id + 1
            next_object_id += 2
            kids.append(page_object_id)
            object_map[page_object_id] = (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_object_id} 0 R >>\n"
            )
            object_map[content_object_id] = (
                f"<< /Length {len(stream)} >>\nstream\n"
                f"{stream.decode('latin1', errors='ignore')}\nendstream\n"
            )
        object_map[2] = f"<< /Type /Pages /Kids [{' '.join([f'{kid} 0 R' for kid in kids])}] /Count {len(kids)} >>\n"

        max_object_id = max(object_map.keys())
        pdf = "%PDF-1.4\n"
        offsets = {0: 0}
        for object_id in range(1, max_object_id + 1):
            offsets[object_id] = len(pdf)
            pdf += f"{object_id} 0 obj\n{object_map[object_id]}endobj\n"
        xref_pos = len(pdf)
        pdf += f"xref\n0 {max_object_id + 1}\n"
        pdf += "0000000000 65535 f \n"
        for object_id in range(1, max_object_id + 1):
            pdf += f"{offsets[object_id]:010d} 00000 n \n"
        pdf += f"trailer\n<< /Size {max_object_id + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF"
        resp = make_response(pdf.encode("latin1", errors="ignore"))
        resp.headers['Content-Type']        = 'application/pdf'
        resp.headers['Content-Disposition'] = 'attachment; filename=sim_report.pdf'
        return resp

    resp = make_response(text)
    resp.headers['Content-Type']        = 'text/plain; charset=utf-8'
    resp.headers['Content-Disposition'] = 'attachment; filename=sim_report.txt'
    return resp

if __name__ == '__main__':
    if not acquire_single_instance_lock():
        print("Another dashboard server is already running on this machine. Close the existing server before starting a new one.")
        raise SystemExit(0)
    print("Starting Simulation Engine...")
    sim.start()
    sim_double_a.start()
    sim_double_b.start()
    print("Starting Web Server on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False, threaded=True)
 
