(function () {
  if (window.__ariaWidgetLoaded) return;
  window.__ariaWidgetLoaded = true;

  var pageContext = window.ARIA_WIDGET_CONTEXT || {};
  var feedSelector = window.ARIA_WIDGET_FEED_SELECTOR || "img";
  var pageName = pageContext.page_name || document.title || "this workbench";
  var assistantLogoSrc = "/static/aria_bot_logo.svg";
  var providers = null;
  var defaultProvider = "groq";
  var chatHistory = [];
  var speakingEnabled = true;
  var attachments = [];

  var css = [
    "@keyframes ariaMiniSpin{to{transform:rotate(360deg)}}",
    "@keyframes ariaMiniPulse{0%,100%{transform:scale(1);opacity:.7}50%{transform:scale(1.16);opacity:1}}",
    "#aria-mini-fab{position:fixed;right:20px;bottom:24px;width:68px;height:68px;border:1px solid rgba(112,206,245,.32);border-radius:50%;z-index:240;background:radial-gradient(circle at 32% 24%,rgba(255,255,255,.16),rgba(8,18,30,.98) 44%,rgba(6,12,20,.98) 100%);color:#dff7ff;font:800 12px/1 Segoe UI,sans-serif;cursor:pointer;box-shadow:0 16px 34px rgba(0,0,0,.34),0 0 0 1px rgba(255,255,255,.04);transition:transform .16s ease,box-shadow .16s ease,border-color .16s ease,background .16s ease;touch-action:manipulation;-webkit-user-select:none;user-select:none;backdrop-filter:blur(12px);overflow:visible}",
    "#aria-mini-fab::before{content:'';position:absolute;inset:5px;border-radius:50%;border:1px solid rgba(112,206,245,.14);pointer-events:none}",
    "#aria-mini-fab span{position:relative;z-index:1;display:flex;align-items:center;justify-content:center;line-height:1.02;font-weight:900;letter-spacing:1px;font-size:10px;text-shadow:none}",
    "#aria-mini-fab:hover,#aria-mini-fab.open{transform:translateY(-1px);border-color:rgba(112,206,245,.52);box-shadow:0 20px 42px rgba(0,0,0,.38),0 0 0 1px rgba(112,206,245,.14)}",
    ".aria-mini-logo{position:absolute;overflow:hidden;border-radius:inherit;pointer-events:none;z-index:1}",
    ".aria-mini-logo img{position:absolute;object-fit:contain;pointer-events:none;filter:drop-shadow(0 10px 18px rgba(0,0,0,.28)) saturate(1.03) brightness(1.02)}",
    ".aria-mini-fab-logo{inset:10px;border-radius:50%;background:transparent}",
    ".aria-mini-fab-logo img{inset:0;width:100%;height:100%}",
    ".aria-mini-fab-badge{position:absolute;right:-2px;bottom:-1px;z-index:3;width:18px;height:18px;border-radius:50%;background:linear-gradient(135deg,#00ffae,#53d6ff);border:2px solid rgba(6,12,20,.96);box-shadow:0 0 0 1px rgba(112,206,245,.10),0 8px 14px rgba(0,0,0,.26);animation:ariaMiniPulse 1.8s ease-in-out infinite}",
    ".aria-mini-fab-close{font-size:13px;letter-spacing:1.2px;color:#e6f7ff}",
    "#aria-mini-panel{position:fixed;right:20px;bottom:96px;width:min(420px,calc(100vw - 36px));height:min(560px,calc(100dvh - 132px));max-height:none;display:none;flex-direction:column;z-index:241;background:linear-gradient(180deg,rgba(8,14,24,.95),rgba(11,18,30,.92));border:1px solid rgba(0,212,255,.20);border-radius:20px;box-shadow:0 18px 40px rgba(0,0,0,.30);overflow:hidden;isolation:isolate;backdrop-filter:blur(18px) saturate(124%);touch-action:auto}",
    "#aria-mini-panel::before{content:'';position:absolute;inset:0;border-radius:inherit;background:linear-gradient(180deg,rgba(255,255,255,.06),rgba(255,255,255,0) 34%,rgba(0,212,255,.04));pointer-events:none}",
    "#aria-mini-panel::after{content:'';position:absolute;inset:-38%;background:conic-gradient(from 0deg,rgba(0,212,255,.18),rgba(0,255,136,.10),rgba(255,96,160,.16),rgba(255,205,90,.10),rgba(0,212,255,.18));filter:blur(24px);opacity:.22;pointer-events:none;animation:ariaMiniSpin 8s linear infinite}",
    "#aria-mini-panel.open{display:flex}",
    "#aria-mini-head{display:flex;align-items:center;gap:10px;padding:12px 14px;border-bottom:1px solid rgba(0,212,255,.10);background:rgba(255,255,255,.03)}",
    "#aria-mini-avatar{width:46px;height:46px;border-radius:50%;display:flex;align-items:center;justify-content:center;background:radial-gradient(circle at 34% 26%,rgba(255,255,255,.16),rgba(8,18,30,.98) 48%,rgba(6,12,20,.98) 100%);position:relative;box-shadow:0 0 0 1px rgba(255,255,255,.08),0 10px 20px rgba(0,0,0,.16)}",
    "#aria-mini-avatar::before{content:'';position:absolute;inset:0;border-radius:inherit;background:linear-gradient(180deg,rgba(255,255,255,.06),rgba(255,255,255,0) 42%,rgba(0,212,255,.03));opacity:1}",
    "#aria-mini-avatar::after{display:none}",
    ".aria-mini-avatar-logo{inset:6px;border-radius:50%;background:transparent}",
    ".aria-mini-avatar-logo img{inset:0;width:100%;height:100%}",
    "#aria-mini-title{font:800 14px/1.2 Segoe UI,sans-serif;color:#eef6ff;letter-spacing:.9px}",
    "#aria-mini-sub{font:11px/1.4 Segoe UI,sans-serif;color:#8da4bb;margin-top:3px}",
    "#aria-mini-close{margin-left:auto;border:none;background:none;color:#90a6bf;font-size:16px;cursor:pointer;min-width:38px;min-height:38px;border-radius:10px;touch-action:manipulation}",
    "#aria-mini-tools{display:flex;gap:8px;flex-wrap:wrap;padding:10px 14px 8px}",
    ".aria-mini-btn{border:1px solid rgba(255,255,255,.10);background:rgba(255,255,255,.04);color:#dce9f7;border-radius:10px;padding:8px 10px;min-height:42px;font-size:11px;cursor:pointer;touch-action:manipulation;-webkit-user-select:none;user-select:none;transition:transform .10s ease,background .12s ease,border-color .12s ease,box-shadow .12s ease;display:inline-flex;align-items:center;justify-content:center}",
    ".aria-mini-btn.active{border-color:rgba(0,255,136,.35);color:#9dffca;background:rgba(0,255,136,.09)}",
    ".aria-mini-btn:active,.aria-mini-btn.pressing,#aria-mini-close:active,#aria-mini-close.pressing,#aria-mini-send:active,#aria-mini-send.pressing,#aria-mini-fab:active,#aria-mini-fab.pressing{transform:scale(.97);background:rgba(0,212,255,.16);border-color:rgba(0,212,255,.34);box-shadow:0 0 0 1px rgba(0,212,255,.16)}",
    ".aria-mini-btn[hidden]{display:none!important}",
    "#aria-mini-body{display:flex;flex-direction:column;flex:1;min-height:0;overflow:hidden}",
    "#aria-mini-note{padding:0 14px 10px;font:11px/1.5 Segoe UI,sans-serif;color:#8da4bb}",
    "#aria-mini-attach{padding:0 14px 10px;font:11px/1.4 Segoe UI,sans-serif;color:#9dffca;display:none}",
    "#aria-mini-attach.show{display:block}",
    "#aria-mini-msgs{flex:1;min-height:0;overflow:auto;padding:0 14px 10px;display:flex;flex-direction:column;gap:10px;scroll-behavior:smooth;overscroll-behavior:contain;touch-action:auto}",
    "#aria-mini-msgs::-webkit-scrollbar{width:4px}#aria-mini-msgs::-webkit-scrollbar-thumb{background:rgba(0,212,255,.18);border-radius:4px}",
    ".aria-mini-bot,.aria-mini-user{max-width:94%;padding:10px 12px;border-radius:13px;font:12px/1.6 Segoe UI,sans-serif;white-space:pre-wrap;word-break:break-word}",
    ".aria-mini-bot{align-self:flex-start;background:rgba(255,255,255,.025);border:1px solid rgba(255,255,255,.08);color:#d8e6f5}",
    ".aria-mini-user{align-self:flex-end;background:rgba(0,212,255,.09);border:1px solid rgba(0,212,255,.22);color:#ecf8ff}",
    "#aria-mini-row{display:flex;gap:8px;padding:10px 14px 14px;border-top:1px solid rgba(255,255,255,.06);position:relative;z-index:2;background:linear-gradient(180deg,rgba(7,11,18,.14),rgba(7,11,18,.96) 22%,rgba(7,11,18,.98))}",
    "#aria-mini-input{flex:1;resize:none;min-height:42px;max-height:100px;border-radius:12px;border:1px solid rgba(0,212,255,.14);background:rgba(255,255,255,.025);color:#e7f2ff;padding:10px 12px;font:12px/1.4 Segoe UI,sans-serif;outline:none;touch-action:auto}",
    "#aria-mini-send{width:46px;height:46px;border:none;border-radius:12px;background:linear-gradient(135deg,#00d4ff,#00a8d8);color:#00131e;font-size:16px;cursor:pointer;touch-action:manipulation;-webkit-user-select:none;user-select:none;transition:transform .10s ease,box-shadow .12s ease,opacity .12s ease}",
    "#aria-mini-send:disabled{opacity:.35;cursor:default;transform:none}",
    "@media (max-width: 900px),(pointer: coarse){#aria-mini-panel{right:8px;left:8px;bottom:86px;width:auto;height:min(52dvh,calc(100dvh - 154px))}#aria-mini-tools{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}#aria-mini-row{padding:10px 12px 12px}#aria-mini-send{width:48px;height:48px}#aria-mini-fab{right:10px;bottom:12px;width:64px;height:64px}}",
    "@keyframes ariaLauncherGlow{0%,100%{transform:scale(1);box-shadow:0 18px 34px rgba(14,116,144,.22),0 0 0 1px rgba(255,255,255,.76) inset,0 0 0 rgba(103,232,249,0)}50%{transform:scale(1.045);box-shadow:0 22px 42px rgba(14,116,144,.28),0 0 0 1px rgba(255,255,255,.88) inset,0 0 26px rgba(103,232,249,.42)}}",
    "@keyframes ariaBoltFlash{0%,100%{filter:brightness(1);transform:scale(1)}45%{filter:brightness(1.35);transform:scale(1.16) rotate(-5deg)}58%{filter:brightness(1.9);transform:scale(.96) rotate(5deg)}}",
    "#aria-mini-fab{width:62px;height:62px;border-radius:20px;background:linear-gradient(145deg,#f8feff,#bae6fd 62%,#a7f3d0);border:1px solid rgba(14,165,233,.46);box-shadow:0 18px 34px rgba(14,116,144,.22),0 0 0 1px rgba(255,255,255,.76) inset;animation:ariaLauncherGlow 2.6s ease-in-out infinite}",
    "#aria-mini-fab::before{inset:0;border-radius:20px;border:0;background:linear-gradient(135deg,rgba(255,255,255,.55),transparent 54%,rgba(16,185,129,.18));opacity:.9}",
    ".aria-mini-fab-logo{inset:7px;border-radius:15px}.aria-mini-fab-logo img{filter:none}.aria-mini-fab-badge{width:24px;height:24px;right:-5px;bottom:-5px;border-radius:10px;background:linear-gradient(135deg,#0ea5e9,#10b981);border:2px solid rgba(248,254,255,.95);box-shadow:0 10px 18px rgba(14,116,144,.24),0 0 0 1px rgba(14,165,233,.20);animation:ariaBoltFlash 1.45s ease-in-out infinite}.aria-mini-fab-badge::before{content:'';display:block;width:11px;height:15px;margin:4px auto 0;background:#fff7ed;clip-path:polygon(58% 0,10% 58%,48% 58%,35% 100%,90% 42%,54% 42%)}#aria-mini-fab:hover:not(.open){transform:scale(1.14);border-color:rgba(14,165,233,.72);box-shadow:0 24px 48px rgba(14,116,144,.34),0 0 0 1px rgba(255,255,255,.95) inset,0 0 34px rgba(103,232,249,.58)}#aria-mini-fab.open{opacity:0;transform:scale(.72);pointer-events:none;visibility:hidden;animation:none}",
    "#aria-mini-panel{border-radius:18px;background:linear-gradient(180deg,rgba(15,23,42,.98),rgba(5,9,20,.97));border:1px solid rgba(148,163,184,.20);box-shadow:0 22px 54px rgba(0,0,0,.42),0 0 0 1px rgba(255,255,255,.04)}",
    "#aria-mini-panel::after{display:none}#aria-mini-panel::before{background:linear-gradient(180deg,rgba(103,232,249,.08),rgba(255,255,255,0) 32%)}",
    "#aria-mini-head{padding:13px 14px;background:rgba(255,255,255,.035);border-bottom:1px solid rgba(148,163,184,.14)}",
    "#aria-mini-avatar{width:42px;height:42px;border-radius:13px;background:#07111d;box-shadow:0 0 0 1px rgba(103,232,249,.26)}.aria-mini-avatar-logo{inset:5px;border-radius:10px}.aria-mini-avatar-logo img{filter:none}",
    "#aria-mini-title{font:800 13px/1.2 Segoe UI,sans-serif;letter-spacing:.2px;color:#f8fafc}#aria-mini-sub{font:11px/1.35 Segoe UI,sans-serif;color:#94a3b8}",
    "#aria-mini-close{color:#94a3b8;background:rgba(255,255,255,.035);border:1px solid rgba(148,163,184,.14)}",
    "#aria-mini-tools{padding:10px 12px 8px}.aria-mini-btn{border-radius:9px;min-height:36px;padding:7px 10px;background:rgba(15,23,42,.72);border:1px solid rgba(148,163,184,.16);font-size:11px;color:#cbd5e1}.aria-mini-btn.active{background:rgba(52,211,153,.10);border-color:rgba(52,211,153,.32);color:#bbf7d0}",
    "#aria-mini-note{padding:0 12px 9px;color:#94a3b8}#aria-mini-msgs{padding:0 12px 10px;gap:9px}.aria-mini-bot,.aria-mini-user{font:12.5px/1.55 Segoe UI,sans-serif;border-radius:12px;padding:10px 12px}.aria-mini-bot{background:rgba(15,23,42,.70);border:1px solid rgba(148,163,184,.13);color:#dbeafe}.aria-mini-user{background:rgba(14,165,233,.13);border:1px solid rgba(56,189,248,.26);color:#f0f9ff}",
    "#aria-mini-row{padding:10px 12px 12px;background:rgba(2,6,23,.88);border-top:1px solid rgba(148,163,184,.12)}#aria-mini-input{border-radius:10px;background:rgba(15,23,42,.82);border:1px solid rgba(148,163,184,.20);font:12.5px/1.42 Segoe UI,sans-serif;color:#f8fafc}#aria-mini-input:focus{border-color:rgba(103,232,249,.45);box-shadow:0 0 0 3px rgba(14,165,233,.12)}#aria-mini-send{border-radius:10px;background:linear-gradient(135deg,#67e8f9,#34d399);color:#04111d;font-weight:900}"
  ].join("");

  var style = document.createElement("style");
  style.textContent = css;
  document.head.appendChild(style);

  var fab = document.createElement("button");
  fab.id = "aria-mini-fab";
  fab.type = "button";
  fab.textContent = "AI";
  var miniFabClosedMarkup = '<span class="aria-mini-logo aria-mini-fab-logo"><img src="' + assistantLogoSrc + '" alt="ARIA AI"></span><span class="aria-mini-fab-badge" aria-hidden="true"></span>';

  var panel = document.createElement("div");
  panel.id = "aria-mini-panel";
  panel.innerHTML = [
    '<div id="aria-mini-head">',
      '<div id="aria-mini-avatar" aria-hidden="true"><span class="aria-mini-logo aria-mini-avatar-logo"><img src="' + assistantLogoSrc + '" alt=""></span></div>',
      '<div><div id="aria-mini-title">ARIA AI</div><div id="aria-mini-sub">Groq-powered AI assistant</div></div>',
      '<button id="aria-mini-close" type="button">x</button>',
    "</div>",
    '<div id="aria-mini-tools">',
      '<button class="aria-mini-btn" id="aria-mini-shot" type="button">Sim Shot</button>',
      '<button class="aria-mini-btn active" id="aria-mini-voice" type="button">Voice On</button>',
      '<button class="aria-mini-btn" id="aria-mini-clear" type="button">Clear Chat</button>',
    "</div>",
    '<div id="aria-mini-body">',
      '<div id="aria-mini-note">Ask anything. ARIA uses Groq and internet search when the question needs fresh information.</div>',
      '<div id="aria-mini-attach"></div>',
      '<div id="aria-mini-msgs"></div>',
    '</div>',
    '<div id="aria-mini-row">',
      '<textarea id="aria-mini-input" placeholder="Ask anything..." rows="1"></textarea>',
      '<button id="aria-mini-send" type="button">></button>',
    "</div>"
  ].join("");

  document.body.appendChild(fab);
  document.body.appendChild(panel);
  fab.innerHTML = miniFabClosedMarkup;

  function $(id) {
    return document.getElementById(id);
  }

  function addMsg(text, cls) {
    var el = document.createElement("div");
    el.className = cls === "user" ? "aria-mini-user" : "aria-mini-bot";
    el.textContent = text;
    $("aria-mini-msgs").appendChild(el);
    $("aria-mini-msgs").scrollTop = $("aria-mini-msgs").scrollHeight;
    return el;
  }

  function appendAssistantMsg(text) {
    var clean = String(text || "").trim();
    if (!clean) return null;
    var last = chatHistory.length ? chatHistory[chatHistory.length - 1] : null;
    if (last && last.role === "assistant" && String(last.content || "").trim() === clean) {
      return null;
    }
    var node = addMsg(clean, "bot");
    chatHistory.push({ role: "assistant", content: clean });
    return node;
  }

  var HINGLISH_MARKERS = ["acha", "achha", "abhi", "agar", "aap", "bolo", "chahta", "chahiye", "hai", "hain", "ho", "ka", "kaise", "kar", "karo", "karna", "kya", "kyu", "kyun", "mai", "main", "mujhe", "nahi", "peg", "robot", "samjha", "samjhao", "sahi", "toh"];

  function replyStyle(text) {
    var raw = (text || "").trim();
    if (!raw) return "english";
    if (/[\u0900-\u097f]/.test(raw)) return "hindi";
    var words = (raw.toLowerCase().match(/[a-z']+/g) || []);
    var hits = words.filter(function (word) { return HINGLISH_MARKERS.indexOf(word) >= 0; }).length;
    return hits >= 2 ? "hinglish" : "english";
  }

  function localText(style, english, hinglish) {
    return style === "hinglish" || style === "hindi" ? hinglish : english;
  }

  function buildClientFallback(text, outgoingAttachments) {
    var style = replyStyle(text);
    var lower = (text || "").trim().toLowerCase();
    var lines = [];
    if (/^(hi|hello|hey|start|good morning|good afternoon|good evening)\b/.test(lower)) {
      lines.push(localText(style, "Hello. ARIA AI is ready on " + pageName + ".", "Hello. ARIA AI " + pageName + " par ready hai."));
      lines.push(localText(style, "Ask anything normally. I will use simulation context only when you ask for it.", "Normal tareeke se kuch bhi poochho. Simulation context main sirf tab use karunga jab tum uske baare me poochoge."));
    }
    if (/status|state|doing|current|workbench|page/.test(lower)) {
      lines.push(localText(style, "Current page context:", "Current page context:"));
      lines.push("- Workbench: " + pageName);
      lines.push("- Path: " + window.location.pathname);
      lines.push("- Title: " + document.title);
    }
    if (/report|pdf|table|chart|graph|analysis|download/.test(lower)) {
      lines.push(localText(style, "This page supports simulation-aware guidance and report-related questions.", "Ye page simulation-aware guidance aur report-related questions support karta hai."));
      lines.push(localText(style, "- Attach a simulation shot if you want scene-aware help.", "- Agar scene-aware help chahiye to simulation shot attach karo."));
      lines.push(localText(style, "- Ask for tables, charts, or a report explanation in plain English.", "- Tables, charts, ya report explanation ke liye seedha pooch sakte ho."));
    }
    if (outgoingAttachments && outgoingAttachments.length) {
      lines.push(localText(style, "Attachment(s) noted: ", "Attachment noted: ") + outgoingAttachments.map(function (item) {
        return item.name || item.type || "attachment";
      }).join(", ") + ".");
    }
    if (!lines.length) {
      lines.push(localText(style, "ARIA AI is ready with the live page context on " + pageName + ".", "ARIA AI " + pageName + " ke live page context ke saath ready hai."));
      lines.push(localText(style, "Ask about this workbench, attach a simulation shot, or ask for a clear explanation.", "Is workbench ke bare me poochho, simulation shot attach karo, ya clear explanation maango."));
    }
    return lines.join("\n");
  }

  function clearMiniChatHistory() {
    chatHistory = [];
    attachments = [];
    setAttachmentNote();
    $("aria-mini-input").value = "";
    resizeInput();
    $("aria-mini-msgs").innerHTML = "";
    addMsg(
      "ARIA AI is ready on " + pageName + ". Ask in English or Hinglish. Use Sim Shot when you want scene-aware help.",
      "bot"
    );
  }

  function setOpen(open) {
    panel.classList.toggle("open", open);
    fab.classList.toggle("open", open);
    fab.innerHTML = open ? '<span class="aria-mini-fab-close">CLOSE</span>' : miniFabClosedMarkup;
    if (open) {
      setTimeout(function () {
        $("aria-mini-input").focus();
      }, 90);
    }
  }

  function resizeInput() {
    var input = $("aria-mini-input");
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 100) + "px";
  }

  function initPressFeedback() {
    function clearPressing() {
      document.querySelectorAll("#aria-mini-panel .pressing, #aria-mini-fab.pressing").forEach(function (el) {
        el.classList.remove("pressing");
      });
    }
    document.addEventListener("pointerdown", function (event) {
      var ctrl = event.target.closest("#aria-mini-fab, #aria-mini-close, #aria-mini-send, .aria-mini-btn, #aria-mini-provider");
      if (!ctrl) return;
      ctrl.classList.add("pressing");
    }, { passive: true });
    ["pointerup", "pointercancel", "touchend", "mouseup", "blur"].forEach(function (evt) {
      window.addEventListener(evt, clearPressing, { passive: true });
    });
  }

  function setAttachmentNote() {
    var host = $("aria-mini-attach");
    if (!attachments.length) {
      host.classList.remove("show");
      host.textContent = "";
      return;
    }
    host.classList.add("show");
    host.textContent = "Attached for next message: " + attachments.map(function (item) {
      return item.name;
    }).join(", ");
  }

  function setToolSupport(id, supported) {
    var button = $(id);
    if (!button) return;
    button.hidden = !supported;
    button.disabled = !supported;
  }

  function syncToolSupport() {
    setToolSupport("aria-mini-shot", true);
    setToolSupport("aria-mini-voice", true);
    setToolSupport("aria-mini-clear", true);
  }

  function updateProviderUi() {
    var providerName = defaultProvider || "groq";
    var info = (providers && providers[providerName]) || (providers && providers.groq);
    var aiReady = !!(info && info.configured);
    $("aria-mini-sub").textContent = aiReady
      ? "Groq-powered AI assistant"
      : "Live dashboard backup mode";
    $("aria-mini-note").textContent = aiReady
      ? (info.notes || "Groq AI is ready for this workbench.")
      : "Groq AI is not reachable right now. I will still answer directly when possible.";
  }

  function loadProviders() {
    fetch("/assistant/config")
      .then(function (r) { return r.json(); })
      .then(function (cfg) {
        providers = (cfg && cfg.providers) || {};
        defaultProvider = (cfg && cfg.default_provider) || "groq";
        updateProviderUi();
      })
      .catch(function () {
        providers = {};
        defaultProvider = "groq";
        updateProviderUi();
        addMsg("Assistant settings could not be refreshed. ARIA AI will still use the live page context on this workbench.", "bot");
      });
  }

  function parseAssistantResult(response) {
    return response.json().then(function (data) {
      return { ok: response.ok, data: data };
    });
  }

  function fetchAssistantReply(payload) {
    var options = {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    };
    var timeoutMs = (payload && payload.force_local) ? 6000 : 32000;
    if ("AbortController" in window) {
      var controller = new AbortController();
      var timer = setTimeout(function () { controller.abort(); }, timeoutMs);
      options.signal = controller.signal;
      return fetch("/assistant/chat", options)
        .then(function (response) {
          clearTimeout(timer);
          return parseAssistantResult(response);
        })
        .catch(function (err) {
          clearTimeout(timer);
          throw err;
        });
    }
    return fetch("/assistant/chat", options).then(parseAssistantResult);
  }

  function captureShotDataUrl() {
    return new Promise(function (resolve) {
      var img = document.querySelector(feedSelector);
      if (!img) return resolve(null);
      try {
        var canvas = document.createElement("canvas");
        canvas.width = img.naturalWidth || img.width || 640;
        canvas.height = img.naturalHeight || img.height || 360;
        canvas.getContext("2d").drawImage(img, 0, 0, canvas.width, canvas.height);
        resolve(canvas.toDataURL("image/jpeg", 0.88));
      } catch (err) {
        resolve(null);
      }
    });
  }

  function speakReply(text) {
    if (!speakingEnabled || !text || !("speechSynthesis" in window)) return;
    try {
      window.speechSynthesis.cancel();
      var utterance = new SpeechSynthesisUtterance(text);
      var locale = replyStyle(text) === "hindi" ? "hi-IN" : (replyStyle(text) === "hinglish" ? "en-IN" : "en-US");
      var voices = window.speechSynthesis.getVoices() || [];
      utterance.lang = locale;
      utterance.rate = 1.0;
      utterance.pitch = 1.0;
      utterance.voice = voices.find(function (voice) {
        return new RegExp(locale.replace("-", "[-_]?"), "i").test((voice.lang || "") + " " + (voice.name || ""));
      }) || voices.find(function (voice) {
        return /india|hindi/i.test((voice.lang || "") + " " + (voice.name || ""));
      }) || voices.find(function (voice) {
        return /en|hi/i.test(voice.lang || "");
      }) || null;
      window.speechSynthesis.speak(utterance);
    } catch (err) {
      // Ignore speech errors and keep chat responsive.
    }
  }

  function toggleVoice() {
    speakingEnabled = !speakingEnabled;
    $("aria-mini-voice").classList.toggle("active", speakingEnabled);
    $("aria-mini-voice").textContent = speakingEnabled ? "Voice On" : "Voice Off";
    if (!speakingEnabled && "speechSynthesis" in window) {
      try { window.speechSynthesis.cancel(); } catch (err) {}
    }
  }

  function send() {
    var input = $("aria-mini-input");
    var text = input.value.trim();
    if (!text) return;
    var historyPayload = chatHistory.slice(-8).map(function (item) {
      return { role: item.role, content: item.content };
    });
    var outgoingAttachments = attachments.slice();
    attachments = [];
    setAttachmentNote();
    input.value = "";
    resizeInput();
    addMsg(text, "user");
    chatHistory.push({ role: "user", content: text });
    var thinking = addMsg("Assistant is thinking...", "bot");
    $("aria-mini-send").disabled = true;

    var requestPayload = {
      message: text,
      history: historyPayload,
      attachments: outgoingAttachments,
      page_context: Object.assign(
        {
          page_name: pageName,
          path: window.location.pathname,
          title: document.title
        },
        pageContext
      )
    };

    fetchAssistantReply(requestPayload)
      .catch(function () {
        return {
          ok: true,
          data: {
            reply: buildClientFallback(text, outgoingAttachments),
            mode: "browser_local"
          }
        };
      })
      .then(function (result) {
        $("aria-mini-send").disabled = false;
        if (thinking && thinking.parentNode) thinking.parentNode.removeChild(thinking);
        var data = result.data || {};
        var reply = data.reply || "";
        if (!result.ok || data.error || !reply) {
          reply = buildClientFallback(text, outgoingAttachments);
        }
        if (appendAssistantMsg(reply)) {
          speakReply(reply);
        }
      })
      .catch(function () {
        $("aria-mini-send").disabled = false;
        if (thinking && thinking.parentNode) thinking.parentNode.removeChild(thinking);
        var reply = buildClientFallback(text, outgoingAttachments);
        if (appendAssistantMsg(reply)) {
          speakReply(reply);
        }
      });
  }

  fab.addEventListener("click", function () {
    setOpen(!panel.classList.contains("open"));
  });

  $("aria-mini-close").addEventListener("click", function () {
    setOpen(false);
  });

  $("aria-mini-send").addEventListener("click", send);
  $("aria-mini-input").addEventListener("input", resizeInput);
  $("aria-mini-input").addEventListener("keydown", function (event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      send();
    }
  });
  $("aria-mini-shot").addEventListener("click", function () {
    captureShotDataUrl().then(function (dataUrl) {
      if (!dataUrl) {
        addMsg("Simulation snapshot could not be captured on this page.", "bot");
        return;
      }
      attachments = [{
        type: "simulation_snapshot",
        name: pageName + " snapshot",
        data_url: dataUrl
      }];
      setAttachmentNote();
      addMsg("Simulation snapshot attached from " + pageName + ".", "bot");
    });
  });
  $("aria-mini-voice").addEventListener("click", toggleVoice);
  $("aria-mini-clear").addEventListener("click", clearMiniChatHistory);

  document.addEventListener("keydown", function (event) {
    if (event.key === "Escape") setOpen(false);
  });

  document.addEventListener("click", function (event) {
    if (!panel.classList.contains("open")) return;
    if (panel.contains(event.target) || fab.contains(event.target)) return;
    setOpen(false);
  });

  addMsg(
    "ARIA AI is ready on " + pageName + ". Ask in English or Hinglish. Use Sim Shot when you want scene-aware help.",
    "bot"
  );
  initPressFeedback();
  syncToolSupport();
  loadProviders();
})();
