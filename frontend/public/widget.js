(function () {
  "use strict";

  var script =
    document.currentScript ||
    (function () {
      var scripts = document.getElementsByTagName("script");
      return scripts[scripts.length - 1];
    })();

  var companyId = (script.getAttribute("data-company-id") || "").trim().toLowerCase();
  var title = script.getAttribute("data-title") || "Ask Company AI";
  var position = (script.getAttribute("data-position") || "bottom-right").trim();
  var baseUrl =
    script.getAttribute("data-base-url") ||
    (script.src ? new URL(script.src).origin : window.location.origin);

  if (!/^[a-z0-9_-]{1,64}$/.test(companyId)) {
    console.error("[SeiraiWidget] Invalid or missing data-company-id.");
    return;
  }

  var host = document.createElement("div");
  host.id = "seirai-chatbot-widget-host";
  host.style.all = "initial";
  host.style.position = "fixed";
  host.style.zIndex = "2147483646";
  host.style.bottom = "20px";
  host.style.right = position.indexOf("left") >= 0 ? "auto" : "20px";
  host.style.left = position.indexOf("left") >= 0 ? "20px" : "auto";
  document.body.appendChild(host);

  var shadow = host.attachShadow({ mode: "open" });
  var style = document.createElement("style");
  style.textContent =
    ".btn{all:initial;font-family:system-ui,sans-serif;background:#0f172a;color:#fff;border-radius:999px;padding:14px 18px;cursor:pointer;box-shadow:0 10px 30px rgba(15,23,42,.25);font-size:14px;font-weight:600;display:inline-flex;align-items:center;gap:8px}" +
    ".btn:focus{outline:2px solid #0891b2;outline-offset:3px}" +
    ".panel{all:initial;position:fixed;bottom:84px;width:min(400px,calc(100vw - 24px));height:min(640px,calc(100vh - 120px));border:0;border-radius:18px;overflow:hidden;box-shadow:0 20px 50px rgba(15,23,42,.35);background:#fff;display:none}" +
    ".panel.open{display:block}" +
    ".panel.left{left:20px;right:auto}.panel.right{right:20px;left:auto}" +
    "iframe{border:0;width:100%;height:100%}";
  shadow.appendChild(style);

  var button = document.createElement("button");
  button.className = "btn";
  button.type = "button";
  button.setAttribute("aria-label", title);
  button.textContent = title;
  shadow.appendChild(button);

  var panel = document.createElement("div");
  panel.className = "panel " + (position.indexOf("left") >= 0 ? "left" : "right");
  panel.setAttribute("role", "dialog");
  panel.setAttribute("aria-label", title);
  var iframe = document.createElement("iframe");
  iframe.title = title;
  iframe.src =
    baseUrl.replace(/\/$/, "") +
    "/embed/" +
    encodeURIComponent(companyId);
  iframe.allow = "clipboard-write";
  panel.appendChild(iframe);
  shadow.appendChild(panel);

  button.addEventListener("click", function () {
    var open = panel.classList.toggle("open");
    button.setAttribute("aria-expanded", open ? "true" : "false");
  });

  window.SeiraiChatbotWidget = {
    status: "ready",
    companyId: companyId,
    open: function () {
      panel.classList.add("open");
    },
    close: function () {
      panel.classList.remove("open");
    },
  };
})();
