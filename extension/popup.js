// popup.js — TrustScan popup controller
// Communicates with content script to display live scan stats

const API_URL = "http://localhost:8000";

// ── DOM refs ───────────────────────────────────────────────
const domainLabel    = () => document.getElementById("domain-label");
const apiStatus      = () => document.getElementById("api-status");
const scannerStatus  = () => document.getElementById("scanner-status");
const summaryCard    = () => document.getElementById("summary-card");
const countReal      = () => document.getElementById("count-real");
const countFake      = () => document.getElementById("count-fake");
const countUncertain = () => document.getElementById("count-uncertain");

// ── Helpers ────────────────────────────────────────────────
function dot(color) {
    return `<span class="dot dot-${color}"></span>`;
}

async function checkApi() {
    const el = apiStatus();
    try {
        const res = await fetch(`${API_URL}/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: "test" }),
            signal: AbortSignal.timeout(3000),
        });
        if (res.ok) {
            el.innerHTML = `${dot("green")}Connected`;
        } else {
            el.innerHTML = `${dot("red")}Error ${res.status}`;
        }
    } catch {
        el.innerHTML = `${dot("red")}Offline`;
    }
}

function detectDomain(url) {
    const el = domainLabel();
    if (!url) { el.textContent = "No page"; return false; }
    if (url.includes("amazon.")) {
        const m = url.match(/amazon\.[a-z.]+/);
        el.textContent = m ? m[0] : "Amazon";
        el.style.color = "#10b981";
        return true;
    }
    el.textContent = "Not Amazon";
    el.style.color = "#ef4444";
    scannerStatus().innerHTML = `${dot("amber")}Idle`;
    return false;
}

function showCounts(real, fake, uncertain) {
    countReal().textContent      = real;
    countFake().textContent      = fake;
    countUncertain().textContent = uncertain;
    summaryCard().style.display  = "block";
}

// ── Get counts from content script ─────────────────────────
function fetchCounts(tabId) {
    chrome.tabs.sendMessage(tabId, { action: "getCounts" }, (response) => {
        if (chrome.runtime.lastError || !response) return;
        showCounts(response.real || 0, response.fake || 0, response.uncertain || 0);
    });
}

// ── Init ───────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    checkApi();

    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        const tab = tabs[0];
        const isAmazon = detectDomain(tab.url);
        if (isAmazon) fetchCounts(tab.id);
    });

    // ── Rescan button ──────────────────────────────────────
    document.getElementById("rescan-btn").addEventListener("click", () => {
        const btn = document.getElementById("rescan-btn");
        btn.textContent = "⏳ Scanning…";
        btn.disabled = true;

        chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
            chrome.tabs.sendMessage(tabs[0].id, { action: "rescan" }, (response) => {
                if (chrome.runtime.lastError || !response) {
                    // Fallback: reload the page
                    chrome.tabs.reload(tabs[0].id);
                    window.close();
                    return;
                }
                setTimeout(() => {
                    fetchCounts(tabs[0].id);
                    btn.innerHTML = "🔄&nbsp; Rescan Page";
                    btn.disabled = false;
                }, 1500);
            });
        });
    });

    // ── Test button ────────────────────────────────────────
    document.getElementById("test-btn").addEventListener("click", async () => {
        const btn = document.getElementById("test-btn");
        btn.textContent = "⏳ Testing…";
        btn.disabled = true;

        const sample = "This product is the best thing ever!!! I love it so much you should definitely buy it NOW!!! Amazing amazing amazing!!!";
        try {
            const res = await fetch(`${API_URL}/predict`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text: sample }),
            });
            const data = await res.json();
            const lbl = data.label || "Unknown";
            const conf = data.confidence != null ? `${(data.confidence * 100).toFixed(0)}%` : "";
            btn.innerHTML = `Result: <b>${lbl}</b> ${conf}`;
            btn.style.color = lbl === "Fake" ? "#ef4444" : lbl === "Real" ? "#10b981" : "#f59e0b";
        } catch {
            btn.textContent = "❌ API unreachable";
            btn.style.color = "#ef4444";
        }
        setTimeout(() => {
            btn.innerHTML = "🧪&nbsp; Test with Sample Review";
            btn.style.color = "";
            btn.disabled = false;
        }, 3000);
    });
});
