// content.js: Scans Amazon reviews and injects modern fake/genuine/uncertain indicators

(() => {
    const SELECTORS = {
        reviewBlock: '[data-hook="review"]',
        reviewBody: '[data-hook="review-body"]',
    };

    /* ── Extract unprocessed reviews ─────────────────────────────── */
    function extractReviews() {
        const blocks = document.querySelectorAll(SELECTORS.reviewBlock);
        const reviews = [];

        blocks.forEach((block, idx) => {
            if (block.hasAttribute("data-trustscan-processed")) return;
            const body = block.querySelector(SELECTORS.reviewBody);
            if (body) {
                reviews.push({ id: idx, text: body.textContent.trim(), element: block });
            }
        });

        return reviews;
    }

    /* ── Confidence bar (mini gauge) ─────────────────────────────── */
    function confidenceBar(score, color) {
        return `
            <div class="ts-bar-track">
                <div class="ts-bar-fill" style="width:${Math.round(score * 100)}%;background:${color}"></div>
            </div>`;
    }

    /* ── Build & inject badge ────────────────────────────────────── */
    function injectIndicator(block, result) {
        if (block.hasAttribute("data-trustscan-processed")) return;

        const label = result.label;          // Fake | Real | Uncertain | Error
        const conf  = result.confidence ?? 0;
        const score = result.combinedScore ?? result.fakeProbability ?? 0;

        let themeClass, icon, labelText, sublabel, barColor;

        switch (label) {
            case "Fake":
                themeClass = "ts-fake";
                icon = "⚠️";
                labelText = "Likely Fake";
                sublabel = `${Math.round(score * 100)}% fake score`;
                barColor = "#ef4444";
                break;
            case "Real":
                themeClass = "ts-genuine";
                icon = "✅";
                labelText = "Looks Genuine";
                sublabel = `${Math.round(conf * 100)}% confidence`;
                barColor = "#10b981";
                break;
            case "Uncertain":
                themeClass = "ts-uncertain";
                icon = "🔍";
                labelText = "Uncertain";
                sublabel = `${Math.round(score * 100)}% fake score — needs manual check`;
                barColor = "#f59e0b";
                break;
            default:
                // Error / offline
                themeClass = "ts-error";
                icon = "🛡️";
                labelText = "Scan Unavailable";
                sublabel = "Backend offline";
                barColor = "#94a3b8";
        }

        const badge = document.createElement("div");
        badge.className = `trust-scan-badge ${themeClass}`;
        badge.innerHTML = `
            <span class="ts-icon">${icon}</span>
            <div class="ts-content">
                <span class="ts-label">${labelText}</span>
                <span class="ts-sub">${sublabel}</span>
                ${label !== "Error" ? confidenceBar(score, barColor) : ""}
            </div>
        `;

        // Show heuristic chips for Fake / Uncertain
        if ((label === "Fake" || label === "Uncertain") && result.heuristics) {
            const h = result.heuristics;
            const chips = [];
            if (h.promo_phrase)                         chips.push("Promo phrases");
            if (h.short_review)                         chips.push("Very short");
            if (h.repeated_word_ratio > 0.3)            chips.push("Repetitive");
            if (h.exclamation_ratio > 0.02)             chips.push("Exclamation heavy");
            if (h.uppercase_ratio > 0.15)               chips.push("CAPS heavy");

            if (chips.length) {
                const chipHtml = chips.map(c => `<span class="ts-chip">${c}</span>`).join("");
                const row = document.createElement("div");
                row.className = "ts-chips";
                row.innerHTML = chipHtml;
                badge.querySelector(".ts-content").appendChild(row);
            }
        }

        const bodySection = block.querySelector(SELECTORS.reviewBody);
        if (bodySection) {
            bodySection.prepend(badge);
        } else {
            block.appendChild(badge);
        }

        block.setAttribute("data-trustscan-processed", "true");
    }

    /* ── Send reviews to background → API ────────────────────────── */
    function processReviews() {
        const reviewsData = extractReviews();
        if (reviewsData.length === 0) return;

        // Inject loading shimmer while waiting
        reviewsData.forEach((r) => {
            if (!r.element.hasAttribute("data-trustscan-processed")) {
                const shimmer = document.createElement("div");
                shimmer.className = "trust-scan-badge ts-loading";
                shimmer.innerHTML = `<span class="ts-icon">🔄</span>
                    <div class="ts-content"><span class="ts-label">Scanning…</span></div>`;
                const body = r.element.querySelector(SELECTORS.reviewBody);
                if (body) body.prepend(shimmer);
            }
        });

        chrome.runtime.sendMessage(
            { action: "classifyReviews", reviews: reviewsData.map((r) => r.text) },
            (response) => {
                // Remove shimmers
                document.querySelectorAll(".ts-loading").forEach((el) => el.remove());

                if (response && response.results) {
                    response.results.forEach((res, idx) => {
                        if (reviewsData[idx]) {
                            injectIndicator(reviewsData[idx].element, res);
                        }
                    });
                }
            }
        );
    }

    /* ── Message listener (popup ↔ content) ─────────────────────── */
    chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
        if (msg.action === "getCounts") {
            const badges = document.querySelectorAll(".trust-scan-badge:not(.ts-loading)");
            let real = 0, fake = 0, uncertain = 0;
            badges.forEach(b => {
                if (b.classList.contains("ts-genuine"))   real++;
                else if (b.classList.contains("ts-fake")) fake++;
                else if (b.classList.contains("ts-uncertain")) uncertain++;
            });
            sendResponse({ real, fake, uncertain });
            return true;
        }
        if (msg.action === "rescan") {
            document.querySelectorAll("[data-trustscan-processed]").forEach(el => {
                el.removeAttribute("data-trustscan-processed");
                el.querySelectorAll(".trust-scan-badge").forEach(b => b.remove());
            });
            processReviews();
            sendResponse({ ok: true });
            return true;
        }
    });

    /* ── Boot ─────────────────────────────────────────────────────── */
    console.log("TrustScan Extension: Scanning for reviews…");
    processReviews();

    // Re-scan when Amazon lazy-loads more reviews
    let lastCount = 0;
    const observer = new MutationObserver(() => {
        const cur = document.querySelectorAll(SELECTORS.reviewBlock).length;
        if (cur > lastCount) {
            lastCount = cur;
            processReviews();
        }
    });
    observer.observe(document.body, { childList: true, subtree: true });
})();
