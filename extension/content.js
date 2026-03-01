// content.js: Scans Amazon reviews and injects fake/genuine indicators

(() => {
    // Selectors for Amazon
    const selectors = {
        reviewBlock: '[data-hook="review"]',
        reviewBody: '[data-hook="review-body"]'
    };

    /**
     * Extracts all reviews from the current page
     */
    function extractReviews() {
        const blocks = document.querySelectorAll(selectors.reviewBlock);
        const reviewsData = [];

        blocks.forEach((block, index) => {
            // Avoid re-processing if already scanned
            if (block.hasAttribute('data-trustscan-processed')) return;

            const body = block.querySelector(selectors.reviewBody);
            if (body) {
                // Keep only text content
                const text = body.textContent.trim();
                reviewsData.push({
                    id: index,
                    text: text,
                    element: block
                });
            }
        });

        return reviewsData;
    }

    /**
     * Injects a modern indicator badge into the review block
     */
    function injectIndicator(block, result) {
        if (block.hasAttribute('data-trustscan-processed')) return;

        // If backend returned error (ML server down)
        if (result.label === "ERROR") {
            const errorBadge = document.createElement('div');
            errorBadge.className = 'trust-scan-badge ts-error';
            errorBadge.innerHTML = '🛡️ <span class="ts-label">Backend is Offline</span>';
            block.querySelector(selectors.reviewBody)?.prepend(errorBadge);
            return;
        }

        const isFake = result.label === "FAKE";
        const badge = document.createElement('div');
        badge.className = `trust-scan-badge ${isFake ? 'ts-fake' : 'ts-genuine'}`;

        const icon = isFake ? '⚠️' : '✅';
        const labelText = isFake ? 'Fake Review (CG)' : 'Genuine Review (OR)';

        badge.innerHTML = `
            <span class="ts-icon">${icon}</span>
            <div class="ts-content">
                <span class="ts-label">${labelText}</span>
                <span class="ts-sub">ML Analysis Verified</span>
            </div>
        `;

        // Find a good place to inject. Amazon's review structure usually has a header.
        const bodySection = block.querySelector(selectors.reviewBody);
        if (bodySection) {
            bodySection.prepend(badge);
        } else {
            block.appendChild(badge);
        }

        // Mark as processed
        block.setAttribute('data-trustscan-processed', 'true');
    }

    /**
     * Sends the extracted reviews to the background script for classification
     */
    function processReviews() {
        const reviewsData = extractReviews();
        if (reviewsData.length === 0) return;

        const reviewTexts = reviewsData.map(r => r.text);

        // Send to background
        chrome.runtime.sendMessage({
            action: "classifyReviews",
            reviews: reviewTexts
        }, (response) => {
            if (response && response.results) {
                response.results.forEach((res, idx) => {
                    if (reviewsData[idx]) {
                        injectIndicator(reviewsData[idx].element, res);
                    }
                });
            }
        });
    }

    // Run automatically when the page is ready
    console.log("TrustScan Extension: Scanning for reviews...");
    processReviews();

    // Re-scan periodically as Amazon loads reviews via AJAX (lazy loading/endless scroll)
    let lastScanCount = 0;
    const observer = new MutationObserver((mutations) => {
        const currentCount = document.querySelectorAll(selectors.reviewBlock).length;
        if (currentCount > lastScanCount) {
            lastScanCount = currentCount;
            processReviews();
        }
    });

    observer.observe(document.body, { childList: true, subtree: true });

})();
