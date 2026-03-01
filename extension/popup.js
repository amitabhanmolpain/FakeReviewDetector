// popup.js: Handles the TrustScan popup interface interactions

document.addEventListener('DOMContentLoaded', () => {
    // Detect current domain
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        const url = tabs[0].url;
        const domainLabel = document.getElementById('domain-label');

        if (url.includes('amazon.')) {
            const domainMatch = url.match(/amazon\.[a-z.]+/);
            domainLabel.textContent = domainMatch ? domainMatch[0] : 'Amazon';
            domainLabel.style.color = '#10b981';
        } else {
            domainLabel.textContent = 'Amazon Only (Current)';
            domainLabel.style.color = '#ef4444';
        }
    });

    // Handle re-scan button
    document.getElementById('rescan-btn').addEventListener('click', () => {
        chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
            // Re-inject/Trigger scanning in the content script
            chrome.scripting.executeScript({
                target: { tabId: tabs[0].id },
                func: () => {
                    // This logic assumes we can trigger a re-scan.
                    // We'll call the function defined in content.js if possible,
                    // or just refresh for now as a simple implementation.
                    location.reload();
                }
            });
            window.close(); // Close popup after action
        });
    });
});
