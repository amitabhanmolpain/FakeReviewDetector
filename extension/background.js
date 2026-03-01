// background.js: Background service worker for TrustScan (Backend Integrated)

const API_URL = "http://localhost:8000/predict";

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "classifyReviews") {
        const reviews = request.reviews;

        const predictReview = async (reviewText) => {
            try {
                const response = await fetch(API_URL, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ text: reviewText }),
                });

                if (!response.ok) {
                    throw new Error(`Backend returned ${response.status}`);
                }

                const data = await response.json();
                // API returns: label ("Fake"/"Real"/"Uncertain"),
                // fake_prob, combined_fake_score, confidence, heuristics
                return {
                    label: data.label,                       // "Fake" | "Real" | "Uncertain"
                    confidence: data.confidence,
                    fakeProbability: data.fake_prob,
                    combinedScore: data.combined_fake_score,
                    heuristics: data.heuristics || {},
                };
            } catch (err) {
                console.error("TrustScan prediction error:", err);
                return { label: "Error", confidence: 0, fakeProbability: 0, combinedScore: 0, heuristics: {} };
            }
        };

        Promise.all(reviews.map(predictReview))
            .then((predictionResults) => {
                sendResponse({ results: predictionResults });
            })
            .catch((error) => {
                console.error("TrustScan batch error:", error);
                sendResponse({ results: [], error: error.message });
            });

        return true; // Keep message channel open for async response
    }
});
