// background.js: Background service worker for TrustScan (Backend Integrated)

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "classifyReviews") {
        const reviews = request.reviews;
        const results = [];

        // Function to call the FastAPI backend for each review
        const predictReview = async (reviewText) => {
            try {
                const response = await fetch('http://localhost:8000/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ review: reviewText })
                });

                if (!response.ok) {
                    throw new Error("Backend connection failed.");
                }

                const data = await response.json();
                return {
                    label: data.prediction === "CG" ? "FAKE" : "GENUINE",
                    confidence: 1.0 // Future: get confidence from ML model
                };
            } catch (err) {
                console.error("Prediction error:", err);
                // Fallback to mock on error or notify UI
                return { label: "ERROR", confidence: 0 };
            }
        };

        // Process all reviews sequentially or in parallel
        Promise.all(reviews.map(predictReview))
            .then(predictionResults => {
                sendResponse({ results: predictionResults });
            })
            .catch(error => {
                console.error("Error processing reviews:", error);
                sendResponse({ results: [], error: error.message });
            });

        return true; // Keep message channel open for async response
    }
});
