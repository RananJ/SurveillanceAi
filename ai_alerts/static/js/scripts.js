document.addEventListener("DOMContentLoaded", function () {
  // API Endpoints
  const ALERTS_API_URL = "/api/alerts/";
  const TRANSCRIPTS_API_URL = "/api/transcripts/";

  // Get references to DOM elements
  const alertsContainer = document.getElementById("alerts-container");
  const detailPlaceholder = document.getElementById("detail-placeholder");
  const detailContent = document.getElementById("detail-content");
  const videoPlayer = document.getElementById("video-player");
  const detailTimestamp = document.getElementById("detail-timestamp");
  const detailCamera = document.getElementById("detail-camera");
  const detailViolations = document.getElementById("detail-violations");
  const detailSummary = document.getElementById("detail-summary");

  // Load and display data when the page is ready
  loadData();

  async function loadData() {
    try {
      const [alertsResponse, transcriptsResponse] = await Promise.all([
        fetch(ALERTS_API_URL),
        fetch(TRANSCRIPTS_API_URL),
      ]);

      const alerts = await alertsResponse.json();
      const transcripts = await transcriptsResponse.json();

      const transcriptsByAlertId = new Map(
        transcripts.map((t) => [t.alert, t])
      );

      const combinedData = alerts.map((alert) => ({
        ...alert,
        summary:
          transcriptsByAlertId.get(alert.id)?.summary ||
          "No summary available.",
      }));

      // --- DEBUGGING LINE 1 ---
      // Let's see if the data was combined correctly.
      console.log("Combined data loaded:", combinedData);

      populateAlertFeed(combinedData);
    } catch (error) {
      console.error("Failed to load data:", error);
      alertsContainer.innerHTML =
        '<p class="placeholder">Error loading data.</p>';
    }
  }

  function populateAlertFeed(data) {
    if (data.length === 0) {
      alertsContainer.innerHTML = '<p class="placeholder">No alerts found.</p>';
      return;
    }

    alertsContainer.innerHTML = "";

    data.forEach((alert) => {
      const item = document.createElement("div");
      item.className = "alert-item";
      item.dataset.alertId = alert.id;

      const timestamp = new Date(alert.timestamp).toLocaleString();

      item.innerHTML = `
                <p><strong>${alert.violations}</strong></p>
                <span class="timestamp">${timestamp}</span>
            `;

      item.addEventListener("click", () => {
        // --- DEBUGGING LINE 2 ---
        // This will show us the exact data for the item you clicked.
        console.log("Clicked alert object:", alert);

        displayAlertDetails(alert);
        document
          .querySelectorAll(".alert-item")
          .forEach((el) => el.classList.remove("selected"));
        item.classList.add("selected");
      });

      alertsContainer.appendChild(item);
    });
  }

  function displayAlertDetails(alert) {
    detailPlaceholder.classList.add("hidden");
    detailContent.classList.remove("hidden");

    videoPlayer.src = alert.video_url;
    detailTimestamp.textContent = new Date(alert.timestamp).toLocaleString();
    detailCamera.textContent = alert.camera_id;
    detailViolations.textContent = alert.violations;
    detailSummary.textContent = alert.summary;
  }
});
