const statusTitle = document.querySelector("#status-title");
const statusMessage = document.querySelector("#status-message");
const statusDetails = document.querySelector("#status-details");
const statusDot = document.querySelector("#status-dot");
const chooseButton = document.querySelector("#choose-root");
const retryButton = document.querySelector("#retry");
const logButton = document.querySelector("#open-log");

function renderStatus(payload = {}) {
  const phase = payload.phase ?? "starting";
  document.body.dataset.phase = phase;
  statusDot.setAttribute("aria-label", phase);
  statusTitle.textContent = payload.title ?? "Apertus Workbench";
  statusMessage.textContent = payload.message ?? "";
  statusDetails.replaceChildren();
  for (const detail of payload.details ?? []) {
    const item = document.createElement("li");
    item.textContent = detail;
    statusDetails.append(item);
  }
  const failed = phase === "error";
  chooseButton.hidden = !failed;
  retryButton.hidden = !failed;
  logButton.hidden = !failed;
}

window.apertusDesktop?.onStatus(renderStatus);
window.apertusDesktop?.onStatusHistory((history) => {
  if (Array.isArray(history) && history.length) renderStatus(history.at(-1));
});

chooseButton.addEventListener("click", async () => {
  chooseButton.disabled = true;
  try {
    const result = await window.apertusDesktop.chooseApertusRoot();
    if (result && result.ok === false) {
      renderStatus({
        phase: "error",
        title: "Vybraná složka není úplná",
        message: "Chybí server nebo modelové soubory.",
        details: result.problems ?? [],
      });
    }
  } finally {
    chooseButton.disabled = false;
  }
});

retryButton.addEventListener("click", () => window.apertusDesktop.retry());
logButton.addEventListener("click", () => window.apertusDesktop.openLog());
