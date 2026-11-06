(async () => {
  "use strict";

  const body = document.body;
  const page = body?.dataset.workbenchPage;
  if (!body || !page || body.dataset.workbenchShellReady === "true") return;
  const i18n = window.ProbabilityGardenI18n;
  const t = i18n?.t ?? ((value) => String(value ?? ""));
  if (!window.ProbabilityGardenApi) {
    await new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = "/static/api-client.js";
      script.onload = resolve;
      script.onerror = () => reject(
        new Error("Sdílený API klient se nepodařilo načíst.")
      );
      document.head.append(script);
    });
  }
  const api = window.ProbabilityGardenApi;
  if (!api) {
    throw new Error("Chybí sdílený klient static/api-client.js.");
  }

  const pages = [
    {
      id: "tokenization",
      href: "/tokenizace",
      label: "Tokenizace",
      short: "Tokenizace",
      meta: "vstupní jednotky"
    },
    {
      id: "embeddings",
      href: "/embeddingy",
      label: "Embeddingy",
      short: "Embeddingy",
      meta: "výchozí geometrie"
    },
    {
      id: "garden",
      href: "/",
      label: "Pravděpodobnostní zahrada",
      short: "Zahrada",
      meta: "živé logity"
    },
    {
      id: "surprise",
      href: "/prekvapeni",
      label: "Překvapení textu",
      short: "Překvapení",
      meta: "hotový text"
    },
    {
      id: "detector",
      href: "/detektor",
      label: "AI detektor",
      short: "AI detektor",
      meta: "stylometrie × Apertus"
    },
    {
      id: "contrast",
      href: "/kontrast",
      label: "Kontrast kontextů",
      short: "Kontrast",
      meta: "dvě pole"
    },
    {
      id: "model-comparison",
      href: "/porovnani-modelu",
      label: "Porovnání modelů",
      short: "Modely",
      meta: "dva modeloví čtenáři"
    },
    {
      id: "raw",
      href: "/vrstvy",
      label: "RAW vrstvy",
      short: "RAW vrstvy",
      meta: "RAW logit lens"
    },
    {
      id: "tuned",
      href: "/tuned-lens",
      label: "RAW × TUNED",
      short: "RAW × TUNED",
      meta: "kalibrovaná čtečka"
    },
    {
      id: "jlens",
      href: "/j-lens",
      label: "RAW × J-LENS",
      short: "RAW × J-LENS",
      meta: "průměrný Jacobian"
    },
    {
      id: "attribution",
      href: "/atribuce",
      label: "Preference",
      short: "Preference",
      meta: "příspěvky vrstev"
    },
    {
      id: "patching",
      href: "/kauzalni-prenos",
      label: "Kauzální přenos",
      short: "Patching",
      meta: "stav A → průchod B"
    },
    {
      id: "steering",
      href: "/steering",
      label: "Steering",
      short: "Steering",
      meta: "kauzální zásah"
    },
    {
      id: "guide",
      href: "/jak-to-cist",
      label: "Průvodce workbenchem",
      short: "Průvodce",
      meta: "mapa metod"
    },
    {
      id: "archive",
      href: "/archiv",
      label: "Archiv měření",
      short: "Archiv",
      meta: "lokální záznamy"
    },
    {
      id: "model-admin",
      href: "/sprava-modelu",
      label: "Správa modelů",
      short: "Správa modelů",
      meta: "instalace a nastavení"
    }
  ];

  const stageDefinitions = [
    {
      id: "direction",
      href: "#stage-direction",
      label: "Směr",
      meta: "kontrasty a zmrazený zdroj"
    },
    {
      id: "immediate",
      href: "#stage-immediate",
      label: "Okamžitý efekt",
      meta: "baseline × steered"
    },
    {
      id: "trajectory",
      href: "#trajectory-section",
      label: "Cesty",
      meta: "tři vlastní kontexty"
    },
    {
      id: "held-out",
      href: "#validation-section",
      label: "Kontrolované pokračování",
      meta: "společná cesta a matched páry"
    },
    {
      id: "sham",
      href: "#sham-control-tool",
      label: "Specificita",
      meta: "norm-matched sham reference"
    },
    {
      id: "sweep",
      href: "#layer-sweep-tool",
      label: "Lokalizace",
      meta: "lokální směry po vrstvách"
    }
  ];

  const pageMeta = pages.find((item) => item.id === page) ?? pages[0];
  const root = (
    body.querySelector(":scope > main.shell") ??
    body.querySelector(":scope > .shell") ??
    body.querySelector(":scope > main")
  );
  if (!root) return;
  const rootParent = root.parentNode;

  const element = (tag, className, text = "") => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text) node.textContent = t(text);
    return node;
  };

  const frame = element("div", "pg-app-frame");
  const contextBar = element("header", "pg-contextbar");
  const frameBody = element("div", "pg-app-body");
  const rail = element("aside", "pg-global-rail");
  const content = element("div", "pg-content");
  root.classList.add("pg-content-root");

  const brand = document.createElement("a");
  brand.className = "pg-brand";
  brand.href = "/";
  brand.setAttribute("aria-label", "Pravděpodobnostní zahrada – domů");
  brand.innerHTML = `
    <span class="pg-brand-dot" aria-hidden="true"></span>
    <span>
      <strong>Pravděpodobnostní zahrada</strong>
      <small>Výzkumný workbench</small>
    </span>
  `;

  const context = element("div", "pg-context");
  context.innerHTML = `
    <div class="pg-context-item pg-context-page">
      <span>Pohled</span>
      <strong data-pg-context="page">${pageMeta.label}</strong>
    </div>
    <div class="pg-context-item pg-context-model" data-pg-context-item="model">
      <span>Model</span>
      <span class="pg-model-slot" data-pg-model-slot></span>
    </div>
    <div class="pg-context-item pg-context-secondary" data-pg-context-item="run">
      <span>Běh</span>
      <strong data-pg-context="run">lokální</strong>
    </div>
    <div class="pg-context-item pg-context-steering" data-pg-context-item="pairs">
      <span>Platné páry</span>
      <strong data-pg-context="pairs">—</strong>
    </div>
    <div class="pg-context-item pg-context-steering" data-pg-context-item="layer">
      <span>Vrstva</span>
      <strong data-pg-context="layer">—</strong>
    </div>
    <div class="pg-context-item pg-context-steering" data-pg-context-item="alpha">
      <span>|α|</span>
      <strong data-pg-context="alpha">—</strong>
    </div>
    <div class="pg-context-item pg-context-prompt" data-pg-context-item="prompt">
      <span>Aktivní prefix</span>
      <strong data-pg-context="prompt">—</strong>
    </div>
  `;

  const status = element("div", "pg-context-status");
  status.dataset.state = "idle";
  status.setAttribute("role", "status");
  status.setAttribute("aria-live", "polite");
  status.innerHTML = `
    <span class="pg-context-status-dot" aria-hidden="true"></span>
    <span data-pg-context="status">Připravuji rozhraní</span>
  `;

  const contextActions = element("div", "pg-context-actions");
  const languageSwitch = element("div", "pg-language-switch");
  languageSwitch.setAttribute("role", "group");
  languageSwitch.setAttribute("aria-label", t("Jazyk rozhraní"));
  for (const language of [
    {id: "cs", label: "CS", title: "Česky"},
    {id: "en", label: "EN", title: "English"}
  ]) {
    const button = element("button", "pg-language-button", language.label);
    button.type = "button";
    button.dataset.language = language.id;
    button.title = t(language.title);
    button.setAttribute(
      "aria-pressed",
      String((i18n?.language ?? "cs") === language.id)
    );
    button.addEventListener("click", () => {
      if ((i18n?.language ?? "cs") === language.id) return;
      i18n?.setLanguage(language.id);
      const url = new URL(window.location.href);
      url.searchParams.delete("lang");
      window.location.replace(url.href);
    });
    languageSwitch.append(button);
  }
  contextActions.append(languageSwitch, status);
  let primaryAction = null;
  if (page === "steering") {
    primaryAction = element(
      "button",
      "pg-context-action",
      "Spustit řez"
    );
    primaryAction.type = "button";
    primaryAction.disabled = true;
    contextActions.append(primaryAction);
  }

  contextBar.append(brand, context, contextActions);

  const viewNav = element("nav", "pg-view-nav");
  viewNav.setAttribute("aria-label", "Pohledy aplikace");
  const viewLabel = element("div", "pg-rail-label", "Pohledy");
  viewNav.append(viewLabel);
  pages.forEach((item) => {
    const link = document.createElement("a");
    link.className = "pg-view-link";
    link.href = item.href;
    link.dataset.page = item.id;
    if (item.id === page) link.setAttribute("aria-current", "page");
    link.innerHTML = `
      <span class="pg-view-dot" aria-hidden="true"></span>
      <span class="pg-view-copy">
        <strong>${item.label}</strong>
        <small>${item.meta}</small>
      </span>
    `;
    viewNav.append(link);
  });
  rail.append(viewNav);

  let stageNav = null;
  const stageLinks = new Map();
  if (page === "steering") {
    const divider = element("div", "pg-rail-divider");
    stageNav = element("nav", "pg-stage-nav");
    stageNav.setAttribute(
      "aria-label",
      "Oblasti steeringového experimentu"
    );
    stageNav.append(element("div", "pg-rail-label", "Oblasti steeringu"));
    stageNav.append(
      element(
        "p",
        "pg-stage-intro",
        "Směr je společný zdroj. Ostatní oblasti jsou samostatné větve, ne povinný průvodce."
      )
    );
    stageDefinitions.forEach((stage) => {
      const link = document.createElement("a");
      link.className = "pg-stage-link";
      link.href = stage.href;
      link.dataset.stageLink = stage.id;
      link.dataset.state = stage.id === "direction" ? "needs-input" : "locked";
      link.innerHTML = `
        <span class="pg-stage-marker" aria-hidden="true"></span>
        <span class="pg-stage-copy">
          <strong>${stage.label}</strong>
          <small>${stage.meta}</small>
          <em data-pg-stage-status>${
            stage.id === "direction" ? "doplň vstup" : "čeká na zdroj"
          }</em>
        </span>
      `;
      if (stage.id !== "direction") link.setAttribute("aria-disabled", "true");
      stageLinks.set(stage.id, link);
      stageNav.append(link);
    });
    rail.append(divider, stageNav);
  }

  const contentHeader = element("div", "pg-mobile-title");
  contentHeader.innerHTML = `
    <span>${pageMeta.meta}</span>
    <strong>${pageMeta.label}</strong>
  `;
  frameBody.append(rail, content);
  frame.append(contextBar, frameBody);
  rootParent.insertBefore(frame, root);
  content.append(contentHeader, root);

  const stageLockNotice = page === "steering"
    ? element("aside", "pg-stage-lock-notice")
    : null;
  if (stageLockNotice) {
    stageLockNotice.hidden = true;
    stageLockNotice.setAttribute("role", "status");
    stageLockNotice.setAttribute("aria-live", "polite");
    stageLockNotice.innerHTML = `
      <strong data-pg-stage-lock-title></strong>
      <span data-pg-stage-lock-copy></span>
    `;
    root.querySelector('[data-stage="direction"]')?.prepend(stageLockNotice);
  }

  const modelSlot = context.querySelector("[data-pg-model-slot]");
  const pageModelSelect = page === "garden"
    ? root.querySelector("#model-mode")
    : null;
  const shellOwnsModelSelect = !pageModelSelect;
  const modelSelect = pageModelSelect ?? document.createElement("select");
  if (!pageModelSelect) {
    modelSelect.id = "pg-model-mode";
    modelSelect.disabled = true;
    modelSelect.setAttribute("aria-label", "Aktivní model");
    const loadingOption = document.createElement("option");
    loadingOption.textContent = "Načítám modely…";
    modelSelect.append(loadingOption);
  }
  modelSelect.classList.add("pg-model-select");
  modelSlot?.append(modelSelect);

  const contextNodes = new Map(
    [...contextBar.querySelectorAll("[data-pg-context]")].map(
      (node) => [node.dataset.pgContext, node]
    )
  );
  const reducedMotion = window.matchMedia?.(
    "(prefers-reduced-motion: reduce)"
  )?.matches ?? false;
  let latestHealth = null;
  let modelOptionsSignature = "";
  let modelSwitching = false;

  const updateContext = (patch = {}) => {
    Object.entries(patch).forEach(([key, value]) => {
      if (key === "statusKind") {
        status.dataset.state = String(value || "idle");
        if (shellOwnsModelSelect && value === "running") {
          modelSelect.disabled = true;
        } else if (shellOwnsModelSelect && latestHealth) {
          renderModelSelect(latestHealth);
        }
        return;
      }
      if (key === "model") {
        if (value !== null && value !== undefined) {
          modelSelect.title = String(value);
        }
        return;
      }
      const node = contextNodes.get(key);
      if (node && value !== null && value !== undefined) {
        node.textContent = t(String(value));
      }
    });
  };

  const renderModelSelect = (health) => {
    latestHealth = health;
    if (!shellOwnsModelSelect) {
      modelSelect.title = (
        health?.target_model?.label ??
        health?.active_model?.label ??
        "Aktivní model"
      );
      return;
    }
    const availableModels = Array.isArray(health?.available_models)
      ? health.available_models
      : [];
    const noLoadedModel = !health?.active_model && !health?.target_model;
    const signature = JSON.stringify([
      noLoadedModel,
      availableModels.map(({id, label, available}) => [
        id,
        label,
        Boolean(available)
      ])
    ]);
    if (signature !== modelOptionsSignature) {
      const placeholder = document.createElement("option");
      placeholder.value = "";
      placeholder.textContent = t("Žádný načtený model");
      placeholder.disabled = true;
      modelSelect.replaceChildren(
        ...(noLoadedModel ? [placeholder] : []),
        ...availableModels.map((model) => {
          const option = document.createElement("option");
          option.value = model.id;
          option.disabled = !model.available;
          option.textContent = model.available
            ? model.label
            : `${model.label} · ${t("není na disku")}`;
          return option;
        })
      );
      modelOptionsSignature = signature;
    }

    const selectedId = (
      health?.target_model?.id ??
      health?.active_model?.id
    );
    if (
      selectedId &&
      [...modelSelect.options].some((option) => option.value === selectedId)
    ) {
      modelSelect.value = selectedId;
    } else if (noLoadedModel) {
      modelSelect.value = "";
    }

    const availableCount = availableModels.filter(
      (model) => model.available
    ).length;
    const serverBusy = Boolean(health?.worker?.busy);
    const transitionBusy = ["loading", "switching", "maintenance"].includes(health?.status);
    modelSelect.disabled = (
      modelSwitching ||
      serverBusy ||
      transitionBusy ||
      availableCount === 0 || (availableCount < 2 && Boolean(health?.active_model))
    );
    modelSelect.setAttribute(
      "aria-busy",
      String(modelSwitching || transitionBusy)
    );
    modelSelect.title = (
      health?.target_model?.label ??
      health?.active_model?.label ??
      "Aktivní model"
    );
  };

  const refreshHealthContext = async () => {
    try {
      const health = await api.fetchJson("/api/health", {silent: true});
      const activeModel = health?.active_model ?? {};
      renderModelSelect(health);
      const capabilities = health?.capabilities;
      const addonReasons = {addon_missing:"Není připojený potřebný experimentální doplněk.",addon_unverified:"Balíček doplňku zatím není ověřený nebo se změnily jeho soubory.",addon_incompatible:"Připojený doplněk není kompatibilní s tímto modelem nebo přesností.",addon_disabled:"Experimentální doplněk je připojený, ale vypnutý."};
      const reasonFor = (id) => addonReasons[health?.experimental_addons?.[id]?.code];
      let setupNotice = root.querySelector("[data-model-setup-notice]");
      if (health.status === "setup_required" && page !== "model-admin") {
        if (!setupNotice) {
          setupNotice = element("p", "pg-model-capability-notice");
          setupNotice.dataset.modelSetupNotice = "true";
          const setupLink = element("a", "", "Otevřít správu modelů");
          setupLink.href = "/sprava-modelu";
          setupNotice.append(element("span", "", "Nejdřív připoj nebo načti model. "), setupLink);
          root.prepend(setupNotice);
        }
      } else setupNotice?.remove();
      if (Array.isArray(capabilities)) {
        const detectorComparison = page === "detector" && health?.worker?.operation === "detector_readers";
        for (const link of viewNav.querySelectorAll(".pg-view-link")) {
          const supported = capabilities.includes(link.dataset.page) || (detectorComparison && link.dataset.page === "detector");
          link.setAttribute("aria-disabled", String(!supported));
          link.style.opacity = supported ? "" : "0.4";
          if (!supported) link.title = t(reasonFor(link.dataset.page) || "Tento pohled není pro aktivní model dostupný.");
          else link.removeAttribute("title");
        }
        let notice = root.querySelector("[data-model-capability-notice]");
        if (!capabilities.includes(page) && !detectorComparison) {
          if (!notice) {
            notice = element("p", "pg-model-capability-notice", "Tento pohled není pro aktivní model dostupný. Zvol Apertus nebo otevři porovnání modelů.");
            notice.dataset.modelCapabilityNotice = "true";
            notice.setAttribute("role", "status");
            root.prepend(notice);
          }
          const reason=reasonFor(page);
          if(reason&&notice.dataset.reason!==reason){notice.dataset.reason=reason;const link=element("a","","Otevřít správu modelů");link.href="/sprava-modelu#experimental-addons";notice.replaceChildren(element("span","",reason+" "),link);}
        } else notice?.remove();
      }
      updateContext({
        model: activeModel.label ?? activeModel.id ?? t("Žádný načtený model"),
        run: health?.model_generation === undefined
          ? "lokální"
          : `generace ${health.model_generation}`
      });
      return health;
    } catch {
      // The page's own status remains authoritative while the server is away.
      return null;
    }
  };

  const delay = (milliseconds) => new Promise(
    (resolve) => window.setTimeout(resolve, milliseconds)
  );

  const switchPhaseLabel = (health, fallbackLabel) => {
    if (health?.phase === "queued") {
      return "Čekám na dokončení výpočtu…";
    }
    if (health?.phase === "unloading") {
      return "Uvolňuji současný model…";
    }
    if (health?.phase === "loading") {
      return `Načítám ${health?.target_model?.label ?? fallbackLabel}…`;
    }
    return `Přepínám na ${fallbackLabel}…`;
  };

  const switchModelFromContext = async () => {
    const requestedId = modelSelect.value;
    let health = latestHealth ?? await refreshHealthContext();
    const requested = health?.available_models?.find(
      (model) => model.id === requestedId
    );
    if (
      !requestedId ||
      !requested?.available ||
      (
        health?.status === "ready" &&
        health?.active_model?.id === requestedId
      )
    ) {
      renderModelSelect(health ?? {});
      return;
    }

    const startingGeneration = Number(health?.model_generation ?? -1);
    modelSwitching = true;
    renderModelSelect(health ?? {});
    body.dataset.modelSwitching = "true";
    content.setAttribute("aria-busy", "true");
    updateContext({
      status: `Přepínám na ${requested.label}…`,
      statusKind: "running"
    });

    try {
      const accepted = await api.fetchJson("/api/model", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({model_id: requestedId}),
        requestKey: "global-model-switch",
        timeoutMs: 30_000
      });
      const expectedTransitionId = accepted?.transition_id;
      const started = Date.now();

      while (Date.now() - started < 300_000) {
        await delay(250);
        health = await refreshHealthContext();
        if (!health) continue;
        updateContext({
          status: switchPhaseLabel(health, requested.label),
          statusKind: "running"
        });
        if (health.status === "error") {
          throw new Error(
            health.error || `${requested.label} se nepodařilo načíst.`
          );
        }
        const completed = (
          health.status === "ready" &&
          health.active_model?.id === requestedId &&
          (
            accepted?.accepted === false ||
            Number(health.model_generation) > startingGeneration
          )
        );
        if (completed) {
          updateContext({
            status: `${requested.label} je připraven`,
            statusKind: "ready"
          });
          window.location.reload();
          return;
        }
        if (
          expectedTransitionId !== undefined &&
          health.transition_id !== expectedTransitionId
        ) {
          throw new Error(
            "Během přepnutí se změnil jiný modelový běh. Obnov stránku a zkontroluj aktivní model."
          );
        }
      }
      throw new Error("Přepnutí modelu překročilo pětiminutový limit.");
    } catch (error) {
      const currentHealth = error.health ?? await refreshHealthContext();
      if (currentHealth) renderModelSelect(currentHealth);
      updateContext({
        status: error.message || "Model se nepodařilo přepnout",
        statusKind: "error"
      });
    } finally {
      modelSwitching = false;
      delete body.dataset.modelSwitching;
      content.removeAttribute("aria-busy");
      if (latestHealth) renderModelSelect(latestHealth);
    }
  };

  if (shellOwnsModelSelect) {
    modelSelect.addEventListener("change", switchModelFromContext);
  }

  let activeStage = null;
  let pendingRequestedStage = null;

  const setPendingRequestedStage = (
    stageId,
    {updateHistory = false} = {}
  ) => {
    const definition = stageDefinitions.find((stage) => stage.id === stageId);
    if (!definition || stageId === "direction") return false;
    pendingRequestedStage = stageId;
    if (updateHistory) {
      history.replaceState(null, "", definition.href);
    }
    if (stageLockNotice) {
      const title = stageLockNotice.querySelector(
        "[data-pg-stage-lock-title]"
      );
      const copy = stageLockNotice.querySelector(
        "[data-pg-stage-lock-copy]"
      );
      if (title) {
        title.textContent = `Etapa „${definition.label}“ zatím čeká na podklady.`;
      }
      if (copy) {
        copy.textContent = ["sham", "sweep"].includes(stageId)
          ? (
              "Nejprve spusť steering a připrav matched páry v části " +
              "Kontrolované pokračování. Jakmile budou podklady dostupné, " +
              "požadovaná etapa se otevře automaticky."
            )
          : (
              "Nejprve spusť steering. Po dokončení se požadovaná etapa " +
              "otevře automaticky."
            );
      }
      stageLockNotice.hidden = false;
    }
    return true;
  };

  const clearPendingRequestedStage = () => {
    pendingRequestedStage = null;
    if (stageLockNotice) stageLockNotice.hidden = true;
  };

  const updateStages = (stages = []) => {
    stages.forEach((entry) => {
      const link = stageLinks.get(entry.id);
      if (!link) return;
      const state = entry.state || "locked";
      link.dataset.state = state;
      const stateText = link.querySelector("[data-pg-stage-status]");
      if (stateText && entry.label) stateText.textContent = t(entry.label);
      const disabled = state === "locked";
      if (disabled) {
        link.setAttribute("aria-disabled", "true");
      } else {
        link.removeAttribute("aria-disabled");
      }
      if (entry.id === "direction" && primaryAction) {
        primaryAction.textContent = t(
          state === "running"
            ? "Počítá se…"
            : state === "frozen" || state === "stale"
            ? "Přepočítat řez"
            : "Spustit řez"
        );
      }
    });
  };

  const activateStage = (
    stageId,
    {updateHistory = true, scroll = true} = {}
  ) => {
    const link = stageLinks.get(stageId);
    if (!link || link.getAttribute("aria-disabled") === "true") return false;
    const target = document.querySelector(link.getAttribute("href"));
    if (!target || target.hidden || target.closest("[hidden]")) return false;
    activeStage = stageId;
    body.dataset.activeWorkbenchStage = stageId;
    body.classList.add("pg-stage-focus");
    document.querySelectorAll("[data-stage]").forEach((stage) => {
      stage.classList.toggle(
        "pg-stage-active",
        stage.dataset.stage === stageId
      );
    });
    stageLinks.forEach((candidate, id) => {
      if (id === stageId) {
        candidate.setAttribute("aria-current", "location");
      } else {
        candidate.removeAttribute("aria-current");
      }
    });
    if (updateHistory) {
      history.replaceState(null, "", link.getAttribute("href"));
    }
    if (pendingRequestedStage === stageId) {
      clearPendingRequestedStage();
    }
    if (scroll) {
      window.requestAnimationFrame(() => {
        target.scrollIntoView({
          behavior: reducedMotion ? "auto" : "smooth",
          block: "start"
        });
        const labelId = target
          .getAttribute("aria-labelledby")
          ?.trim()
          .split(/\s+/)[0];
        const focusTarget = labelId
          ? document.getElementById(labelId)
          : target;
        if (!focusTarget.hasAttribute("tabindex")) {
          focusTarget.setAttribute("tabindex", "-1");
        }
        focusTarget.focus({preventScroll: true});
      });
    }
    return true;
  };

  const activateRequestedStage = ({scroll = true} = {}) => {
    if (!pendingRequestedStage) return false;
    return activateStage(pendingRequestedStage, {
      updateHistory: false,
      scroll
    });
  };

  const shellApi = {
    updateContext,
    updateStages,
    activateStage,
    activateRequestedStage,
    focusStage(stageId) {
      const link = stageLinks.get(stageId);
      if (link) link.focus();
    }
  };
  window.ProbabilityGardenShell = shellApi;

  window.addEventListener(api.eventName, (event) => {
    const request = event.detail ?? {};
    if (request.silent) return;
    body.dataset.apiRequestState = request.phase ?? "idle";
    body.dataset.apiActiveRequests = String(request.activeCount ?? 0);
    if ((request.activeCount ?? 0) > 0) {
      content.setAttribute("aria-busy", "true");
    } else if (!modelSwitching) {
      content.removeAttribute("aria-busy");
    }
    if (request.phase === "stale") {
      body.dataset.apiLastOutcome = "stale";
    } else if (request.phase === "error") {
      body.dataset.apiLastOutcome = "error";
    } else if (request.phase === "success") {
      body.dataset.apiLastOutcome = "success";
    }
  });

  window.addEventListener(
    "probability-garden:workbench-state",
    (event) => {
      updateContext(event.detail?.context);
      updateStages(event.detail?.stages);
    }
  );

  rail.addEventListener("click", (event) => {
    const link = event.target.closest("[data-stage-link]");
    if (!link) return;
    if (link.getAttribute("aria-disabled") === "true") {
      event.preventDefault();
      setPendingRequestedStage(link.dataset.stageLink, {
        updateHistory: true
      });
      activateStage("direction", {
        updateHistory: false,
        scroll: true
      });
      return;
    }
    event.preventDefault();
    activateStage(link.dataset.stageLink);
  });

  window.addEventListener("hashchange", () => {
    if (page !== "steering") return;
    const requestedStage = stageDefinitions.find(
      (stage) => stage.href === window.location.hash
    )?.id;
    if (!requestedStage) return;
    const activated = activateStage(requestedStage, {
      updateHistory: false,
      scroll: true
    });
    if (activated) return;
    setPendingRequestedStage(requestedStage);
    activateStage("direction", {
      updateHistory: false,
      scroll: true
    });
  });

  const mirrorStatus = () => {
    refreshHealthContext();
    const source = (
      root.querySelector("#model-state-text") ??
      root.querySelector("#storage-state")
    );
    if (!source) {
      updateContext({status: "Lokální nástroj", statusKind: "ready"});
      return;
    }
    const sourcePill = source.closest("[data-state]");
    const sync = () => {
      const sourceState = sourcePill?.dataset.state;
      updateContext({
        status: source.textContent?.trim() || "Lokální nástroj",
        statusKind:
          sourceState === "error" || sourceState === "missing"
            ? "error"
            : sourceState === "loading"
            ? "running"
            : "ready"
      });
      refreshHealthContext();
    };
    sync();
    const observer = new MutationObserver(sync);
    observer.observe(source, {
      childList: true,
      characterData: true,
      subtree: true
    });
    if (sourcePill) {
      observer.observe(sourcePill, {
        attributes: true,
        attributeFilter: ["data-state"]
      });
    }
  };

  if (page !== "steering") {
    mirrorStatus();
  } else {
    refreshHealthContext();
  }
  i18n?.translateTree(frame);
  body.dataset.workbenchShellReady = "true";
  body.classList.add("pg-shell-ready");
  window.addEventListener("focus", refreshHealthContext);
  const healthRefreshInterval = window.setInterval(() => {
    if (!document.hidden) refreshHealthContext();
  }, 2_500);
  window.addEventListener(
    "pagehide",
    () => window.clearInterval(healthRefreshInterval),
    {once: true}
  );
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) refreshHealthContext();
  });
  if (page === "steering") {
    const requestedStage = stageDefinitions.find(
      (stage) => stage.href === window.location.hash
    )?.id;
    const activatedRequestedStage = activateStage(
      requestedStage ?? "direction",
      {
        updateHistory: false,
        scroll: false
      }
    );
    if (!activatedRequestedStage) {
      if (requestedStage) setPendingRequestedStage(requestedStage);
      activateStage("direction", {
        updateHistory: false,
        scroll: false
      });
    }
    const sourceRunButton = root.querySelector("#run-steering");
    const syncPrimaryAction = () => {
      if (!primaryAction || !sourceRunButton) return;
      primaryAction.disabled = sourceRunButton.disabled;
    };
    syncPrimaryAction();
    if (sourceRunButton) {
      const observer = new MutationObserver(syncPrimaryAction);
      observer.observe(sourceRunButton, {
        attributes: true,
        attributeFilter: ["disabled"]
      });
      primaryAction?.addEventListener("click", () => {
        if (primaryAction.disabled) return;
        activateStage("direction", {scroll: false});
        sourceRunButton.click();
      });
    }
  }
  window.dispatchEvent(
    new CustomEvent("probability-garden:request-workbench-state")
  );
})();
