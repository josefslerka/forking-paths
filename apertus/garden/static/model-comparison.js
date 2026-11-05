(() => {
  "use strict";
  const api = window.ProbabilityGardenApi;
  const t = window.ProbabilityGardenI18n?.t ?? ((v) => String(v));
  const escape = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[c]);
  const phaseLabel = (v) => t(({loading:"Načítání modelu",scoring:"Výpočet",restoring:"Obnova modelu",saving:"Ukládání",done:"Hotovo",queued:"Zařazuji srovnávací běh…"})[v] || v);
  const statusLabel = (v) => t(({completed:"dokončeno",complete:"dokončeno",cancelled:"přerušeno",failed:"chyba"})[v] || v);
  const number = (v) => Number.isFinite(v) ? v.toFixed(3) : "—";
  const chart = (rows, labels) => {
    const values = rows.flatMap((r) => r.complete ? [r.a_bits, r.b_bits] : []).filter(Number.isFinite);
    if (!values.length) return "";
    const width = 960, height = 230, left = 48, right = 940, bottom = 196;
    const max = Math.max(1, ...values) * 1.1;
    const x = (i) => left + (right-left) * i / Math.max(1,rows.length-1);
    const y = (v) => bottom - 170*v/max;
    const path = (key) => {
      let pen = false;
      return rows.map((r,i) => { if (!r.complete || !Number.isFinite(r[key])) { pen=false; return ""; } const segment = `${pen ? "L" : "M"}${x(i)},${y(r[key])}`; pen=true; return segment; }).join(" ");
    };
    return `<svg class="chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="${escape(t("Průběh surprisalu po společných úsecích"))}"><line x1="${left}" y1="${bottom}" x2="${right}" y2="${bottom}" stroke="#62766b"/><text x="8" y="25" fill="#52665b">${escape(t("bity"))}</text><text x="8" y="${bottom}" fill="#52665b">0</text><text x="8" y="45" fill="#52665b">${number(max)}</text><path d="${path("a_bits")}" stroke="#287755" stroke-width="2.5" fill="none"/><path d="${path("b_bits")}" stroke="#6a53a1" stroke-width="2.5" fill="none"/></svg><div class="legend" data-i18n-ignore><span class="a">A · ${escape(labels[0])}</span><span class="b">B · ${escape(labels[1])}</span></div>`;
  };
  const recordHtml = (record) => {
    if (record.payload.request.steering) return window.ProbabilitySteeringComparison.recordHtml(record);
    const payload = record.payload, models = payload.request.model_ids;
    const labels = models.map((id) => record.provenance.models[id]?.label || id);
    const inputs = new Map(payload.request.items.map((i) => [i.id,i]));
    const summary = payload.summary;
    return `<div class="panel"><h2>${escape(record.annotation.title)}</h2><div class="metrics"><div>${escape(t("Spárovaná zadání"))}<strong>${summary.paired_items} / ${summary.requested_items}</strong></div><div>${escape(t("Shoda směru kontextového účinku"))}<strong>${summary.context_direction_agreement_count} / ${summary.context_direction_eligible_count}</strong></div><div>${escape(t("Stav záznamu"))}<strong>${escape(statusLabel(record.capture.status))}</strong></div></div><div class="actions"><a class="action-link" href="/archiv?id=${encodeURIComponent(record.id)}">${escape(t("Otevřít v archivu"))}</a><a class="action-link" href="/api/experiments/${encodeURIComponent(record.id)}/export.json">${escape(t("Export JSON"))}</a><a class="action-link" href="/api/experiments/${encodeURIComponent(record.id)}/export.csv">${escape(t("Export CSV"))}</a></div>${record.capture.warnings.length ? `<p class="error">${record.capture.warnings.map(escape).join("<br>")}</p>` : ""}</div>` + payload.comparisons.map((item) => {
      const input = inputs.get(item.id);
      if (item.kind === "context") return `<article class="panel"><h3 data-i18n-ignore>${escape(item.id)}</h3><div class="grid" data-i18n-ignore><p>A: ${escape(input.prefix_a)}</p><p>B: ${escape(input.prefix_b)}</p></div><p data-i18n-ignore>${escape(input.continuation)}</p><div class="table-scroll"><table><thead><tr><th>${escape(t("Model"))}</th><th>${escape(t("S za prefixem A"))}</th><th>${escape(t("S za prefixem B"))}</th><th>${escape(t("Účinek B − A"))}</th></tr></thead><tbody>${models.map((id,i) => {const m=payload.measurements[id][item.id]; return `<tr><td data-i18n-ignore>${escape(labels[i])}</td><td class="num">${number(m.a.summary.total_continuation_surprisal_bits)}</td><td class="num">${number(m.b.summary.total_continuation_surprisal_bits)}</td><td class="num">${number(m.context_effect_bits)}</td></tr>`;}).join("")}</tbody></table></div></article>`;
      return `<article class="panel"><h3 data-i18n-ignore>${escape(item.id)}</h3><p data-i18n-ignore>${escape(input.text)}</p>${chart(item.units,labels)}<div class="table-scroll"><table><thead><tr><th>${escape(t("Úsek"))}</th><th>A · ${escape(t("bity"))}</th><th>B · ${escape(t("bity"))}</th><th>B − A</th><th>${escape(t("Zarovnání"))}</th></tr></thead><tbody>${item.units.map((r) => `<tr><td data-i18n-ignore>${escape(r.text)}</td><td class="num">${number(r.a_bits)}</td><td class="num">${number(r.b_bits)}</td><td class="num">${number(r.delta_b_minus_a_bits)}</td><td>${escape(t(r.complete ? "úplné" : "neúplné"))}${r.complete ? "" : `<details><summary>${escape(t("Důvod"))}</summary><span data-i18n-ignore>${escape([...r.a_reasons,...r.b_reasons].join(", "))}</span></details>`}</td></tr>`).join("")}</tbody></table></div></article>`;
    }).join("") + `<details class="panel"><summary>${escape(t("Protokol a modely"))}</summary><pre data-i18n-ignore>${escape(JSON.stringify(record.provenance,null,2))}</pre></details>`;
  };
  window.ProbabilityModelComparison = {recordHtml};
  if (document.body.dataset.workbenchPage !== "model-comparison") return;
  const $ = (id) => document.getElementById(id);
  const status = $("comparison-status"), run = $("run-comparison"), cancel = $("cancel-comparison");
  let busy = false, jobId = null, signature = "";
  const unavailableSelection = () => ["model-a","model-b"].some((key) => !$(key).value || $(key).selectedOptions[0]?.disabled);
  const showStatus = (message, error=false) => { status.textContent=message; status.classList.toggle("error",error); };
  const setBusy = (value) => {window.ProbabilitySteeringComparison?.setBusy(value);["model-a","model-b","input-mode","experiment-title"].forEach((key)=>{$(key).disabled=value;});busy=value; run.disabled=value || $("model-a").options.length < 2 || unavailableSelection(); cancel.disabled=!value; $("load-examples").disabled=value;};
  const refreshModels = async () => {
    const health = await api.fetchJson("/api/health",{silent:true});
    const available = health.available_models.filter((m) => m.available && ($("input-mode").value!=="steering" || m.capabilities?.includes("steering")));
    const next = JSON.stringify([$("input-mode").value,available.map((m) => [m.id,m.label])]);
    if (signature !== next) {
      signature = next;
      ["model-a","model-b"].forEach((key,i) => {const select=$(key), previous=select.value; select.replaceChildren(...available.map((m)=>{const option=document.createElement("option");option.value=m.id;option.textContent=m.label;return option;}));select.value=available.some((m)=>m.id===previous) ? previous : available.find((m)=>i===0 ? ["hybrid","mock"].includes(m.id) : ["eurollm9","mock-eurollm9"].includes(m.id))?.id || available[i]?.id || "";});
    }
    if (!busy) {run.disabled=health.status!=="ready" || health.worker.busy || available.length<2 || unavailableSelection(); showStatus(available.length<2 ? t("Pro srovnání potřebuješ dva dostupné modely.") : unavailableSelection() ? t("Model uloženého měření není dostupný. Pro nový běh vyber dostupný model.") : health.status==="ready" ? t("Připraveno k porovnání.") : t("Model se načítá…"));}
  };
  const poll = async (id) => {
    try {
      const job = await api.fetchJson(`/api/model-comparison/jobs/${encodeURIComponent(id)}`,{silent:true});
      if (id !== jobId) return;
      showStatus(`${t("Měření")}: ${job.completed_measurements} / ${job.total_measurements} · ${job.current_model_id || ""} · ${phaseLabel(job.phase)}`);
      if (job.phase === "done") {
        localStorage.removeItem("garden:model-comparison-job");
        setBusy(false);
        if (job.record) $("comparison-results").innerHTML=recordHtml(job.record);
        showStatus(job.storage_error ? `${t("Výsledek se nepodařilo uložit")}: ${job.storage_error}` : `${t("Běh dokončen")}: ${statusLabel(job.status)}`, job.status==="failed" || Boolean(job.storage_error));
        if (job.storage_error && job.record) {
          const button=document.createElement("button");button.textContent=t("Uložit znovu");button.onclick=async()=>{try{await api.fetchJson("/api/experiments",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({record:job.record})});showStatus(t("Výsledek uložen."));button.remove();}catch(error){showStatus(error.message,true);}};$("comparison-results").prepend(button);
        }
      } else setTimeout(()=>poll(id),750);
    } catch(error) {if (error.status === 404) localStorage.removeItem("garden:model-comparison-job"); setBusy(false); showStatus(error.message,true);}
  };
  const syncMode = () => {
    ["text","context","batch","steering"].forEach((key)=>{$(`${key}-input`).hidden=$("input-mode").value!==key;});
    $("load-examples").hidden=$("input-mode").value==="steering";
  };
  $("input-mode").onchange = () => {syncMode();void refreshModels().catch((error)=>showStatus(error.message,true));};
  ["model-a","model-b"].forEach((key)=>{$(key).onchange=()=>{if (!busy) void refreshModels();};});
  const restoreInputs = (record) => {
    const request=record.payload.request;
    $("experiment-title").value=request.title;
    request.model_ids.forEach((id,i)=>{const select=$(i===0 ? "model-a" : "model-b");if (![...select.options].some((option)=>option.value===id)) {const option=document.createElement("option");option.value=id;option.textContent=record.provenance.models[id]?.label || id;option.disabled=true;select.append(option);}select.value=id;});
    if (request.steering) {$("input-mode").value="steering";window.ProbabilitySteeringComparison.restore(request);}
    else if (request.items.length===1) {const item=request.items[0];$("input-mode").value=item.kind;if (item.kind==="text") $("comparison-text").value=item.text;else ["prefix-a","prefix-b","continuation"].forEach((key)=>{$(key).value=item[key.replaceAll("-","_")];});}
    else {$("input-mode").value="batch";$("batch-json").value=JSON.stringify(request.items,null,2);}
    $("input-mode").onchange();setBusy(false);
  };
  $("load-examples").onclick = async () => {try{const data=await api.fetchJson("/api/model-comparison/examples");$("batch-json").value=JSON.stringify(data.items,null,2);$("input-mode").value="batch";$("input-mode").onchange();showStatus(t("Načteno 30 průzkumných příkladů."));}catch(error){showStatus(error.message,true);}};
  $("comparison-form").onsubmit = async(event)=>{
    event.preventDefault();
    if (busy) return;
    try {
      const mode=$("input-mode").value;
      const steeringFields=mode==="steering" ? window.ProbabilitySteeringComparison.requestFields() : {};
      const items=mode==="steering" ? steeringFields.items : mode==="batch" ? JSON.parse($("batch-json").value) : mode==="context" ? [{id:"context-1",prefix_a:$("prefix-a").value,prefix_b:$("prefix-b").value,continuation:$("continuation").value}] : [{id:"text-1",text:$("comparison-text").value}];
      setBusy(true);showStatus(t("Zařazuji srovnávací běh…"));
      const job=await api.fetchJson("/api/model-comparison",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({model_ids:[$("model-a").value,$("model-b").value],items,title:$("experiment-title").value,...(steeringFields.steering ? {steering:steeringFields.steering} : {})})});
      jobId=job.id;localStorage.setItem("garden:model-comparison-job",jobId);void poll(jobId);
    }catch(error){setBusy(false);showStatus(error.message,true);}
  };
  cancel.onclick=async()=>{try{await api.fetchJson("/api/model-comparison/cancel",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({job_id:jobId})});cancel.disabled=true;showStatus(t("Běh se přeruší po dokončení aktuálního měření."));}catch(error){showStatus(error.message,true);}};
  const initialize=async()=>{
    try {
      await refreshModels();
      const experiment=new URLSearchParams(location.search).get("experiment");
      if (experiment) {const data=await api.fetchJson(`/api/experiments/${encodeURIComponent(experiment)}`);const record=data.record || data;restoreInputs(record);$("comparison-results").innerHTML=recordHtml(record);await refreshModels();}
      const pending=localStorage.getItem("garden:model-comparison-job");
      if (pending && !experiment) {jobId=pending;setBusy(true);void poll(pending);}
    }catch(error){showStatus(error.message,true);}
  };
  if (new URLSearchParams(location.search).get("mode")==="steering") $("input-mode").value="steering";
  syncMode();
  void initialize();
  const refreshTimer=setInterval(()=>{if(run.disabled && !busy) void refreshModels().catch(()=>{});},2000);
  window.addEventListener("pagehide",()=>clearInterval(refreshTimer),{once:true});
  window.addEventListener("probability-garden:model-switched",()=>{void refreshModels();});
})();
