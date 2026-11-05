(() => {
  "use strict";
  const t = (value) => window.ProbabilityGardenI18n?.t(value) ?? value;
  const escape = (value) => String(value ?? "").replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[c]);
  const number = (value) => Number.isFinite(value) ? value.toFixed(3) : "—";
  const DEFAULT_PAIRS = [
    {negative_text:"Ten tón působí chladně v tomto rámci",positive_text:"Ten tón působí vřele v tomto rámci"},
    {negative_text:"Odpověď je odmítavá v tomto rámci",positive_text:"Odpověď je vstřícná v tomto rámci"},
    {negative_text:"Vyznění zůstává odtažité v tomto rámci",positive_text:"Vyznění zůstává přátelské v tomto rámci"}
  ];
  const EXAMPLE_ITEMS = [
    {id:"uvitani",test_text:"Způsob, jakým přivítala návštěvu, byl",negative_text:" nevlídný.",positive_text:" srdečný."},
    {id:"soused",test_text:"Chování nového souseda působilo",negative_text:" nepříjemně.",positive_text:" mile."},
    {id:"dopis",test_text:"Z jejího dopisu vyzařovala",negative_text:" nevraživost.",positive_text:" náklonnost."}
  ];
  const profile = (item, labels) => {
    const width=780, height=250, xs=[75,390,705], top=28, bottom=205;
    const keys=["negative","baseline","positive"];
    const series=[item.a,item.b].map((summary) => keys.map((key) => summary.contrast_bits[key]-summary.contrast_bits.baseline));
    const scale=Math.max(.05,...series.flat().map(Math.abs))*1.2;
    const y=(value) => (top+bottom)/2-value*(bottom-top)/(2*scale);
    return `<svg class="chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="${escape(t("Posun preference proti α = 0"))}"><line x1="50" y1="${y(0)}" x2="730" y2="${y(0)}" stroke="#8c9c92"/><text x="8" y="${top}" fill="#52665b">${number(scale)}</text><text x="8" y="${y(0)-5}" fill="#52665b">0</text><text x="8" y="${bottom}" fill="#52665b">${number(-scale)}</text>${series.map((values,i) => `<polyline points="${values.map((v,j)=>`${xs[j]},${y(v)}`).join(" ")}" fill="none" stroke="${i ? "#6a53a1" : "#287755"}" stroke-width="3"/>${values.map((v,j)=>`<circle cx="${xs[j]}" cy="${y(v)}" r="4" fill="${i ? "#6a53a1" : "#287755"}"/>`).join("")}`).join("")}${["−α","0","+α"].map((label,i)=>`<text x="${xs[i]}" y="237" text-anchor="middle" fill="#52665b">${label}</text>`).join("")}</svg><p class="muted">${escape(t("Posun preference proti α = 0"))} · ${escape(t("bity"))}</p><div class="legend" data-i18n-ignore><span class="a">A · ${escape(labels[0])}</span><span class="b">B · ${escape(labels[1])}</span></div>`;
  };
  const recordHtml = (record) => {
    const payload=record.payload, models=payload.request.model_ids;
    const labels=models.map((id)=>record.provenance.models[id]?.label || id);
    const inputs=new Map(payload.request.items.map((item)=>[item.id,item]));
    const summary=payload.summary;
    const state=({complete:"dokončeno",completed:"dokončeno",cancelled:"přerušeno",failed:"chyba"})[record.capture.status] || record.capture.status;
    return `<div class="panel"><h2>${escape(record.annotation.title)}</h2><p>${escape(t("Steering na stejných příkladech"))}</p><div class="metrics"><div>${escape(t("Spárovaná zadání"))}<strong>${summary.paired_items} / ${summary.requested_items}</strong></div><div>${escape(t("Shoda směru steeringového účinku"))}<strong>${summary.steering_direction_agreement_count ?? 0} / ${summary.steering_direction_eligible_count ?? 0}</strong></div><div>${escape(t("Stav záznamu"))}<strong>${escape(t(state))}</strong></div></div><div class="actions"><a class="action-link" href="/archiv?id=${encodeURIComponent(record.id)}">${escape(t("Otevřít v archivu"))}</a><a class="action-link" href="/api/experiments/${encodeURIComponent(record.id)}/export.json">${escape(t("Export JSON"))}</a><a class="action-link" href="/api/experiments/${encodeURIComponent(record.id)}/export.csv">${escape(t("Export CSV"))}</a></div>${record.capture.warnings.length ? `<p class="error">${record.capture.warnings.map(escape).join("<br>")}</p>` : ""}</div>` + payload.comparisons.map((item)=>{
      const input=inputs.get(item.id);
      return `<article class="panel"><h3 data-i18n-ignore>${escape(item.id)}</h3><p data-i18n-ignore>${escape(input.test_text)}</p><div class="grid" data-i18n-ignore><p>− ${escape(input.negative_text)}</p><p>+ ${escape(input.positive_text)}</p></div>${profile(item,labels)}<div class="table-scroll"><table><thead><tr><th>${escape(t("Model"))}</th><th>${escape(t("Vrstva"))}</th><th>C(−α)</th><th>C(0)</th><th>C(+α)</th><th>${escape(t("Symetrický efekt"))}</th><th>${escape(t("Kontroly ≥ efekt"))}</th><th>${escape(t("Zásah / reziduum"))}</th></tr></thead><tbody>${[item.a,item.b].map((measurement,i)=>`<tr><td data-i18n-ignore>${escape(labels[i])}</td><td class="num">L${measurement.layer} / ${measurement.layer_count}</td>${["negative","baseline","positive"].map((branch)=>`<td class="num">${number(measurement.contrast_bits[branch])}</td>`).join("")}<td class="num">${number(measurement.effect_bits)}</td><td class="num">${measurement.sham_count_at_or_above_target} / ${measurement.sham_sample_count}</td><td class="num">${number(measurement.prefix_relative_shift_norm)}</td></tr>`).join("")}</tbody></table></div><p class="muted">${escape(t("Kladný efekt znamená posun ke kladnému pokračování. Kontrolní směry poskytují popisnou referenci, nikoli test významnosti."))}</p><details><summary>${escape(t("Kontrolní směry"))}</summary>${[item.a,item.b].map((measurement,i)=>`<p data-i18n-ignore>${escape(labels[i])}: ${measurement.sham_effect_bits.map(number).join(", ")} bit</p>`).join("")}</details></article>`;
    }).join("") + `<details class="panel"><summary>${escape(t("Protokol a modely"))}</summary><pre data-i18n-ignore>${escape(JSON.stringify(record.provenance,null,2))}</pre></details>`;
  };
  const $=(id)=>document.getElementById(id);
  let pairs=DEFAULT_PAIRS.map((pair)=>({...pair}));
  const readPairs=()=>[...$("comparison-steering-pairs").querySelectorAll("[data-steering-pair]")].map((row)=>({negative_text:row.querySelector('[data-pole="negative"]').value,positive_text:row.querySelector('[data-pole="positive"]').value}));
  const renderPairs=()=>{
    $("comparison-steering-pairs").innerHTML=pairs.map((pair,i)=>`<div data-steering-pair class="panel"><div class="grid"><label>${escape(t("Záporná strana"))}<textarea data-pole="negative" data-i18n-ignore>${escape(pair.negative_text)}</textarea></label><label>${escape(t("Kladná strana"))}<textarea data-pole="positive" data-i18n-ignore>${escape(pair.positive_text)}</textarea></label></div><button type="button" data-remove-pair="${i}" ${pairs.length<=3 ? "disabled" : ""}>${escape(t("Odebrat pár"))}</button></div>`).join("");
    $("add-comparison-steering-pair").disabled=pairs.length>=8;
  };
  const syncBatch=()=>{const batch=$("comparison-steering-batch").checked;$("steering-single-input").hidden=batch;$("steering-batch-input").hidden=!batch;};
  const requestFields=()=>({
    steering:{contrast_pairs:readPairs(),relative_depth:Number($("comparison-steering-depth").value),alpha_magnitude:Number($("comparison-steering-alpha").value),sham_samples:Number($("comparison-steering-shams").value),seed:Number($("comparison-steering-seed").value)},
    items:$("comparison-steering-batch").checked ? JSON.parse($("comparison-steering-items").value) : [{id:"steering-1",test_text:$("comparison-steering-prefix").value,negative_text:$("comparison-steering-negative").value,positive_text:$("comparison-steering-positive").value}]
  });
  const restore=(request)=>{
    pairs=request.steering.contrast_pairs.map((pair)=>({...pair}));renderPairs();
    for (const [id,key] of [["depth","relative_depth"],["alpha","alpha_magnitude"],["shams","sham_samples"],["seed","seed"]]) {
      const field=$(`comparison-steering-${id}`), value=String(request.steering[key]);
      if (field.tagName==="SELECT" && ![...field.options].some((option)=>option.value===value)) {const option=document.createElement("option");option.value=value;option.textContent=value;field.append(option);}
      field.value=value;
    }
    $("comparison-steering-batch").checked=request.items.length!==1;
    $("comparison-steering-items").value=JSON.stringify(request.items,null,2);
    if (request.items.length===1) {const item=request.items[0];$("comparison-steering-prefix").value=item.test_text;$("comparison-steering-negative").value=item.negative_text;$("comparison-steering-positive").value=item.positive_text;}
    syncBatch();
  };
  const setBusy=(value)=>{
    if (!$("steering-input")) return;
    $("steering-input").querySelectorAll("input,select,textarea,button").forEach((field)=>{field.disabled=value;});
    if (!value) {$("add-comparison-steering-pair").disabled=readPairs().length>=8;$("comparison-steering-pairs").querySelectorAll("[data-remove-pair]").forEach((button)=>{button.disabled=readPairs().length<=3;});}
  };
  window.ProbabilitySteeringComparison={recordHtml,requestFields,restore,setBusy};
  if (!$("steering-input")) return;
  renderPairs();
  $("add-comparison-steering-pair").onclick=()=>{pairs=readPairs();if(pairs.length<8){pairs.push({negative_text:"",positive_text:""});renderPairs();}};
  $("comparison-steering-pairs").onclick=(event)=>{const button=event.target.closest("[data-remove-pair]");if(!button)return;pairs=readPairs();if(pairs.length>3){pairs.splice(Number(button.dataset.removePair),1);renderPairs();}};
  $("comparison-steering-batch").onchange=syncBatch;
  $("load-steering-examples").onclick=()=>{ $("comparison-steering-items").value=JSON.stringify(EXAMPLE_ITEMS,null,2);$("comparison-steering-batch").checked=true;syncBatch();};
})();
