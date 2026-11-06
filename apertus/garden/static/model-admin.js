(() => {
  "use strict";
  const api = window.ProbabilityGardenApi;
  const t = window.ProbabilityGardenI18n?.t ?? ((value) => value);
  const $ = (id) => document.getElementById(id);
  let snapshot = null, inspection = null, plan = null, disconnect = null;
  let pending = false, refreshRunning = false, cardSignature = "", installJob = null;
  const profiles = {bf16:"Apertus 8B · BF16", "8bit":"Apertus 8B · 8bit", "4bit":"Apertus 8B · 4bit", hybrid:"Apertus 8B · Hybrid 8bit/BF16",eurollm9:"EuroLLM 9B · 6bit", "eurollm9-hf":"EuroLLM 9B · HF"};
  const capabilityNames = {tokenization:"Tokenizace",garden:"Pravděpodobnostní zahrada",surprise:"Překvapení textu",contrast:"Kontrast kontextů","model-comparison":"Porovnání modelů",embeddings:"Embeddingy",raw:"RAW vrstvy",steering:"Steering",attribution:"Preference",patching:"Kauzální přenos",tuned:"RAW × TUNED",jlens:"RAW × J-LENS",detector:"AI detektor"};
  const errors = {
    dependency_missing:"Nejdřív doinstaluj potřebné výpočetní knihovny podle návodu. Žádné váhy se zatím nestáhly.",
    registry_conflict:"Nastavení se mezitím změnilo. Přehled jsme obnovili; zkontroluj jej a akci zopakuj.",
    reload_required:"Nejdřív použij změny registru ze souboru.",model_busy:"Právě probíhá jiná operace. Počkej na její dokončení.",
    active_model:"Nejdřív načti jiný model. Aktivní model nelze skrýt ani odpojit.",default_model:"Nejdřív nastav jiný výchozí model.",
    model_dependency:"Tento záznam potřebují jiné profily. Můžeš ho skrýt, ale zatím ho nelze odpojit.",
    target_exists:"Cílová složka už existuje, ale není úplná. Necháváme ji beze změny; zkontroluj ji ručně nebo zvol jiné umístění v registru.",
    profile_conflict:"Stejné ID už patří jinému nastavení. Existující model ani jeho cestu nepřepíšeme.",profile_disabled:"Tento profil je skrytý. Nejdřív ho vrať do nabídky.",
    disk_space:"Na disku není dost místa pro odhadované zdroje a výsledné váhy.",invalid_model_path:"Vlož celou cestu k adresáři modelu, ne cestu k jednotlivému souboru.",
    invalid_model_files:"Soubory modelu nejsou úplné nebo neodpovídají zvolenému prostředí.",invalid_adapter:"Tento adaptér neodpovídá konfiguraci modelu. Zkus automatickou volbu nebo základní měření.",
    duplicate_model:"Toto ID už je použité. Zvol jiné.",invalid_registration:"Zkontroluj název, ID a nastavení modelu.",
    plan_expired:"Plán už není aktuální. Nech si znovu ukázat rozsah instalace.",confirmation_required:"Nejdřív potvrď rozsah stahování.",
    load_failed:"Model se nepodařilo načíst. Zkontroluj formát vah, výpočetní knihovny a volnou paměť.",
    setup_failed:"Příprava se nezdařila. Původní váhy se nepřepsaly; podrobnosti najdeš níže.",
    invalid_output:"Výpočet nevrátil platné pravděpodobnosti. Model zatím nepoužívej pro měření.",
    restoration_failed:"Původní model se nepodařilo znovu načíst. Vyber jiný připravený model.",
    registry_unavailable:"Registr nelze přečíst. Zkontroluj soubor v technických podrobnostech.",
    demo_readonly:"V demu se skutečné nastavení nemění. Spusť aplikaci v běžném režimu.",
    registry_required:"Tento způsob spuštění nepoužívá registr. Spusť aplikaci běžným příkazem podle návodu."
  };
  const phases = {queued:"Čeká ve frontě",preparing:"Připravuji prostředí",download:"Stahuji původní váhy",convert:"Převádím váhy",compose:"Kontroluji zdroje Hybridu",loading:"Načítám model",checking:"Počítám zkušební distribuci",restoring:"Vracím původní model",done:"Hotovo"};
  const node = (tag, text, className) => { const element=document.createElement(tag); if(text!==undefined)element.textContent=text; if(className)element.className=className; return element; };
  const userNode = (tag, text, className) => {const element=node(tag,text,className);element.dataset.i18nIgnore="true";return element;};
  const formatBytes = (number) => number ? `${(number/1e9).toLocaleString(window.ProbabilityGardenI18n?.localeTag?.() || "cs",{maximumFractionDigits:1})} GB` : t("Bez dalšího stahování");
  const isBusy = () => pending || snapshot?.readonly || snapshot?.busy;
  function message(text="", error=null) {
    const target=$("admin-message"); target.replaceChildren(); target.classList.toggle("error",Boolean(error));
    if(text)target.append(node("span",text));
    if(error) {const details=node("details"); details.append(node("summary",t("Technické podrobnosti")),userNode("pre",[error.message, error.payload?.details ? JSON.stringify(error.payload.details,null,2):""].filter(Boolean).join("\n")));target.append(details);}
  }
  async function request(path, payload) {
    return api.fetchJson(path,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload),silent:true,timeoutMs:20000});
  }
  async function action(callback) {
    if(pending)return; pending=true; lockForms(); message();
    try {await callback();} catch(error) {message(t(errors[error.code] || "Akci se nepodařilo dokončit."),error);await refresh();}
    finally {pending=false;lockForms();}
  }
  function lockForms() {
    for(const id of ["inspect-model","review-installation","add-model","reload-registry"])$(id).disabled=Boolean(isBusy()) || (id==="add-model" && !inspection?.files_ok);
    $("start-installation").disabled=Boolean(isBusy()) || !$("confirm-download").checked || !plan;
    $("confirm-disconnect").disabled=Boolean(isBusy());
    for(const button of document.querySelectorAll("[data-admin-action]"))button.disabled=pending||button.dataset.blocked==="true";
  }
  function unavailableReason(id, model) {
    if(["tuned","jlens","detector"].includes(id))return "Vyžaduje samostatný kalibrovaný artefakt pro tento checkpoint.";
    if(snapshot.backend==="torch")return "Pohledy do vrstev zatím vyžadují podporovaný model v MLX.";
    if(["attribution","patching"].includes(id))return "Tento nástroj má další podmínky pro architekturu a přesnost Apertusu.";
    return "Pro tento adaptér zatím není pohled implementovaný.";
  }
  function renderCards() {
    const signature=JSON.stringify([snapshot.models,snapshot.busy,snapshot.readonly,snapshot.status]);
    if(signature===cardSignature)return;cardSignature=signature;
    const list=$("model-list");list.replaceChildren();
    for(const model of snapshot.models) {
      const card=node("article",undefined,`model-card${model.active?" active":""}`);card.dataset.modelId=model.id;
      const heading=node("div",undefined,"card-heading");heading.append(userNode("h3",model.label));card.append(heading);
      const badges=node("div",undefined,"badges");
      const badge=(text,kind="")=>badges.append(node("span",t(text),`badge ${kind}`));
      if(model.active)badge("Právě načtený");if(model.default)badge("Výchozí");if(!model.enabled)badge("Skrytý","plain");
      if(model.kind==="source")badge("Zdroj pro převod","plain");
      badge(model.check?.passed?"Výpočet ověřen":model.files_ok?"Soubory připraveny":"Chybí nebo nevyhovují soubory",model.files_ok?"":"warn");card.append(badges);
      const path=node("div",undefined,"model-path");path.append(userNode("code",model.path));card.append(path);
      if(model.layers)card.append(node("p",`${t("Vrstev")}: ${model.layers} · ${model.bytes?formatBytes(model.bytes):t("Velikost vah nezjištěna")}`,"muted small"));
      if(model.kind!=="source") {
        const support=node("details");support.append(node("summary",t("Co bude fungovat")));
        const items=node("ul",undefined,"support-list");
        for(const [id,label]of Object.entries(capabilityNames)) {const supported=model.capabilities.includes(id);const item=node("li",`${supported?"✓":"—"} ${t(label)}`,supported?"":"not-supported");if(!supported)item.append(node("small",t(unavailableReason(id,model))));items.append(item);}
        support.append(items);card.append(support);
      }
      const technical=node("details");technical.append(node("summary",t("Soubory a technické údaje")),userNode("pre",JSON.stringify({id:model.id,adapter:model.adapter,family:model.family,errors:model.errors,checkpoint:model.checkpoint,check:model.check,required_by:model.required_by},null,2)));card.append(technical);
      const actions=node("div",undefined,"actions card-actions");
      function button(text,callback,disabled=false,primary=false) {const b=node("button",t(text),primary?"primary":undefined);b.type="button";b.dataset.adminAction="true";b.dataset.blocked=String(Boolean(disabled||snapshot.readonly||snapshot.busy));b.disabled=pending||b.dataset.blocked==="true";b.addEventListener("click",()=>action(callback));actions.append(b);return b;}
      if(model.kind!=="source") {
        button(model.active?"Používá se":"Použít",async()=>{await request("/api/model",{model_id:model.id});message(t("Model se načítá. Stav se průběžně obnovuje."));await refresh();},!model.files_ok||!model.enabled||model.active,true);
        button("Ověřit výpočtem",async()=>{const job=await request("/api/model-admin/jobs",{action:"smoke",model_id:model.id,registry_sha256:snapshot.registry_sha256});installJob=job;renderJob(job);await refresh();},!model.files_ok||!model.enabled);
        if(!model.default)button("Nastavit jako výchozí",()=>edit("set_default",model.id),!model.enabled);
        button(model.enabled?"Skrýt z nabídky":"Vrátit do nabídky",()=>edit("set_enabled",model.id,{enabled:!model.enabled}),model.active||model.default);
      }
      const remove=button("Odpojit",()=>{disconnect=model;$("disconnect-name").textContent=model.label;$("disconnect-dialog").showModal();},model.active||model.default||model.required_by.length>0);
      if(model.required_by.length)remove.title=t("Tento záznam potřebují jiné profily.");
      card.append(actions);list.append(card);
    }
  }
  function renderJob(job) {
    if(!job){$("job-panel").hidden=true;return;}
    installJob=job;$("job-panel").hidden=false;
    const done=job.phase==="done";
    $("job-title").textContent=t(job.action==="smoke"?"Zkušební výpočet":"Příprava modelu");
    let label=t(phases[job.phase]||"Pracuji…");
    if(job.status==="cancelling")label=t("Přerušuji po dokončení bezpečného kroku…");
    if(done)label=t(job.status==="completed"?(job.action==="smoke"?"Načtení a zkušební výpočet proběhly v pořádku.":"Soubory jsou připravené. Model můžeš načíst tlačítkem Použít."):job.status==="cancelled"?"Operace byla přerušena. Dokončené váhy zůstaly na disku.":errors[job.error_code]||"Operace se nezdařila.");
    if(!done&&job.steps.length)label+=` · ${t("Krok")} ${job.step}/${job.steps.length}`;
    $("job-message").textContent=label;$("job-details").textContent=[job.error||"",...job.logs,job.result?JSON.stringify(job.result,null,2):""].filter(Boolean).join("\n");
    $("job-progress").hidden=done;if(job.steps.length){$("job-progress").max=job.steps.length;$("job-progress").value=Math.max(0,job.step-1);}else $("job-progress").removeAttribute("value");
    $("cancel-job").disabled=done||job.status==="cancelling";
  }
  async function refresh() {
    if(refreshRunning)return;refreshRunning=true;
    try {
      const value=await api.fetchJson("/api/model-admin",{silent:true});snapshot=value;
      $("demo-notice").hidden=!value.mock;$("legacy-notice").hidden=!value.readonly||value.mock;$("setup-notice").hidden=value.status!=="setup_required";
      $("backend-notice").hidden=value.backend_installed||value.readonly;
      $("backend-command").textContent=`python -m pip install -e '.[${value.backend}]'`;
      $("load-error-notice").hidden=value.status!=="error";
      $("load-error-details").textContent=value.error||"";
      $("reload-notice").hidden=value.registry_sha256===value.applied_sha256||value.readonly;
      $("registry-path").textContent=value.registry_path;$("cache-path").textContent=value.cache_path;$("backup-path").textContent=value.backup_directory;$("backend-name").textContent=value.backend;
      $("model-state-text").textContent=t(value.status==="setup_required"?"Vyber model, který chceš načíst.":value.busy?"Probíhá operace s modelem…":"Nastavení je připravené.");
      const current=$("profile-id").value;
      if(JSON.stringify(value.profiles.map(p=>p.id))!==JSON.stringify([...$("profile-id").options].map(p=>p.value))){$("profile-id").replaceChildren(...value.profiles.map(p=>{const option=node("option",profiles[p.id]||p.id);option.value=p.id;option.dataset.i18nIgnore="true";return option;}));if(current)$("profile-id").value=current;}
      renderCards();renderJob(value.job);lockForms();
    } catch(error) {message(t(errors[error.code]||"Nastavení se nepodařilo načíst."),error);}
    finally{refreshRunning=false;}
  }
  async function edit(actionName,modelId,extra={}) {snapshot=await request("/api/model-admin/edit",{action:actionName,model_id:modelId,registry_sha256:snapshot.registry_sha256,...extra});message(t("Nastavení bylo uloženo. Soubory vah se nezměnily."));cardSignature="";await refresh();}
  $("local-model-form").addEventListener("submit",event=>{event.preventDefault();action(async()=>{
    const body={path:$("model-path").value};if($("model-adapter").value)body.adapter=$("model-adapter").value;
    inspection=await request("/api/model-admin/inspect",body);$("inspection-result").hidden=false;$("inspection-summary").textContent=t(inspection.files_ok?"Složka obsahuje potřebné soubory.":"Složka zatím není připravená k připojení.");
    $("inspection-errors").replaceChildren(...inspection.errors.map(error=>userNode("li",error)));$("model-label").value=inspection.label;$("model-id").value=inspection.model_id;lockForms();
  });});
  for(const id of ["model-path","model-adapter"])$(id).addEventListener("input",()=>{inspection=null;$("inspection-result").hidden=true;lockForms();});
  $("add-model").addEventListener("click",()=>action(async()=>{if(!inspection?.files_ok)return;await edit("add",$("model-id").value,{path:inspection.path,label:$("model-label").value,adapter:inspection.adapter});$("inspection-result").hidden=true;inspection=null;}));
  $("install-profile-form").addEventListener("submit",event=>{event.preventDefault();action(async()=>{
    plan=await request("/api/model-admin/plan",{profile_id:$("profile-id").value,registry_sha256:snapshot.registry_sha256});$("install-size").textContent=`${t("Odhad nového místa na disku")}: ${formatBytes(plan.estimated_bytes)}`;
    $("install-steps").replaceChildren(...plan.steps.map(step=>{const item=node("li",undefined,"install-step");item.append(node("strong",profiles[step.model_id]||step.model_id),node("small",` · ${t(step.ready?"Už připraveno":phases[step.operation]||"Připravit")} · ${formatBytes(step.estimated_bytes)}`),userNode("code",step.path));return item;}));
    $("confirm-download").checked=false;$("install-dialog").showModal();
  });});
  $("confirm-download").addEventListener("change",lockForms);
  $("close-install-dialog").addEventListener("click",()=>$("install-dialog").close());
  $("start-installation").addEventListener("click",()=>action(async()=>{if(!plan||!$("confirm-download").checked)return;const job=await request("/api/model-admin/jobs",{action:"install",plan_id:plan.id,registry_sha256:plan.registry_sha256,confirmed:true});$("install-dialog").close();plan=null;renderJob(job);await refresh();}));
  $("cancel-job").addEventListener("click",()=>action(async()=>{if(installJob)renderJob(await request("/api/model-admin/cancel",{job_id:installJob.id}));}));
  $("cancel-disconnect").addEventListener("click",()=>$("disconnect-dialog").close());
  $("confirm-disconnect").addEventListener("click",()=>action(async()=>{if(disconnect)await edit("remove",disconnect.id);$("disconnect-dialog").close();disconnect=null;}));
  $("refresh-models").addEventListener("click",()=>refresh());
  $("reload-registry").addEventListener("click",()=>action(async()=>{await request("/api/model-admin/reload",{registry_sha256:snapshot.registry_sha256});await refresh();message(t("Změny byly použity."));}));
  const desktop=window.apertusDesktop;
  if(desktop?.chooseModelDirectory){$("choose-model-directory").hidden=false;$("browser-path-help").hidden=true;$("choose-model-directory").addEventListener("click",()=>action(async()=>{const result=await desktop.chooseModelDirectory();if(!result.canceled){$("model-path").value=result.path;$("model-path").dispatchEvent(new Event("input"));}}));}
  refresh();const timer=window.setInterval(()=>{if(!document.hidden)refresh();},1500);window.addEventListener("pagehide",()=>clearInterval(timer),{once:true});
})();
