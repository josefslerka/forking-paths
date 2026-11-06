(() => {
  "use strict";

  // Czech remains the authored source of truth. The English catalogue is
  // generated offline and then curated below; runtime localization never
  // calls a translation service. Experimental prompts, observed tokens and
  // model completions are protected because translating them would change
  // the object being measured.

  const STORAGE_KEY = "probability-garden:language";
  const SUPPORTED = new Set(["cs", "en"]);
  const catalogue = window.ProbabilityGardenEnglishCatalogue ?? {};
  const crossPageCatalogue = {};
  for (const map of Object.values(catalogue)) {
    for (const [source, translated] of Object.entries(map ?? {})) {
      if (!(source in crossPageCatalogue)) {
        crossPageCatalogue[source] = translated;
      }
    }
  }

  const overrides = new Map(Object.entries({
    "Rozcestí": "Forking Paths",
    "Rozcestí – domů": "Forking Paths — home",
    "Výzkum a výuka": "Research and teaching",
    "Výzkumná a výuková laboratoř jazykových modelů": "A research and teaching workbench for exploring language models",
    "Technické detaily vstupních embeddingů": "Input embedding technical details",
    "Technické detaily kontextového stavu": "Contextual state technical details",
    "Kontrolní náhled prvních 16 souřadnic posledního tokenu; nejde o významové rysy.": "Inspection preview of the last token's first 16 coordinates; these are not semantic features.",
    "Kontrolní náhled prvních 16 souřadnic; nejde o významové rysy.": "Inspection preview of the first 16 coordinates; these are not semantic features.",
    "Kontrolní náhled souřadnic kontextového stavu": "Inspection preview of contextual state coordinates",
    "Každý token má vlastní vstupní embedding. L0 zde ukazuje pouze embedding posledního střepu, ne průměr za celé slovo. Od L1 sledujeme stav na konci slova, který už může využívat jeho předchozí střepy i text nalevo.": "Each token has its own input embedding. Here, L0 shows only the last fragment's embedding, not a whole-word average. From L1 onwards we track the state at the end of the word, which can use its preceding fragments and the text to the left.",
    "04 · Kontextové stavy": "04 · Contextual states",
    "05 · Geometrie vstupu": "05 · Input geometry",
    "06 · Průběh změny": "06 · Change across layers",
    "Stejný začátek, dva kontextové stavy": "One shared start, two contextual states",
    "Od vstupu ke kontextovému stavu": "From input to contextual state",
    "Oba panely ukazují stejnou vrstvu a používají stejné osy i měřítko. Změnou vrstvy sleduješ, jak se společný vstup proměňuje v každém kontextu.": "Both panels show the same layer and use the same axes and scale. Change the layer to see how the shared input develops in each context.",
    "Přepínáním vrstev sleduješ, jak se vstupní kód mění při zpracování předchozího textu. Osy i měřítko obrázku zůstávají stejné.": "Switch layers to see how the input representation changes as the preceding text is processed. The axes and scale stay fixed.",
    "Vrstva modelu": "Model layer",
    "Společný vstup · L0": "Shared input · L0",
    "Vstup · L0": "Input · L0",
    "Největší rozdíl": "Largest difference",
    "L0 je vstup před první vrstvou. Další vrstvy už mohou využívat předchozí text.": "L0 is the input before the first layer. Subsequent layers can use the preceding text.",
    "Body jsou zjednodušený pohled na číselné stavy, nikoli mapa významů. Jejich vzdálenost na obrázku může klamat; uvedená kosinová vzdálenost se počítá ze všech rozměrů.": "The points are a simplified view of numerical states, not a map of meanings. Distances in the image can be misleading; the reported cosine distance uses all dimensions.",
    "Společná vstupní mapa": "Shared input map",
    "Tato mapa ukazuje vstupní embeddingy tokenů, ještě bez vlivu okolní věty. Proto je pro stejné tokeny v obou kontextech jen jedna. U vícetokenového slova jsou to jeho jednotlivé části, ne jeden vektor celého slova.": "This map shows input token embeddings, before the surrounding sentence has an effect. The same tokens therefore share one input map across both contexts. For a multi-token word these are its individual fragments, not one whole-word vector.",
    "Kontextový stav": "Contextual state",
    "Sledujeme stav na konci slova:": "Tracking the state at the end of the word:",
    "Sledujeme stav na pozici tokenu:": "Tracking the state at the token position:",
    "Společná projekce kontextového stavu": "Shared projection of a contextual state",
    "Barevný bod je stav v právě zvolené vrstvě; šedé čáry jsou obě cesty zpracování.": "The coloured point is the state at the selected layer; the grey lines show both processing paths.",
    "Barevný bod je stav v právě zvolené vrstvě; šedá čára je celá cesta zpracování.": "The coloured point is the state at the selected layer; the grey line shows the full processing path.",
    "Rozměr stavu": "State dimension",
    "Na vstupu jsou oba stavy totožné.": "Both states are identical at the input.",
    "Rozdíl obou stavů v této vrstvě:": "Difference between the states at this layer:",
    "Proměna oproti vlastnímu vstupu:": "Change relative to the initial state:",
    "(kosinová vzdálenost).": "(cosine distance).",
    "Společný vstup obou kontextů": "Shared input for both contexts",
    "Vstup před zpracováním kontextu": "Input before contextual processing",
    "Každá z těchto částí má svůj pevný číselný kód (vstupní embedding), zatím bez vlivu okolní věty.": "Each fragment has a fixed numerical representation (input embedding), before the surrounding sentence has an effect.",
    "Tento token má v modelu svůj pevný číselný kód (vstupní embedding), zatím bez vlivu okolní věty.": "This token has a fixed numerical representation (input embedding) in the model, before the surrounding sentence has an effect.",
    "Oba kontexty začínají stejnými vstupními kódy. Níže už sledujeme dva samostatné kontextové stavy.": "Both contexts start from the same input representations. Below, we track two separate contextual states.",
    "Níže sledujeme, jak se z tohoto vstupu stává stav ovlivněný předchozím textem.": "Below, we track how this input develops into a state shaped by the preceding text.",
    "Společná projekce zachycuje": "The shared projection captures",
    "rozptylu směrů stavů ze všech zobrazených kontextů a vrstev. Pro oba panely se počítá jen jednou.": "of the variance in state directions across all displayed contexts and layers. It is fitted once for both panels.",
    "Steering na stejných příkladech": "Steering on matched examples",
    "Stejný text, dva modeloví čtenáři. Porovnej překvapení, účinek kontextu nebo steering a uchovej obě měření v jednom experimentu.": "The same text, two model readers. Compare surprisal, context effects or steering and retain both measurements in one experiment.",
    "Konstrukční kontrasty": "Direction-building contrasts",
    "Každý model odvodí vlastní směr ze stejných 3 až 8 párů. Obě strany každého páru musí v daném modelu končit stejným tokenem. Testovací věty použij odlišné.": "Each model derives its own direction from the same 3–8 pairs. Both sides of each pair must end with the same token within that model. Use different sentences for testing.",
    "Přidat kontrastní pár": "Add contrast pair",
    "Načíst tři testovací věty": "Load three test sentences",
    "Hloubka zásahu": "Intervention depth",
    "Čtvrtina vrstev": "One quarter of the layers",
    "Polovina vrstev": "Half of the layers",
    "Tři čtvrtiny vrstev": "Three quarters of the layers",
    "Poslední vrstva": "Last layer",
    "Síla α": "Strength α",
    "Počet kontrolních směrů": "Number of control directions",
    "Seed kontrolních směrů": "Control-direction seed",
    "Polovina odpovídá L16 z 32 u Apertusu a L21 ze 42 u EuroLLM. Síla se vztahuje k typickému rozdílu konstrukčních párů každého modelu. Kontrolní směry mají stejnou skutečnou normu zásahu uvnitř daného modelu.": "Half means L16 of 32 for Apertus and L21 of 42 for EuroLLM. Strength is relative to the typical contrast-pair difference within each model. Control directions match the effective intervention norm within that model.",
    "Nový testovací prefix": "New test prefix",
    "Záporné pokračování, včetně úvodní mezery": "Negative continuation, including its leading space",
    "Kladné pokračování, včetně úvodní mezery": "Positive continuation, including its leading space",
    "Více testovacích prefixů": "Multiple test prefixes",
    "Testovací zadání v JSON, nejvýše 12 položek": "Test items in JSON, up to 12 items",
    "Každá položka obsahuje id, test_text, negative_text a positive_text. Úvodní mezera patří do pokračování.": "Each item contains id, test_text, negative_text and positive_text. The leading space belongs to the continuation.",
    "Větve −α, 0 a +α skórují stejná pokračování. Kladný efekt znamená posun ke kladné variantě. Náhodné směry poskytují popisnou referenci; jejich pořadí není test statistické významnosti.": "The −α, 0 and +α branches score the same continuations. A positive effect indicates a shift towards the positive variant. Random directions provide a descriptive reference; their ranking is not a significance test.",
    "U steeringu porovnáváme změnu logaritmu poměru pravděpodobností celých stejných pokračování v bitech. Každý model má vlastní směr, vrstvu a škálu zásahu. Výsledek popisuje zásah do modelové predikce.": "For steering, we compare changes in the log probability ratio of the same whole continuations, in bits. Each model has its own direction, layer and intervention scale. The result describes an intervention in model predictions.",
    "Posun preference proti α = 0": "Preference shift relative to α = 0",
    "Shoda směru steeringového účinku": "Agreement in steering-effect direction",
    "Symetrický efekt": "Symmetric effect",
    "Kontroly ≥ efekt": "Controls ≥ effect",
    "Zásah / reziduum": "Shift / residual",
    "Kladný efekt znamená posun ke kladnému pokračování. Kontrolní směry poskytují popisnou referenci, nikoli test významnosti.": "A positive effect indicates a shift towards the positive continuation. Control directions provide a descriptive reference, not a significance test.",
    "Kontrolní směry": "Control directions",
    "Záporná strana": "Negative side",
    "Kladná strana": "Positive side",
    "Odebrat pár": "Remove pair",
    "Steering celých stejných pokračování s větvemi −α / 0 / +α a kontrolními směry.": "Steering of the same whole continuations with −α / 0 / +α branches and control directions.",
    "Porovnat steering dvou modelů na stejných příkladech": "Compare steering of two models on matched examples",
    "Apertus a EuroLLM · explorativní zásah do mezistavu": "Apertus and EuroLLM · exploratory hidden-state intervention",
    "Model počítá steering": "The model is computing steering",
    "Aktivní model nepodporuje steering.": "The active model does not support steering.",
    "Pravděpodobnostní zahrada": "Probability Garden",
    "Pravděpodobnostní zahrada – domů": "Probability Garden — home",
    "Pravděpodobnostní zahrada · Apertus 8B": "Probability Garden · Apertus 8B",
    "Výzkumný workbench": "Research workbench",
    "Import JSON": "Import JSON",
    "Export archivu JSON": "Export archive JSON",
    "Export archivu CSV": "Export archive CSV",
    "Surprisal textu": "Text surprisal",
    "Trajektorie vrstev": "Layer trajectories",
    "Kontext.": "Context.",
    "Rys": "Feature",
    "slov": "words",
    "Poloha": "Position",
    "Modely v2": "Models v2",
    "Rozhoduje": "Determines verdict",
    "Kalibrace AUC": "AUC calibration",
    "Alternativy": "Alternatives",
    "Prohodit A ↔ B": "Swap A ↔ B",
    "P pod A": "P under A",
    "P pod B": "P under B",
    "Porovnat kontexty": "Compare contexts",
    "Hranice predikce": "Prediction boundary",
    "Pozice v posloupnosti": "Position in sequence",
    "Úvodní mezera": "Leading space",
    "Unicode: mám / mám": "Unicode: mám / mám",
    "a zápis": "and the spelling",
    "Koncovka: zavalil / zavalily": "Ending: zavalil / zavalily",
    "Slovo.": "Word.",
    "01 · Kontexty": "01 · Contexts",
    "04 · Geometrie vstupu": "04 · Input geometry",
    "05 · Kontextualizace": "05 · Contextualization",
    "Sledovat vrstvami": "Track through layers",
    "Embeddingy · Apertus": "Embeddings · Apertus",
    "nikoli stejnou kontextovou reprezentaci.": "not the same contextual representation.",
    "Archiv": "Archive",
    "Archiv a export": "Archive and export",
    "Sestroj": "Construct",
    "Zahrada": "Garden",
    "Kontrast": "Contrast",
    "Distribuce": "Distribution",
    "Intervence": "Interventions",
    "Telemetrie": "Telemetry",
    "Chci tvrdit": "I want to claim",
    "Mapa workbenche": "Workbench map",
    "Attention × MLP": "Attention × MLP",
    "Readout mezistavu": "Intermediate-state readout",
    "Zafixuj jednotku.": "Fix the unit.",
    "Zkontroluj hranice textu.": "Check the text boundaries.",
    "Nezaměňuj": "Do not mistake",
    "Velký příspěvek za důkaz, že komponenta je sama nutná, dostačující nebo kauzální příčinou výsledku.": "a large contribution for evidence that the component is itself necessary, sufficient, or a causal driver of the outcome.",
    "Interchange intervence": "Interchange intervention",
    "mezera + Praha": "space + Praha",
    "Hledat": "Search",
    "Archiv ·": "Archive ·",
    "1 sloupec": "1 column",
    "5 + zbytek": "5 + remainder",
    "8 + zbytek": "8 + remainder",
    "10 + zbytek": "10 + remainder",
    "12 + zbytek": "12 + remainder",
    "Varianta targetu": "Target variant",
    "Nesoulad RAW / J‑LENS top 1": "RAW/J-LENS top-1 disagreement",
    "Top tokeny vrstvy": "Layer top tokens",
    "Zkontrolovat tokeny": "Check tokens",
    "0 < R < 1 · k A": "0 < R < 1 · toward A",
    "R ≈ 1 · obnova A": "R ≈ 1 · recovery of A",
    "Reference A · donor stavu": "Reference A · state donor",
    "Δ entropie": "Δ entropy",
    "Shoda top k": "Top-k overlap",
    "0 · identita": "0 · identity",
    "Norma posunu": "Shift norm",
    "+2 · k positive": "+2 · toward positive",
    "−2 · k negative": "−2 · toward negative",
    "cos k referenci": "cosine to reference",
    "Spustit steering": "Run steering",
    "Spustit layer sweep": "Run layer sweep",
    "Spustit sham kontrolu": "Run sham control",
    "Norma rezidua prefixu": "Prefix-residual norm",
    "Profil steerovatelnosti": "Steerability profile",
    "Norm-matched sham kontrola": "Norm-matched sham control",
    "Posun / norma rezidua prefixu": "Shift / prefix-residual norm",
    "Oblasti steeringu": "Steering areas",
    "zdroj zmrazen": "source frozen",
    "lze spustit": "ready to run",
    "Steered − baseline v bitech.": "Steered − baseline in bits.",
    "· první rozchod": "· first divergence",
    "Intervence, ne detektor konceptu": "Intervention, not a concept detector",
    "Tokeny a znaky.": "Tokens and characters.",
    "Perplexita textu": "Text perplexity",
    "Porovnat texty": "Compare texts",
    "Jednotka čtení výsledku": "Result unit",
    "Přepnutí nic nepřepočítává: stejnou stopu lze číst jako tokeny modelu nebo jako slova pro psycholingvistickou analýzu.": "Switching views does not recompute anything: the same trace can be read as model tokens or as words for psycholinguistic analysis.",
    "Jednotka zobrazení": "Display unit",
    "Tokeny · model": "Tokens · model",
    "Slova · psycholingvistika": "Words · psycholinguistics",
    "Psycholingvistické čtení po slovech": "Psycholinguistic word-level view",
    "Slovo je zde pozorovatelný ortografický úsek; tokeny se ukážou po rozkliknutí.": "Here, a word is an observable orthographic span; expand it to inspect its tokens.",
    "Psycholingvistické čtení textu": "Psycholinguistic view of the text",
    "Slova jsou hlavní jednotky; interpunkce zůstává vidět ve vlastních řádcích a tokenové střepy se ukážou po rozkliknutí.": "Words are the primary units; punctuation remains visible in separate rows, and token fragments can be expanded.",
    "Slovní souhrn": "Word-level summary",
    "Ortografické slovo": "Orthographic word",
    "Interpunkční jednotka": "Punctuation unit",
    "Interpunkce": "Punctuation",
    "interpunkce": "punctuation",
    "slovo": "word",
    "různé druhy jednotek": "different unit types",
    "Slovní surprisal": "Word surprisal",
    "Entropie před slovem": "Pre-word entropy",
    "Surprisal interpunkce": "Punctuation surprisal",
    "Surprisal jednotky": "Unit surprisal",
    "Entropie dalšího tokenu před slovem": "Next-token entropy before word",
    "Entropie dalšího tokenu před interpunkcí": "Next-token entropy before punctuation",
    "Entropie dalšího tokenu před jednotkou": "Next-token entropy before unit",
    "Entropie dalšího tokenu · A → B": "Next-token entropy · A → B",
    "ΔI oproti předchozímu slovu": "ΔI from previous word",
    "ΔI mezi slovy": "ΔI between words",
    "Mimo slovní ΔI": "Excluded from word-level ΔI",
    "nezahrnuto": "excluded",
    "Tokenové střepy a jejich pravděpodobnostní metriky": "Token fragments and their probabilistic metrics",
    "Střep": "Fragment",
    "I tokenu": "Token surprisal",
    "H před tokenem": "Pre-token entropy",
    "Pořadí": "Rank",
    "Ortografická slova": "Orthographic words",
    "Průměrné slovní I": "Mean word surprisal",
    "Medián slovního I": "Median word surprisal",
    "Průměrná H před slovem": "Mean pre-word entropy",
    "Průměrná entropie dalšího tokenu před slovem": "Mean next-token entropy before word",
    "Pozice · slovo": "Position · word",
    "Pozice · slovo / interpunkce": "Position · word / punctuation",
    "Pozice · jednotka A / B": "Position · unit A / B",
    "ΔI proti předchozímu": "ΔI from previous word",
    "Slov A / B": "Words A / B",
    "Interpunkce A / B": "Punctuation A / B",
    "Přesně srovnatelných pozic": "Exactly comparable positions",
    "Průměrné slovní I · A": "Mean word surprisal · A",
    "Průměrné slovní I · B": "Mean word surprisal · B",
    "Pozice · slovo A / B": "Position · word A / B",
    "Slovní I · A → B": "Word surprisal · A → B",
    "Surprisal interpunkce · A → B": "Punctuation surprisal · A → B",
    "Surprisal jednotky · A → B": "Unit surprisal · A → B",
    "ΔI · B − A": "ΔI · B − A",
    "H před slovem · A → B": "Pre-word entropy · A → B",
    "Export slovní tabulky": "Export word table",
    "Export psycholingvistické tabulky": "Export psycholinguistic table",
    "Slovní tabulka je úzký export pro statistickou analýzu po slovech.": "The word table is a tidy export for word-level statistical analysis.",
    "Psycholingvistická tabulka je úzký export slov a samostatné interpunkce; slovní souhrny zůstávají bez interpunkce.": "The psycholinguistic table is a tidy export of words and separate punctuation; word-level summaries exclude punctuation.",
    "Slovní surprisal je součet −log₂ p všech tokenových střepů slova. H před slovem je entropie tokenového slovníku před jeho prvním střepem. ΔI zde znamená změnu proti předchozímu ortografickému slovu; není to samostatný modelový odhad.": "Word surprisal is the sum of −log₂ p across all token fragments of the word. Pre-word entropy is the entropy of the token vocabulary before the word's first fragment. Here, ΔI is the change from the previous orthographic word, not a separate model estimate.",
    "Slovní surprisal je součet −log₂ p všech tokenových střepů slova. Entropie dalšího tokenu před slovem je entropie tokenového slovníku před prvním střepem — není to entropie distribuce příštího slova. Interpunkce má vlastní řádky, ale nevstupuje do slovních souhrnů ani do ΔI mezi sousedními slovy.": "Word surprisal is the sum of −log₂ p across all token fragments of the word. Next-token entropy before a word is the entropy of the token-vocabulary distribution before its first fragment—not the entropy of a next-word distribution. Punctuation has separate rows but is excluded from word summaries and ΔI between adjacent words.",
    "Slova a interpunkci ukazujeme v pořadí textu. ΔI = I(B) − I(A) počítáme jen pro přesně stejný Unicode řetězec stejného druhu na stejné pozici pozorované jednotky. Jde o popisný kontrast podmínek, nikoli o automatické jazykové zarovnání.": "Words and punctuation are shown in text order. ΔI = I(B) − I(A) is computed only for the exact same Unicode string of the same unit type at the same observed-unit position. This is a descriptive contrast between conditions, not automatic linguistic alignment.",
    "Interpunkce zůstává samostatnou pozorovanou jednotkou. Její surprisal je součtem tokenových I; do slovních průměrů ani slovního ΔI nevstupuje.": "Punctuation remains a separate observed unit. Its surprisal is the sum of token surprisals; it is excluded from word-level means and word-level ΔI.",
    "Slovní I je součet níže uvedených tokenových I. Uvedená entropie popisuje distribuci dalšího tokenu před prvním střepem slova, nikoli distribuci slov.": "Word surprisal is the sum of the token surprisals listed below. The reported entropy describes the next-token distribution before the word's first fragment, not a distribution over words.",
    "Hranice interpunkce není tokeny pokryta čistě; její surprisal ani entropii proto neodhadujeme.": "The punctuation boundary is not cleanly covered by tokens, so its surprisal and entropy are not estimated.",
    "Hranice slova není tokeny pokryta čistě; slovní I ani entropii proto neodhadujeme.": "The word boundary is not cleanly covered by tokens, so word surprisal and entropy are not estimated.",
    "ΔI = I(B) − I(A) ukazujeme jen pro přesně stejný Unicode řetězec na stejné pořadové pozici slova. Jde o popisný kontrast podmínek, nikoli o automatické jazykové zarovnání.": "ΔI = I(B) − I(A) is shown only for the exact same Unicode string at the same word position. This is a descriptive contrast between conditions, not automatic linguistic alignment.",
    "Psycholingvistické čtení: __PG_SLOT_0__ slov, __PG_SLOT_1__ s úplnými metrikami.": "Psycholinguistic view: __PG_SLOT_0__ words, __PG_SLOT_1__ with complete metrics.",
    "Psycholingvistické čtení: __PG_SLOT_0__ slov, __PG_SLOT_1__ s úplnými metrikami; __PG_SLOT_2__ interpunkčních jednotek.": "Psycholinguistic view: __PG_SLOT_0__ words, __PG_SLOT_1__ with complete metrics; __PG_SLOT_2__ punctuation units.",
    "Psycholingvistické srovnání: __PG_SLOT_0__ slov v A, __PG_SLOT_1__ slov v B; __PG_SLOT_2__ přesně srovnatelných pozic.": "Psycholinguistic comparison: __PG_SLOT_0__ words in A, __PG_SLOT_1__ words in B; __PG_SLOT_2__ exactly comparable positions.",
    "Psycholingvistické srovnání: __PG_SLOT_0__ slov v A, __PG_SLOT_1__ slov v B; __PG_SLOT_2__ / __PG_SLOT_3__ interpunkčních jednotek; __PG_SLOT_4__ přesně srovnatelných pozic.": "Psycholinguistic comparison: __PG_SLOT_0__ words in A, __PG_SLOT_1__ words in B; __PG_SLOT_2__ / __PG_SLOT_3__ punctuation units; __PG_SLOT_4__ exactly comparable positions.",
    "Hotový text · psycholingvistické čtení T = 1": "Existing text · psycholinguistic view at T = 1",
    "Trajektorie očekávání po slovech": "Word-by-word expectation trajectory",
    "Slovní surprisal ve dvou podmínkách": "Word surprisal across two conditions",
    "Mezerový úsek": "Whitespace-delimited span",
    "Mezerový úsek je pozorovatelný řetězec oddělený bílým místem, nikoli autoritativně určené slovo.": "A whitespace-delimited span is an observable string separated by whitespace, not an authoritative linguistic word.",
    "Řádek": "Line",
    "Větný úsek": "Sentence span",
    "Větný úsek (pravidlový)": "Rule-based sentence span",
    "Tatáž jednotka platí pro obě samostatné stopy.": "The same unit is used for both independent traces.",
    "A gap segment is an observable string separated by a space, not an authoritatively determined word. Tatáž jednotka platí pro obě samostatné stopy.": "A whitespace-delimited span is an observable string, not an authoritative linguistic word. The same unit is used for both independent traces.",
    "1–3 bity": "1–3 bits",
    "Hotový text · syrová distribuce T = 1": "Existing text · raw distribution at T = 1",
    "Například: vlastní odstavec, kontrola češtiny": "For example: your own paragraph, a Czech-language check",
    "Měřím realizované překvapení I a entropii H…": "Measuring realized surprisal I and entropy H…",
    "Hotovo — barvy používají pevné absolutní prahy surprisal v bitech.": "Done — colours use fixed absolute surprisal thresholds in bits.",
    "Křivka zpřístupňování sleduje první token cílového pokračování. Pravděpodobnost celého pokračování se počítá pouze za úplnými prefixy.": "The access curve tracks the first token of the target continuation. The probability of the full continuation is computed only after complete prefixes.",
    "Kontrast popisuje dvě podmíněné jazykové distribuce; není pravděpodobností faktu ani kauzálním zásahem.": "The contrast describes two conditional language distributions; it is neither a probability of the fact nor a causal intervention.",
    "Sestavuji pevně škálovanou tokenovou stopu…": "Constructing the fixed-scale token trace…",
    "Medián I mezerového úseku / nebílý znak": "Median span surprisal / non-whitespace character",
    "IQR mezerových úseků / nebílý znak": "Span-surprisal IQR / non-whitespace character",
    "MAD mezerových úseků / nebílý znak": "Span-surprisal MAD / non-whitespace character",
    "Legenda": "Legend",
    "Souhrn kontrastu": "Contrast summary",
    "Provenience J‑Lensu": "J-Lens provenance",
    "Tabulka trajektorie vrstev": "Layer-trajectory table",
    "Legenda heatmapy": "Heatmap legend",
    "Export layer sweepu": "Export layer sweep",
    "Export sham kontroly": "Export sham control",
    "Transformace textu A": "Transformation of text A",
    "Provenience kalibrace": "Calibration provenance",
    "Held-out metriky kalibrace": "Held-out calibration metrics",
    "bity": "bits",
    "Vývojová simulace": "Development simulation",
    "Vývojová simulace je připraven": "Development simulation is ready",
    "Vývojová simulace je připravena": "Development simulation is ready",
    "Vývojová simulace · vývojová simulace": "Development simulation · development simulation",
    "BF16 pouze": "BF16 only",
    "simulace": "simulation",
    "projekce": "projection",
    "souhrn": "summary",
    "Změna vedoucího tokenu.": "Change of leading token.",
    "stabilní odtud": "stable from here",
    "větné znaménko": "sentence punctuation",
    "nový řádek": "newline",
    "chyba": "error",
    "neuvedeno": "not specified",
    "Tokeny A / B": "Tokens A / B",
    "pozic ×": "positions ×",
    "vrstev =": "layers =",
    "Patch ukazuje kauzální dostatečnost transplantovaného stavu pro tento průchod, nikoli nezbytnost nebo jedinečnou přirozenou cestu.": "The patch demonstrates the causal sufficiency of the transplanted state for this run, not its necessity or a unique natural pathway.",
    "Přenáší se celý resid_pre vektor. Buňka proto není čistým atomem jednoho významu.": "The entire resid_pre vector is transferred, so the cell is not a clean atom of a single meaning.",
    "Pozice jsou zarovnány přesně podle modelového indexu; výpočet nepředpokládá slovní ekvivalenci rozdílných tokenů.": "Positions are aligned strictly by model index; the computation assumes no word-level equivalence between different tokens.",
    "· Patch ukazuje kauzální dostatečnost transplantovaného stavu pro tento průchod, nikoli nezbytnost nebo jedinečnou přirozenou cestu.": "· The patch demonstrates the causal sufficiency of the transplanted state for this run, not its necessity or a unique natural pathway.",
    "· Přenáší se celý resid_pre vektor. Buňka proto není čistým atomem jednoho významu.": "· The entire resid_pre vector is transferred, so the cell is not a clean atom of a single meaning.",
    "· Pozice jsou zarovnány přesně podle modelového indexu; výpočet nepředpokládá slovní ekvivalenci rozdílných tokenů.": "· Positions are aligned strictly by model index; the computation assumes no word-level equivalence between different tokens.",
    "8bit · vyvážený": "8-bit · balanced",
    "4bit · rychlý": "4-bit · fast",
    "Pohled": "View",
    "Pohledy": "Views",
    "Pohledy aplikace": "Application views",
    "Model": "Model",
    "Běh": "Run",
    "lokální": "local",
    "Lokální nástroj": "Local tool",
    "Lokální záznamy": "Local records",
    "Platné páry": "Valid pairs",
    "Vrstva": "Layer",
    "Vrstvy": "Layers",
    "Vrstva zásahu": "Intervention layer",
    "Aktivní prefix": "Active prefix",
    "Aktivní model": "Active model",
    "Načítám modely…": "Loading models…",
    "není na disku": "not on disk",
    "Připravuji rozhraní": "Preparing interface",
    "Tokenizace": "Tokenization",
    "Embeddingy": "Embeddings",
    "Překvapení textu": "Text surprisal",
    "Překvapení": "Surprisal",
    "AI detektor": "AI detector",
    "Kontrast kontextů": "Context contrast",
    "RAW vrstvy": "RAW layers",
    "RAW × TUNED": "RAW × TUNED",
    "RAW × J-LENS": "RAW × J-LENS",
    "Komponenty": "Components",
    "Kauzální přenos": "Causal transfer",
    "Steering": "Steering",
    "Průvodce workbenchem": "Workbench guide",
    "Průvodce": "Guide",
    "01 · Rozcestník": "01 · Road map",
    "02 · Orientace": "02 · Orientation",
    "03 · Vstup a společný základ": "03 · Input and common ground",
    "04 · Výstupní rovina": "04 · Output space",
    "05 · Dva kontexty": "05 · Two contexts",
    "06 · Vertikální řez": "06 · Vertical slice",
    "07 · Účetnictví průchodu": "07 · Forward-pass accounting",
    "08 · Intervence": "08 · Interventions",
    "09 · V praxi": "09 · In practice",
    "10 · Reprodukovatelnost": "10 · Reproducibility",
    "11 · Hranice výpovědi": "11 · Limits of interpretation",
    "12 · Rychlá reference": "12 · Quick reference",
    "Lokální runtime": "Local runtime",
    "Archiv měření": "Measurement archive",
    "Generace modelu": "Model generation",
    "I celkem": "Total I",
    "indexy": "indices",
    "znaky —": "characters —",
    "Populace souhrnu": "Summary population",
    "Jazyky": "Languages",
    "Held-out pozic": "Held-out positions",
    "Archiv je zatím prázdný.": "The archive is empty for now.",
    "Archiv zatím není dostupný.": "The archive is not available yet.",
    "vstupní jednotky": "input units",
    "výchozí geometrie": "initial geometry",
    "živé logity": "live logits",
    "živý výpočet": "live computation",
    "z cache": "from cache",
    "hotový text": "existing text",
    "stylometrie × Apertus": "stylometry × Apertus",
    "dvě pole": "two fields",
    "32 vrstev": "32 layers",
    "kalibrovaná čtečka": "calibrated readout",
    "průměrný Jacobian": "mean Jacobian",
    "attention × MLP": "attention × MLP",
    "stav A → průchod B": "state A → run B",
    "kauzální zásah": "causal intervention",
    "mapa metod": "map of methods",
    "lokální záznamy": "local records",
    "Směr": "Direction",
    "kontrasty a zmrazený zdroj": "contrasts and frozen source",
    "Okamžitý efekt": "Immediate effect",
    "baseline × steered": "baseline × steered",
    "Cesty": "Trajectories",
    "tři vlastní kontexty": "three separate contexts",
    "Kontrolované pokračování": "Controlled continuation",
    "Převzít baseline větev": "Use baseline branch",
    "Výřez top-k se změnil": "The top-k view has changed",
    "Výřez top-k změněný": "Top-k view changed",
    "jiný výřez top-k": "different top-k view",
    "společná cesta a matched páry": "shared path and matched pairs",
    "Specificita": "Specificity",
    "norm-matched sham reference": "norm-matched sham reference",
    "Lokalizace": "Localization",
    "lokální směry po vrstvách": "layer-specific directions",
    "doplň vstup": "complete input",
    "čeká na zdroj": "waiting for source",
    "Spustit řez": "Run slice",
    "Přepočítat řez": "Recompute slice",
    "Přepočítat": "Recompute",
    "Počítá se…": "Computing…",
    "Přeskočit na obsah": "Skip to content",
    "Přeskočit k analýze": "Skip to analysis",
    "Přejít k obsahu": "Skip to content",
    "Zrušit": "Cancel",
    "Zrušit čekání": "Cancel waiting",
    "Skrýt čekání": "Hide progress",
    "Zjišťuji stav modelu…": "Checking model status…",
    "Čekám na model.": "Waiting for model.",
    "Čekám na Apertus…": "Waiting for Apertus…",
    "Model se načítá…": "Model is loading…",
    "Model se přepíná…": "Switching model…",
    "Model se nenačetl": "Model failed to load",
    "Tokenová predikce není pravda.": "Token prediction is not a truth judgement.",
    "Zobrazení záměrně zachovává tokenovou rovinu. ␠ je mezera, ↵ nový řádek a ⇥ tabulátor. Agregované úseky níže jsou odvozený souhrn; syrové tokeny zůstávají primárním měřením modelu.": "The display deliberately preserves the token level. ␠ marks a space, ↵ a newline and ⇥ a tab. The aggregated spans below are derived summaries; raw tokens remain the model's primary measurement.",
    "Server není dostupný.": "Server is unavailable.",
    "aktuální": "current",
    "efektivně": "effectively",
    "nezasažená změna": "unapplied edit",
    "Připravená": "Ready",
    "Připraveno": "Ready",
    "__PG_SLOT_0__ je připraven": "__PG_SLOT_0__ is ready",
    "__PG_SLOT_0__ je připravena": "__PG_SLOT_0__ is ready",
    "Hotovo": "Done",
    "Aktivní": "Active",
    "Zaměřit": "Focus",
    "Teplota": "Temperature",
    "Zasadit": "Plant",
    "Odebrat": "Remove",
    "Hloubka": "Depth",
    "Entropie H": "Entropy H",
    "Masa mimo top 10": "Mass outside top 10",
    "otevřít větev": "open branch",
    "token z trajektorie · otevřít větev": "token from trajectory · open branch",
    "Načítám": "Loading",
    "Chybí": "Missing",
    "Neznámá": "Unknown",
    "Nekompatibilní": "Incompatible",
    "Neověřená": "Unverified",
    "Prefix": "Prefix",
    "Kontext": "Context",
    "Kontext A": "Context A",
    "Kontext B": "Context B",
    "Text": "Text",
    "Vstup": "Input",
    "Dva kontexty": "Two contexts",
    "Hotový text": "Existing text",
    "Povaha výsledku": "Nature of the result",
    "Praktický postup": "Practical workflow",
    "Text A": "Text A",
    "Text B": "Text B",
    "Token": "Token",
    "Tokeny": "Tokens",
    "Znaky": "Characters",
    "znak": "character",
    "znaky": "characters",
    "znaků": "characters",
    "Rozsah": "Range",
    "Souhrn": "Summary",
    "Výsledek": "Result",
    "Výpočet": "Computation",
    "Pravděpodobnost": "Probability",
    "Pravděpodobnosti": "Probabilities",
    "Entropie": "Entropy",
    "Perplexita": "Perplexity",
    "Efektivní šířka": "Effective width",
    "Celková variace": "Total variation",
    "Celková variační vzdálenost.": "Total variation distance.",
    "JS divergence": "JS divergence",
    "Doznívání": "Persistence",
    "Doznívání po společné cestě": "Persistence along the shared path",
    "Poznámka": "Note",
    "Poznámka k měření": "Measurement note",
    "Export měření": "Export measurement",
    "Export JSON": "Export JSON",
    "Export CSV": "Export CSV",
    "Stáhnout JSON": "Download JSON",
    "Uložit experiment": "Save experiment",
    "Otevřít archiv": "Open archive",
    "Vybraný token": "Selected token",
    "Vybraný token:": "Selected token:",
    "Společný token:": "Shared token:",
    "Vybrané slovo:": "Selected word:",
    "Společné slovo:": "Shared word:",
    "Zadej text": "Enter text",
    "volitelné porovnání": "optional comparison",
    "povinný": "required",
    "Rozbalit přesnou provenienci": "Show full provenance",
    "Diagnostika, použitá data a interpretační omezení": "Diagnostics, data, and interpretive limits",
    "Jak výsledek číst": "How to read the result",
    "Co přesně se měří": "What is measured",
    "rozbalit metodickou poznámku": "show methodological note",
    "Finální top 1": "Final top 1",
    "Vrstva po vrstvě": "Layer by layer",
    "Tokenizace vstupu": "Input tokenization",
    "Tokenová stavba": "Token composition",
    "Unicode zápisu tokenizuje jinak, změř oba tvary odděleně.": "Unicode representation tokenizes differently; measure the two forms separately.",
    "Tokenizace hotová, ale texty nemají společné tokenové ID.": "Tokenization complete, but the texts share no token ID.",
    "Společné pokračování": "Shared continuation",
    "Kontrola součtu": "Sum check",
    "do veličiny": "to the quantity",
    "touž finální RMSNorm": "the same final RMSNorm",
    "Nejsilnější komponenta": "Strongest component",
    "Jeden realizovaný forward": "One realized forward pass",
    "Přímá atribuce": "Direct attribution",
    "Přesná cesta": "Exact path",
    "Celé pole": "Full field",
    "Mezivrstvová čtečka": "Intermediate-layer readout",
    "Mezivrstvové čtečky": "Intermediate-layer readouts",
    "Naučený překlad": "Learned transformation",
    "Reziduální proud": "Residual stream",
    "Teacher forcing": "Teacher forcing",
    "Čtečka": "Readout",
    "Readout / čtečka": "Readout",
    "Slovník": "Vocabulary",
    "kalibrace bez ID": "calibration without an ID",
    "úsek oddělený mezerami": "whitespace-delimited span",
    "neprázdný řádek": "non-empty line",
    "větně podobný úsek": "sentence-like span",
    "Celkem": "Total",
    "Minimum": "Minimum",
    "Maximum": "Maximum",
    "Medián": "Median",
    "Průměr": "Mean",
    "Rozptyl": "Variance",
    "Shoda": "Agreement",
    "Neshoda": "Mismatch",
    "ano": "yes",
    "ne": "no",
    "žádná": "none",
    "bez změny": "unchanged",
    "Remíza": "Tie",
    "blíž": "closer",
    "dál": "farther",
    "zesílil": "increased",
    "zeslábl": "decreased",
    "s varováním": "with warning",
    "zastaralé": "stale",
    "Výsledek je zastaralý": "Result is stale",
    "nepozorován": "not observed",
    "Žádný pokles JS pod polovinu kořene nepřetrval do konce pozorovaného úseku.": "No drop in JS below half the root persisted to the end of the observed segment.",
    "Výsledek neodpovídá současnému vstupu nebo modelovému běhu. Spusťte porovnání znovu.": "This result does not match the current input or model run. Run the comparison again.",
    "Česky": "Czech",
    "Anglicky": "English",
    "Jazyk rozhraní": "Interface language",
    "začátek": "beginning",
    "střed": "middle",
    "konec": "end",
    "Překvapení / neprázdný znak": "Surprisal / non-whitespace character",
    "IQR tokenového překvapení": "IQR of token surprisal",
    "95. percentil tokenového překvapení": "95th percentile of token surprisal",
    "Koncentrace horních 5 %": "Top-5% concentration",
    "Překvapení − entropie": "Surprisal − entropy",
    "Párová stylometrie v5": "Paired stylometry v5",
    "AI přepis · celkem": "AI rewrite · total",
    "Bez AI signálu": "No AI signal",
    "Přísný AI signál nezachycen": "No strict AI signal detected",
    "Nerozhoduje · gate neprošla": "Non-decisive · gate failed",
    "Skutečný původ": "Ground truth",
    "Vývojové AUC": "Development AUC",
    "výzkumně tedy signál Apertusu podporuje.": "thus supports the Apertus signal as a research diagnostic.",
    "Text výrazně míchá písma nebo používá jiné než latinské písmo.": "The text mixes scripts extensively or uses a non-Latin script.",
    "Text nevypadá jako čeština; anglická funkční slova převažují.": "The text does not appear to be Czech; English function words predominate.",
    "Text obsahuje náhradní znak Unicode a může být poškozený.": "The text contains a Unicode replacement character and may be corrupted.",
    "Bez srovnání kvůli quality gate": "No comparison because of the quality gate",
    "Shoda: oba přísný signál": "Agreement: both show a strict signal",
    "Shoda: oba v šedém pásmu": "Agreement: both are in the grey zone",
    "Shoda: oba pod screeningem": "Agreement: both are below screening",
    "Neshoda: výš je párová stylometrie": "Disagreement: paired stylometry is higher",
    "Neshoda: výš je Apertus": "Disagreement: Apertus is higher",
    "Obě skóre zůstávají diagnostická; vstup neprošel kontrolou kvality.": "Both scores remain diagnostic; the input did not pass the quality check.",
    "Apertus a párová stylometrie v5 leží ve stejném pásmu svých vlastních prahů.": "Apertus and paired stylometry v5 fall in the same band relative to their respective thresholds.",
    "Modely leží v různých pásmech. Jde o výzkumný záznam neshody; hlavní verdikt se řídí pouze Apertusem.": "The models fall in different bands. This is a research record of disagreement; only Apertus determines the primary verdict.",
    "Text se neukládá; lokálně se ukládá jen technický čas běhu bez obsahu textu.": "The text is not stored; only runtime timing without text content is saved locally.",
    "Text byl analyzován lokálně a neukládá se.": "The text was analyzed locally and is not stored.",
    "Uložené experimenty": "Saved experiments",
    "Kdo tlačí token?": "What pushes a token?",
    "Kontrast komponent": "Component contrast",
    "Waterfall příspěvků": "Contribution waterfall",
    "Aditivní trajektorie": "Additive trajectory",
    "Lokální RAW čtení": "Local RAW readout",
    "Kontrakt měření": "Measurement specification",
    "Největší posuny": "Largest shifts",
    "Rozkládám kontrast": "Decomposing contrast",
    "Jedno pokračování, dva kontexty": "One continuation, two contexts",
    "Informační výhoda A − B": "Information advantage A − B",
    "Kumulativní výhoda A − B": "Cumulative advantage A − B",
    "Trvalý pokles pod polovinu": "Sustained fall below half",
    "A je proti B pod rozsahem float64.": "The A/B ratio underflows float64.",
    "Pole před prvním tokenem cíle": "Distribution before the target's first token",
    "Kdy se cíl zpřístupňuje": "When does the target become accessible?",
    "Jak rozdíl doznívá": "How does the difference decay?",
    "Token po tokenu": "Token by token",
    "Text k analýze": "Text to analyze",
    "Kdo má jakou roli": "What each component does",
    "Stylometrie v2 · původní skóre": "Stylometry v2 · original score",
    "Experimentální fúze v2": "Experimental fusion v2",
    "Popisný profil Apertusu": "Descriptive Apertus profile",
    "Rozšířená confusion matice": "Extended confusion matrix",
    "Největší příspěvky párové stylometrie v5": "Largest contributions from paired stylometry v5",
    "Apertus okna": "Apertus windows",
    "Od identity k hodnotě": "From identity to value",
    "Vyber jednotku v textu": "Choose a unit in the text",
    "Rozdělit na tokeny": "Tokenize",
    "Rozdělit přesně": "Tokenize precisely",
    "Tokenizer je připraven": "Tokenizer is ready",
    "Tokenizer je připraven. Analýza nepoužije váhy modelu.": "The tokenizer is ready. This analysis does not use the model weights.",
    "4 · Tokenová ID": "4 · Token IDs",
    "Pole se otevře nad přesnými ID této hranice.": "The field will open from the exact IDs at this boundary.",
    "Server vrátil neplatný tokenizační řez.": "The server returned an invalid tokenization slice.",
    "Sledovat tento token od embeddingu": "Track this token from its embedding",
    "Výchozí embeddingový řádek": "Initial embedding row",
    "Toto je společná výchozí podmínka obou výskytů.": "This is the shared initial condition of both occurrences.",
    "Toto je výchozí podmínka zvoleného výskytu. Jednotlivé": "This is the initial condition of the selected occurrence. Individual",
    "Který výskyt sledujeme?": "Which occurrence are we tracking?",
    "Jeden začátek, jedna cesta": "One starting point, one path",
    "Co leží poblíž?": "What lies nearby?",
    "Lokální konstelace": "Local constellation",
    "Nejbližší řádky": "Nearest rows",
    "Rozbalovací síť tokenových řádků": "Expandable network of token rows",
    "Jak se stav promění?": "How does the state change?",
    "Proměna kontextového stavu": "Contextual-state transformation",
    "Ne word2vec": "Not word2vec",
    "Soused není synonymum": "A neighbor is not a synonym",
    "Rozchod není význam sám": "Divergence is not meaning itself",
    "Začni otázkou, ne nástrojem": "Start with a question, not a tool",
    "Co model skutečně dostane?": "What does the model actually receive?",
    "Co může přijít právě teď?": "What could come next right now?",
    "Kde text model překvapil?": "Where did the text surprise the model?",
    "Co zpřístupnila změna prefixu?": "What did the change in prefix make accessible?",
    "Jak se mění mezivrstvový readout?": "How does the intermediate-layer readout change?",
    "Jak obraz závisí na překladači?": "How does the picture depend on the readout?",
    "Které komponenty tlačily A proti B?": "Which components pushed A against B?",
    "Který stav A dokáže změnit průchod B?": "Which state from A can change run B?",
    "Co se změní po zásahu?": "What changes after an intervention?",
    "Jak výsledek zachovat?": "How can the result be preserved?",
    "Od pozorování k zásahu": "From observation to intervention",
    "Nejdřív text, potom token, teprve pak distribuce": "First the text, then the token, and only then the distribution",
    "Pole možností a hotová cesta": "A field of possibilities and a realized path",
    "Co se zpřístupnilo, přestavělo a přetrvalo": "What became accessible, was rearranged, and persisted",
    "Tentýž mezistav, různé čtečky": "The same intermediate state, different readouts",
    "Kdo tlačí token A proti tokenu B?": "What pushes token A against token B?",
    "Přenes nebo změň stav a změř následek": "Transfer or alter a state and measure the consequence",
    "Kauzální přenos A → B": "Causal transfer A → B",
    "A/B kontrolní přepočet": "A/B control recomputation",
    "Kauzální přenos dokončen": "Causal transfer complete",
    "Kauzální přenos selhal": "Causal transfer failed",
    "Kauzální přenos · Apertus": "Causal transfer · Apertus",
    "Počítám kauzální přenos": "Computing causal transfer",
    "Spustit kauzální přenos": "Run causal transfer",
    "Steering aktivací": "Activation steering",
    "Jeden experiment, jedna otázka": "One experiment, one question",
    "Archiv je výzkumná paměť, ne galerie": "The archive is research memory, not a gallery",
    "Co jednotlivé obrazovky nedokazují": "What the individual views do not prove",
    "Slovníček, vzorce a lokální model": "Glossary, formulas, and the local model",
    "Apertus 8B v několika režimech": "Apertus 8B in several modes",
    "Promptové zahrady": "Prompt gardens",
    "Porovnání aktivních uzlů A → B": "Comparison of active nodes A → B",
    "Vyhledat a porovnat token": "Find and compare a token",
    "Jacobianová čtečka": "Jacobian readout",
    "Nesoulad čteček": "Readout disagreement",
    "Uložit párový řez": "Save paired slice",
    "Zrození predikce": "The birth of a prediction",
    "Zrození predikce · RAW logit lens · Apertus": "The birth of a prediction · RAW logit lens · Apertus",
    "Apertus prosvěcuje vrstvy": "Apertus is reading the layers",
    "Prosvítit vrstvy": "Read the layers",
    "Napiš prefix, který chceš prosvítit.": "Enter a prefix to inspect across the layers.",
    "Nová tečka značí změnu vedoucího tokenu.": "A new dot marks a change in the leading token.",
    "Výpočet se spustí pouze tímto tlačítkem; koncové mezery se odstraní.": "The computation starts only with this button; trailing spaces are trimmed.",
    "__PG_SLOT_0__ · vývojová simulace": "__PG_SLOT_0__ · development simulation",
    "__PG_SLOT_0__ řádků · top __PG_SLOT_1__": "__PG_SLOT_0__ rows · top __PG_SLOT_1__",
    "L__PG_SLOT_0__ · rozchod": "L__PG_SLOT_0__ · divergence",
    "__PG_SLOT_0__ ID ve forwardu": "__PG_SLOT_0__ IDs in the forward pass",
    "Trajektorie po vrstvách": "Trajectory through the layers",
    "Souvislé běhy top 1": "Continuous top-1 runs",
    "Uložit trajektorii": "Save trajectory",
    "Přenést stav": "Transfer a state",
    "Transplantuj přirozený stav mezi zarovnanými prefixy a nech příjemce znovu doběhnout.": "Transplant a naturally occurring state between aligned prefixes, then run the recipient to completion again.",
    "Zarovnání modelových pozic": "Alignment of model positions",
    "Kde se stav přenáší?": "Where does the transferred state take effect?",
    "Recovery logitového kontrastu": "Recovery of the logit contrast",
    "Detail buňky": "Cell detail",
    "Přenáším stavy": "Transferring states",
    "Směr a konfigurace": "Direction and configuration",
    "Kontrastní páry": "Contrast pairs",
    "Distribuce dalšího tokenu": "Next-token distribution",
    "Největší přesuny tokenů": "Largest token shifts",
    "Trajektorie do konce věty": "Trajectories to the end of the sentence",
    "Specificita a lokalizace": "Specificity and localization",
    "Diagnostika zásahu": "Intervention diagnostics",
    "Deskriptivní z": "Descriptive z-score",
    "Vývoj": "Trend",
    "lze měřit": "ready to measure",
    "počítá se": "computing",
    "Kde text porušil očekávání": "Where the text violated expectations",
    "Pevná barevná škála překvapení I": "Fixed color scale for surprisal I",
    "Popisné srovnání A/B": "Descriptive A/B comparison",
    "Celkové I": "Total I",
    "Syrové tokeny": "Raw tokens",
    "Bitů / nebílý znak": "Bits / non-whitespace character",
    "IQR úseků / nebílý znak": "Span IQR / non-whitespace character",
    "Medián úseků / nebílý znak": "Span median / non-whitespace character",
    "Syrový MAD úseků / nebílý znak": "Raw span MAD / non-whitespace character",
    "Větám podobné úseky": "Sentence-like spans",
    "Agregované úseky": "Aggregated spans",
    "Jeden text, nebo přesný pár": "One text or an exact pair",
    "Od textu k tokenovým ID": "From text to token IDs",
    "Kde právě stojí model?": "Where is the model now?",
    "Token není slovo": "A token is not a word",
    "Koncový token": "Final token",
    "Roundtrip je kontrola": "The round trip is a consistency check",
    "Tokenizer předchází modelu": "The tokenizer precedes the model",
    "Kalibrace čtečky": "Readout calibration",
    "Srovnání čteček": "Readout comparison",
    "Validace čtečky mimo trénink": "Readout validation outside the training data",
    "Uložit srovnání čteček": "Save readout comparison",
    "shoda s finálním top 1": "agreement with final top-1",
    "Apertus počítá RAW × TUNED": "Apertus is computing RAW × TUNED",
    "Lokální výzkumná paměť": "Local research memory",
    "Neměnné řezy zahradou a vrstvami, které lze znovu číst bez inference a odnést jako reprodukovatelnou zálohu.": "Immutable snapshots of the garden and its layers that can be revisited without inference and exported as a reproducible record.",
    "Apertus · účetnictví reziduálního proudu": "Apertus · residual-stream accounting",
    "Rozlož kontrast dvou možných pokračování na skutečné vektory, které do průběžného stavu přidaly attention a MLP v každé vrstvě. Nečteme váhy pozornosti; sledujeme, kam míří její výstup.": "Decompose the contrast between two possible continuations into the actual vectors added to the residual stream by attention and the MLP at each layer. We do not read attention weights; we track the direction of the resulting output.",
    "Apertus · vztah dvou polí": "Apertus · relating two fields",
    "Dva přesné prefixy, jedno společné pokračování. Sleduj, co se zpřístupní, jak se přestaví celé pole dalších tokenů a jak dlouho si společná cesta nese rozdíl předchozího kontextu.": "Two exact prefixes, one shared continuation. Observe what becomes accessible, how the entire next-token distribution is rearranged, and how long the shared path retains the effect of the preceding context.",
    "Champion–challenger · výzkumný provoz": "Champion–challenger · research mode",
    "Vlož český text. Přísný verdikt řídí pět rysů překvapení Apertusu. Párová stylometrie v5 běží vedle něj jako výzkumný challenger; ukáže druhý pohled, ale verdikt nezmění.": "Paste a Czech text. Five Apertus surprisal features determine the conservative verdict. Paired stylometry v5 runs alongside it as a research challenger: it offers a second view but does not change the verdict.",
    "Apertus · od tabulky ke kontextu": "Apertus · from lookup table to context",
    "Výchozí embeddingový řádek tokenu, jeho sousedé ve slovníku a rozchod kontextových stavů v modelu Apertus.": "The token's initial embedding row, its neighbors in the vocabulary, and the divergence of contextual states in Apertus.",
    "Začni jedním slovem nebo větou. Vyber token či celé tokenizované slovo a sleduj, jak se jeho vstupní řádky mění v kontextový stav. Druhý kontext přidej teprve tehdy, když chceš porovnat dvě cesty.": "Start with a single word or a sentence. Select a token or a complete tokenized word and follow how its input rows develop into a contextual state. Add a second context only when you want to compare two paths.",
    "Rozložení je schematické: polohy uzlů nejsou souřadnice embeddingu. Autoritativní je pouze hrana a její kosinová podobnost. Uzly jsou tokeny slovníku, nikoli celá slova, synonyma nebo významy. Každý uzel se počítá jen po kliknutí, bez modelového forwardu.": "The layout is schematic: node positions are not embedding coordinates. Only an edge and its cosine similarity are authoritative. Nodes are vocabulary tokens—not whole words, synonyms, or meanings. A node's neighbors are computed only after it is clicked, without a model forward pass.",
    "Toto je výchozí podmínka zvoleného výskytu. Jednotlivé osy nemají samostatně pojmenovaný lidský význam.": "This is the selected occurrence's initial condition. The individual axes do not have separately named human meanings.",
    "Apertus 8B · výzkumný workbench": "Apertus 8B · research workbench",
    "Výchozím checkpointem je předtrénovaný základní kauzální model": "The underlying checkpoint is a pretrained base causal language model",
    ", nikoli chatová ani Instruct varianta. Hybrid je naše lokální experimentální implementace, ne oficiální varianta Apertusu.": ", not a chat or Instruct variant. Hybrid is our local experimental implementation, not an official Apertus variant.",
    "Lokální experiment: 8bitové affine tělo (group size 64) s přesným BF16 vstupním embeddingem a výstupní hlavou. Je výchozí a má připravenou TUNED čtečku.": "Local experiment: an 8-bit affine body (group size 64) with the exact BF16 input embedding and output head. It is the default and has a prepared TUNED readout.",
    "Mezivrstvový reziduální stav není hotová distribuce slov. Aby jej rozhraní zobrazilo jako tokeny, musí zvolit čtečku. Nesoulad čteček proto není technická nepříjemnost k ukrytí, ale součást měření.": "An intermediate-layer residual state is not a ready-made distribution over words. To display it as tokens, the interface must choose a readout. Readout disagreement is therefore part of the measurement, not a technical nuisance to hide.",
    "Nejde o jednu obrazovku, ale o": "This is not a single screen but",
    "soustavu různých řezů týmž modelem": "a set of distinct views into the same model",
    "Průvodce výzkumným workbenchem Apertus: pole očekávání, překvapení textu, kontrast kontextů, mezivrstvové čtečky, komponentové účetnictví, kauzální přenos a steering.": "A guide to the Apertus research workbench: fields of expectation, text surprisal, context contrast, intermediate-layer readouts, component accounting, causal transfer, and steering.",
    "Apertus má pro něj extrémně dlouhý řádek nevázané výstupní hlavy, který může saturovat RAW i J‑Lens readout, přestože je token ve skutečné finální distribuci na dně. Diagnostika normy řádku, finální rank a srovnání čteček zde nejsou dekorace — zabraňují tomu, aby se artefakt projekce vydával za „myšlenku modelu“.": "Apertus has an exceptionally high-norm row for this token in its untied output head. It can saturate both RAW and J-Lens readouts even though the token sits near the bottom of the actual final distribution. The row-norm diagnostic, final rank, and readout comparison are not decorative: they prevent a projection artefact from masquerading as a “thought of the model”.",
    "Na každý mezistav položí finální RMSNorm a výstupní hlavu modelu. Je jednoduchá a stejná pro všechny vrstvy, zvlášť dole však může být silně mimo distribuci, pro kterou byla hlava vycvičena.": "It applies the model's final RMSNorm and output head to every intermediate state. This readout is simple and identical across layers, but at early layers it may be far outside the distribution on which the output head was trained.",
    "Napříč workbenchem se opakují stejné stavební kameny. Jejich přesná podoba je důležitější než jejich přirozeně jazykové zdání.": "The same building blocks recur throughout the workbench. Their exact computational form matters more than their seemingly natural-language appearance.",
    "Mezivrstvová čtečka může selhat, účetnictví může být přesné a přesto nekauzální a steering může mít účinek, aniž by použitý směr byl čistým nositelem zamýšleného pojmu.": "An intermediate-layer readout can fail; the accounting can be exact yet non-causal; and steering can have an effect even when its direction is not a pure carrier of the intended concept.",
    "s kombinovanou čárkou mohou být kanonicky ekvivalentní Unicode, ale přesto vstoupit jako jiná tokenová posloupnost. Dostupné režimy Apertusu používají bitově totožný": "with a combining acute accent can be canonically equivalent in Unicode, yet still enter as a different token sequence. The available Apertus modes use a bit-identical",
    "drží dva prefixy A a B a vnucuje jim tutéž přesnou pokračovací cestu. Odděluje tři otázky, které nelze slít do jednoho skóre.": "compares prefixes A and B along the same exact continuation path. It separates three questions that cannot be collapsed into a single score.",
    "Modelově specifická low‑rank čtečka se učí přiblížit mezistav finální distribuci. „Blíž“ znamená nižší validační divergenci k finále, ne pravdivější interpretaci.": "A model-specific low-rank readout learns to map an intermediate state toward the final distribution. “Closer” means lower validation divergence from that final distribution, not a truer interpretation.",
    "8bitové tělo s BF16 vstupní a výstupní projekcí. Je výchozí a má připravenou TUNED čtečku.": "An 8-bit body with BF16 input and output projections. It is the default mode and has a ready TUNED readout.",
    "Nejde o jednu obrazovku, ale o soustavu různých řezů týmž modelem . Každý pohled odpovídá na jinou otázku a dovoluje jinak silné tvrzení: od popisu distribuce přes čtení mezistavů až po skutečný zásah.": "This is not a single display but a set of different views into the same model. Each answers a different question and supports a different strength of claim: from describing a distribution, through reading intermediate states, to an actual intervention.",
    ". Každý pohled odpovídá na jinou otázku a dovoluje jinak silné tvrzení: od popisu distribuce přes čtení mezistavů až po skutečný zásah.": ". Each view answers a different question and supports a different strength of claim: from describing a distribution, through reading intermediate states, to an actual intervention.",
    "Apertus 8B · živé logity": "Apertus 8B · live logits",
    "Rozvětvuj distribuci libovolným směrem. Jednou otevřené cesty zůstávají na ploše a každý uzel lze znovu zaměřit.": "Branch through the distribution in any direction. Once opened, paths remain on the canvas, and any node can become the focus again.",
    "Apertus · dvě čtečky téhož mezistavu": "Apertus · two readouts of the same intermediate state",
    "Porovnej finální výstupní hlavu nad syrovým reziduálem s týmž stavem přeneseným průměrným Jacobianem. Nesoulad čteček je měření, nikoli chyba ani automatický důkaz, že jedna z nich čte vrstvu správněji.": "Compare the final output head applied directly to the residual stream with the same state transported by the mean Jacobian. Readout disagreement is a measurement—not an error, nor automatic evidence that either readout interprets the layer more correctly.",
    "Apertus · vertikální řez": "Apertus · vertical slice",
    "Sleduj, co by model předpověděl po každé vrstvě — od prvního mezistavu až po finální distribuci.": "Follow what the model would predict after each layer, from the first intermediate state to the final distribution.",
    "Apertus · interchange intervention": "Apertus · interchange intervention",
    "Vezmi přirozený stav z reference A, vlož jej do zarovnané pozice příjemce B před zvolenou vrstvou a nech B znovu doběhnout. Heatmapa ukáže, kde transplantace skutečně posunula budoucí pole k A.": "Take a naturally occurring state from source A, insert it at the aligned position in recipient B before a selected layer, and run B to completion again. The heatmap shows where the transplant actually shifted the future distribution toward A.",
    "Apertus · explorativní zásah do mezistavu": "Apertus · exploratory intervention in an intermediate state",
    "Sestav směr z párových kontrastů a porovnej kontrolní a změněnou distribuci dalšího tokenu nad týmž testovacím promptem. Zásah pak můžeš sledovat ve volných větách, po jedné společné tokenové cestě nebo ve skóre celých párových pokračování.": "Construct a direction from paired contrasts and compare the baseline and intervened next-token distributions for the same test prompt. Then trace the intervention in free-running sentences, along one shared token path, or in scores for complete paired continuations.",
    "Apertus · teacher-forced čtení": "Apertus · teacher-forced readout",
    "Projdi už napsaný text token po tokenu a sleduj, jak pravděpodobný byl každý skutečně následující token v okamžiku, kdy jej model ještě neviděl. Barva ukazuje překvapení v pevné škále — nikoli pravdivost, kvalitu ani důležitost výrazu.": "Read an existing text token by token and observe how probable each actual next token was before the model had seen it. Color encodes surprisal on a fixed scale—not the truth, quality, or importance of an expression.",
    "Apertus · nulté patro workbenche": "Apertus · the workbench's ground floor",
    "Podívej se na přesné jednotky a ID, které pevný tokenizer vyrobí ještě před prvním průchodem modelem. Porovnej i dva lidsky podobné zápisy bez toho, abys jejich rozdíl schoval normalizací.": "Inspect the exact units and IDs produced by the fixed tokenizer before the model's first forward pass. Compare even two forms that look alike to a person, without hiding their difference through normalization.",
    "Apertus · kalibrovaný vertikální řez": "Apertus · calibrated vertical slice",
    "Polož dvě čtečky nad tentýž mezistav a sleduj, jak naučený překlad promění jeho vztah k finální distribuci.": "Apply two readouts to the same intermediate state and observe how a learned transformation changes its relation to the final distribution.",
    "Hotovo — dva forwardy za __PG_SLOT_0__ ms.": "Done — two forward passes in __PG_SLOT_0__ ms.",
    "Počítám dva přesné forwardy a společnou trajektorii…": "Computing two exact forward passes and the shared trajectory…",
    "Připravuji dva přesné forwardy…": "Preparing two exact forward passes…",
    "Stylometrie se počítá lokálně; Apertus provede tři teacher-forced průchody.": "Stylometry is computed locally; Apertus performs three teacher-forced passes.",
    "Kontext __PG_SLOT_0__ zpřístupňuje přesné pokračování více.": "Context __PG_SLOT_0__ makes the exact continuation more accessible.",
    "Přesnou cestu zvýhodňuje kontext __PG_SLOT_0__.": "Context __PG_SLOT_0__ favors the exact path.",
    "Přesná tokenizace · A __PG_SLOT_0__ ·": "Exact tokenization · A __PG_SLOT_0__ ·",
    "Rank finále RAW → __PG_SLOT_11__": "Final-token rank: RAW → __PG_SLOT_11__",
    "J‑LENS vůči RAW podle JS k finále": "J-LENS versus RAW by JS divergence to the final distribution",
    "Vzdálenost k finále podle JS": "Distance to the final distribution by JS divergence",
    "finále: __PG_SLOT_7__ · I __PG_SLOT_8____PG_SLOT_9__": "final output: __PG_SLOT_7__ · I __PG_SLOT_8____PG_SLOT_9__",
    "Pravděpodobnost může být po softmaxu numericky 0 nebo 100 %. Překvapení I = −log₂ p a odstup od vítěze jsou počítané stabilně přímo z logitů; proto zůstávají čitelné i při saturaci. „Finále“ ukazuje rank a překvapení téhož tokenu po posledním bloku modelu, ne další odhad čtečky. ‖w": "After softmax, a probability can be numerically 0 or 100%. Surprisal I = −log₂ p and the gap from the winner are computed stably from the logits, so they remain readable even under saturation. “Final output” gives the rank and surprisal of that same token after the model's last block—not another readout estimate. ‖w",
    "__PG_SLOT_0__ dávek už server bezpečně zpracoval; opakování je idempotentní.": "The server has already safely processed __PG_SLOT_0__ batches; repeating the import is idempotent.",
    "Δ je definováno jako __PG_SLOT_4__. Tabulka pouze přepisuje uložené souhrnné hodnoty; jednotlivé tokenové řady A a B zůstávají oddělené.": "Δ is defined as __PG_SLOT_4__. The table only transcribes the stored summary values; the individual token sequences for A and B remain separate.",
    "JS → finále": "JS → final distribution",
    "JS k finále RAW → TUNED": "JS to final distribution: RAW → TUNED",
    "Sestavuji společné metriky k finále…": "Computing shared metrics against the final distribution…",
    "má po kalibraci nižší průměrnou JS divergenci k finále.": "has lower mean JS divergence to the final distribution after calibration.",
    "Rank finálního tokenu RAW → TUNED": "Final-token rank: RAW → TUNED",
    "Medián ranku finále RAW → TUNED": "Median final-token rank: RAW → TUNED",
    "TUNED blíž k finále podle KL": "TUNED closer to the final distribution by KL",
    "Živé srovnání RAW a TUNED čtečky": "Live comparison of RAW and TUNED readouts",
    "Srovnání RAW a TUNED čtečky na held-out datech po vrstvách": "Layer-by-layer comparison of RAW and TUNED readouts on held-out data",
    "Rank finále": "Final-token rank",
    "Rozhodovací rezerva": "Decision margin",
    "Rozhodovací rezerva se kvůli quality gate nevykládá.": "The decision margin is not interpreted because the quality gate withheld a verdict.",
    "Analýza hotova; quality gate správně nevydal verdikt.": "Analysis complete; as intended, the quality gate withheld a verdict.",
    "Analýza hotova. Skóre patří přesně tomuto vloženému textu.": "Analysis complete. The score applies only to this exact pasted text.",
    "Čára sčítá komponenty společnou finální čtečkou. Vlastní měřítko nechává vyniknout, ve kterých vrstvách se kontrast skutečně začal stavět.": "The line accumulates component contributions through a shared final readout. Its own scale highlights the layers in which the contrast began to form.",
    "Čekání v prohlížeči bylo zrušeno. Již spuštěný modelový výpočet může doběhnout na serveru.": "Waiting in the browser was cancelled. A model computation already started may still finish on the server.",
    ". Každá buňka se proto poměřuje kontrolami z vlastního průchodu, ne baseline spočtenou jinou numerickou cestou.": ". Each cell is therefore compared with controls from its own pass, not with a baseline computed through a different numerical path.",
    "Úvodní mezera pokračování patří k cíli; koncové mezery se oříznou · Ctrl/⌘ + Enter": "The continuation's leading space belongs to the target; trailing spaces are trimmed · Ctrl/⌘ + Enter",
    "JSON zachová úplnou odpověď serveru. CSV rozepíše souhrn, zpřístupňování, doznívání i viditelný kořenový top-k.": "JSON preserves the complete server response. CSV expands the summary, access curve, persistence, and visible root top-k.",
    "Izotropní off-manifold sham není sémantický nulový směr. Když target překoná náhodné směry, ukazuje to strukturovanost nad obecný efekt stejně velkého zásahu — samo o sobě ještě ne jednoznačný význam „vlídnost“. Z-skóre je deskriptivní; při 20 shamech je nejmenší dosažitelné p": "An isotropic off-manifold sham is not a semantic null direction. If the target outperforms random directions, the effect is more structured than the generic effect of an equally large intervention; this alone still does not establish the intended meaning of “kindness”. The z-score is descriptive; with 20 sham directions, the smallest attainable p",
    "s korekcí +1 rovno 1/21 ≈ 0,0476. Protože target a kolmé shamy nejsou výměnné, jde o deskriptivní Monte Carlo tail fraction": "with the +1 correction is 1/21 ≈ 0.0476. Because the target and orthogonal sham directions are not exchangeable, this is a descriptive Monte Carlo tail fraction",
    "vůči zvolené referenci, nikoli o obecně kalibrovanou p-hodnotu. Vyšší počet shamů pouze zjemňuje tuto referenci; neměří nejistotu napříč held-out jazykovými položkami.": "relative to the chosen reference, not a generally calibrated p-value. More sham directions only refine this reference; they do not measure uncertainty across held-out language items.",
    "Cílový směr porovná s izotropními náhodnými směry kolmými na něj. Každý sham má tutéž normu aplikovaného posunu, stejnou vrstvu, α i teacher-forced páry. Primární veličinou je symetrický posun preference positive proti negative, vyjádřený v bitech na token.": "Compare the target direction with isotropic random directions orthogonal to it. Every sham uses the same applied-shift norm, layer, α, and teacher-forced pairs. The primary measure is the symmetric shift in preference for the positive continuation over the negative one, expressed in bits per token."
  }));

  const normalize = (value) => String(value ?? "")
    .replace(/\s+/g, " ")
    .trim();

  const urlLanguage = new URLSearchParams(window.location.search).get("lang");
  if (SUPPORTED.has(urlLanguage)) {
    try {
      window.localStorage.setItem(STORAGE_KEY, urlLanguage);
    } catch {
      // A private browser context may deny storage; the query still works.
    }
  }

  const readStoredLanguage = () => {
    if (SUPPORTED.has(urlLanguage)) return urlLanguage;
    try {
      const stored = window.localStorage.getItem(STORAGE_KEY);
      if (SUPPORTED.has(stored)) return stored;
    } catch {
      // Keep the Czech source language when storage is unavailable.
    }
    return "cs";
  };

  let language = readStoredLanguage();
  document.documentElement.lang = language;
  document.documentElement.dataset.language = language;

  const pageCatalogue = () => {
    const page = document.body?.dataset.workbenchPage;
    return catalogue[page] ?? {};
  };

  const fallbackCatalogue = (() => {
    const result = {};
    for (const page of Object.values(catalogue)) {
      for (const [source, translated] of Object.entries(page ?? {})) {
        if (!(source in result)) result[source] = translated;
      }
    }
    return result;
  })();

  const priorityPatterns = [
    [/^Kořen hotový za (.+?) ms · (\d+) token v prefixu$/i, "Root ready in $1 ms · $2 token in the prefix"],
    [/^Kořen hotový za (.+?) ms · (\d+) tokeny v prefixu$/i, "Root ready in $1 ms · $2 tokens in the prefix"],
    [/^Kořen hotový za (.+?) ms · (\d+) tokenů v prefixu$/i, "Root ready in $1 ms · $2 tokens in the prefix"],
    [/^Hloubka: (\d+) token$/i, "Depth: $1 token"],
    [/^Hloubka: (\d+) tokeny$/i, "Depth: $1 tokens"],
    [/^Hloubka: (\d+) tokenů$/i, "Depth: $1 tokens"],
    [/^Slovo (.+): 1 střep, 1 trajektorie$/i, "Word $1: 1 fragment, 1 trajectory"],
    [/^Slovo (.+): 1 střep, (\d+) trajektorie$/i, "Word $1: 1 fragment, $2 trajectories"],
    [/^Slovo (.+): (\d+) střepy, 1 trajektorie$/i, "Word $1: $2 fragments, 1 trajectory"],
    [/^Slovo (.+): (\d+) střepy, (\d+) trajektorie$/i, "Word $1: $2 fragments, $3 trajectories"],
    [/^Token (.+): jeden řádek, 1 trajektorie$/i, "Token $1: one row, 1 trajectory"],
    [/^Token (.+): jeden řádek, (\d+) trajektorie$/i, "Token $1: one row, $2 trajectories"],
    [/^Hotovo — (\d+) střepy a 1 trajektorie za (.+)\.$/i, "Done — $1 fragments and 1 trajectory in $2."],
    [/^Hotovo — (\d+) střepy a (\d+) trajektorie za (.+)\.$/i, "Done — $1 fragments and $2 trajectories in $3."],
    [/^Hotovo — (\d+) sousedů a 1 trajektorie za (.+)\.$/i, "Done — $1 neighbors and 1 trajectory in $2."],
    [/^Hotovo — (\d+) sousedů a (\d+) trajektorie za (.+)\.$/i, "Done — $1 neighbors and $2 trajectories in $3."],
    [/^1 tokeny · čtecí hranice #(\S+)$/i, "1 token · readout boundary #$1"],
    [/^(\d+) tokeny · čtecí hranice #(\S+)$/i, "$1 tokens · readout boundary #$2"],
    [/^(\d+) znak · (\d+) bajt$/i, "$1 character · $2 byte"],
    [/^(\d+) znak · (\d+) bajty$/i, "$1 character · $2 bytes"],
    [/^(\d+) znak · (\d+) bajtů$/i, "$1 character · $2 bytes"],
    [/^(\d+) znaky · (\d+) bajt$/i, "$1 characters · $2 byte"],
    [/^(\d+) znaky · (\d+) bajty$/i, "$1 characters · $2 bytes"],
    [/^(\d+) znaky · (\d+) bajtů$/i, "$1 characters · $2 bytes"],
    [/^(\d+) znaků · (\d+) bajt$/i, "$1 characters · $2 byte"],
    [/^(\d+) znaků · (\d+) bajty$/i, "$1 characters · $2 bytes"],
    [/^(\d+) znaků · (\d+) bajtů$/i, "$1 characters · $2 bytes"],
    [/^(\d+) znak · (\d+) B$/i, "$1 character · $2 B"],
    [/^(\d+) znaky · (\d+) B$/i, "$1 characters · $2 B"],
    [/^(\d+) znaků · (\d+) B$/i, "$1 characters · $2 B"],
    [/^(\d+) znaků · (\d+) UTF‑8 bajtů$/i, "$1 characters · $2 UTF-8 bytes"],
    [/^(\d+) kontextových ID včetně BOS · (\d+) UTF‑8 bajtů prefixu\. Pole se otevře nad přesnými ID této hranice\.$/i, "$1 context IDs including BOS · $2 UTF-8 bytes in the prefix. The field will open from the exact IDs at this boundary."],
    [/^Hotovo — dostupné modelové režimy mají bitově totožný tokenizer\.json\.$/i, "Done — the available model modes have a bit-identical tokenizer.json."],
    [/^1 znak · top (\d+) alternativ · výpočet začne až tlačítkem$/i, "1 character · top $1 alternatives · computation starts on button press"],
    [/^([\d +]+) znaky · top (\d+) alternativ · výpočet začne až tlačítkem$/i, "$1 characters · top $2 alternatives · computation starts on button press"],
    [/^([\d +]+) znaků · top (\d+) alternativ · výpočet začne až tlačítkem$/i, "$1 characters · top $2 alternatives · computation starts on button press"],
    [/^(.+? %) ze všech textů$/i, "$1 of all texts"],
    [/^(\d+) \+ (\d+) znaků prefixu · (\d+) znaků cíle$/i, "$1 + $2 prefix characters · $3 target characters"],
    [/^Vývojová simulace (.+) je připravena\.$/i, "Development simulation $1 is ready."],
    [/^(\d+) kalibrovaných vrstev · výpočet pouze po stisknutí tlačítka · koncové mezery se odstraní\.$/i, "$1 calibrated layers · computation starts only on button press · trailing spaces are trimmed."],
    [/^(\d+) kalibrovaných vrstev · top (\d+) na čtečku$/i, "$1 calibrated layers · top $2 per readout"],
    [/^(\d+) platný · minimum (\d+)$/i, "$1 valid · minimum $2"],
    [/^(\d+) platné · minimum (\d+)$/i, "$1 valid · minimum $2"],
    [/^(\d+) pár · zásah po L(\d+) · α (.+?)\. Výpočet proběhne až po submitu\.$/i, "$1 pair · intervention after L$2 · α $3. Computation starts after submission."],
    [/^(\d+) páry · zásah po L(\d+) · α (.+?)\. Výpočet proběhne až po submitu\.$/i, "$1 pairs · intervention after L$2 · α $3. Computation starts after submission."],
    [/^(\d+) párů · zásah po L(\d+) · α (.+?)\. Výpočet proběhne až po submitu\.$/i, "$1 pairs · intervention after L$2 · α $3. Computation starts after submission."],
    [/^(\d+) stav · hodnoty nezávislé na slideru$/i, "$1 state · slider-independent values"],
    [/^(\d+) stavů · hodnoty nezávislé na slideru$/i, "$1 states · slider-independent values"],
    [/^(\d+) stav · slider-independent values$/i, "$1 state · slider-independent values"],
    [/^(\d+) stavů · slider-independent values$/i, "$1 states · slider-independent values"],
    [/^([\d.,]+) bitu$/i, "$1 bits"],
    [/^([\d.,]+) bitů$/i, "$1 bits"],
    [/^([\d.,]+) tokenu$/i, "$1 tokens"],
    [/^([\d.,]+) tokenů$/i, "$1 tokens"],
    [/^Uzel #(\d+) · hloubka (\d+)$/i, "Node #$1 · depth $2"],
    [/^(Prompt \d+ · )uzel (#\d+ · .*)$/i, "$1node $2"],
    [/^(.+ · target L[^·]+ · )(\d+) vrstev$/i, "$1$2 layers"],
    [/^Vývojová simulace · 1 kontext · živý výpočet$/i, "Development simulation · 1 context · live computation"],
    [/^Vývojová simulace · (\d+) kontexty · živý výpočet$/i, "Development simulation · $1 contexts · live computation"],
    [/^Vývojová simulace · 1 kontext · z cache$/i, "Development simulation · 1 context · from cache"],
    [/^Vývojová simulace · (\d+) kontexty · z cache$/i, "Development simulation · $1 contexts · from cache"],
    [/^· 1 tokenový střep · čtecí hranice #(\S+)\. Jde o celé ortografické slovo, ne lemma\. Šedé pokračování model nepoužije\.$/i, "· 1 token fragment · readout boundary #$1. This is the complete orthographic word, not a lemma. The model does not use the grey continuation."],
    [/^· (\d+) tokenové střepy · čtecí hranice #(\S+)\. Jde o celé ortografické slovo, ne lemma\. Šedé pokračování model nepoužije\.$/i, "· $1 token fragments · readout boundary #$2. This is the complete orthographic word, not a lemma. The model does not use the grey continuation."],
    [/^· (\d+) tokenových střepů · čtecí hranice #(\S+)\. Jde o celé ortografické slovo, ne lemma\. Šedé pokračování model nepoužije\.$/i, "· $1 token fragments · readout boundary #$2. This is the complete orthographic word, not a lemma. The model does not use the grey continuation."],
    [/^První dvě lokální osy zachytí ([\d.,]+) % rozptylu zobrazeného výřezu\. Zlatě jsou další skutečné střepy slova; zeleně jeho koncový střep\.$/i, "The first two local axes capture $1% of the variance in the displayed excerpt. The other actual fragments of the word are gold; its final fragment is green."],
    [/^Hotovo — A a B jsou dvě samostatné tokenové stopy; Δ znamená B − A\.$/i, "Done — A and B are two independent token traces; Δ means B − A."],
    [/^([+−-]?[\d.,]+) bit \/ nebílý znak$/i, "$1 bits / non-whitespace character"],
    [/^token (\d+) z (\d+) · #(\S+) · hint začátku slova$/i, "token $1 of $2 · #$3 · word-beginning hint"],
    [/^token (\d+) z (\d+) · #(\S+) · pokračování tokenizovaného slova$/i, "token $1 of $2 · #$3 · continuation of a tokenized word"],
    [/^(\d+) · (\d+) úplných$/i, "$1 · $2 complete"],
    [/^(.+?) · (\d+) nejednoznačných tokenů$/i, "$1 · $2 ambiguous tokens"],
    [/^Mezerový úsek: (\d+) úseků, (\d+) úplných; pokrytí tokenů (.+)\.$/i, "Whitespace-delimited span: $1 segments, $2 complete; token coverage $3."],
    [/^Řádek: (\d+) úseků, (\d+) úplných; pokrytí tokenů (.+)\.$/i, "Line: $1 segments, $2 complete; token coverage $3."],
    [/^Větný úsek: (\d+) úseků, (\d+) úplných; pokrytí tokenů (.+)\.$/i, "Sentence-like span: $1 segments, $2 complete; token coverage $3."],
    [/^Mezerový úsek (\d+):$/i, "Whitespace-delimited span $1:"],
    [/^Řádek (\d+):$/i, "Line $1:"],
    [/^Větný úsek (\d+):$/i, "Sentence-like span $1:"],
    [/^(\d+) tokenů · znaky \[([^\]]+)\)$/i, "$1 tokens · characters [$2)"],
    [/^(\d+) tokenů · (\d+) znaků · top (\d+)$/i, "$1 tokens · $2 characters · top $3"],
    [/^(\d+) tokenů · délkově závislé, bez Δ$/i, "$1 tokens · length-dependent, without Δ"],
    [/^(\d+) forwardů: (\d+) kontrastních prefixů \+ 1 společný test…$/i, "$1 forward passes: $2 contrastive prefixes + 1 shared test…"],
    [/^Připraveno pro α (.+), 0 a (.+)\.$/i, "Ready for α $1, 0, and $2."],
    [/^Shoda top (\d+)$/i, "Top-$1 overlap"],
    [/^Shoda top (\d+) (.+)$/i, "$2 top-$1 agreement"],
    [/^pořadí #(\S+)$/i, "rank #$1"],
    [/^(.+) · jiný kontext$/i, "$1 · different context"],
    [/^Mezerový úsek · (\d+) úseků · (\d+) úplných · medián I \/ nebílý znak (.+)$/i, "Whitespace-delimited span · $1 spans · $2 complete · median I / non-whitespace character $3"],
    [/^Větný úsek · (\d+) úseků · (\d+) úplných · medián I \/ nebílý znak (.+)$/i, "Sentence span · $1 spans · $2 complete · median I / non-whitespace character $3"],
    [/^Mezerový úsek je pozorovatelný řetězec oddělený bílým místem, nikoli autoritativně určené slovo\. Tatáž jednotka platí pro obě samostatné stopy\.$/i, "A whitespace-delimited span is an observable string, not an authoritative linguistic word. The same unit is used for both independent traces."],
    [/^Řádek zachovává explicitní zalomení vloženého textu; samo zalomení mu nepřidává jazykový ani literární význam\. Tatáž jednotka platí pro obě samostatné stopy\.$/i, "A line preserves the explicit line break in the supplied text; the break itself adds no linguistic or literary meaning. The same unit is used for both independent traces."],
    [/^Větný úsek vzniká pravidlovým dělením podle interpunkce a není výsledkem syntaktické analýzy\. Tatáž jednotka platí pro obě samostatné stopy\.$/i, "A sentence-like span is produced by rule-based punctuation splitting, not syntactic analysis. The same unit is used for both independent traces."],
    [/^([+−-]?[\d.,]+) bit popisuje pouze zadaný tokenový řetězec; kořenová JS ([\d.,]+) bit současně říká, jak silně se přestavělo celé pole\.$/i, "The $1-bit information advantage describes only the specified token sequence; the root JS divergence of $2 bits shows how strongly the full field was rearranged."],
    [/^První token cíle: (.+) · #(\S+)$/i, "First target token: $1 · #$2"],
    [/^Přesná tokenizace · A (\d+) · B (\d+) · cíl (\d+) tokenů$/i, "Exact tokenization · A $1 · B $2 · target $3 tokens"],
    [/^(\d+) vrstev · slovník (\S+) · nový výpočet$/i, "$1 layers · vocabulary $2 · new computation"],
    [/^(\d+) vrstev · slovník (\S+) · výsledek z cache$/i, "$1 layers · vocabulary $2 · result from cache"],
    [/^forward (.+) · projekce (.+) · souhrn (.+)$/i, "forward $1 · projection $2 · summary $3"],
    [/^forward (.+) · projekce (.+) · z cache$/i, "forward $1 · projection $2 · from cache"],
    [/^forward (.+) · projekce (.+) · živý výpočet$/i, "forward $1 · projection $2 · live computation"],
    [/^forward (.+) · projekce (.+)$/i, "forward $1 · projection $2"],
    [/^(\d+) held-out pozic · (\d+) vrstev$/i, "$1 held-out positions · $2 layers"],
    [/^chyba po uzávěrce (.+) · před ní (.+)$/i, "post-closure error $1 · before closure $2"],
    [/^Hotovo: (\d+) intervenčních buněk, nejsilnější R (.+)\.$/i, "Done: $1 intervention cells; strongest R $2."],
    [/^(\d+) pozic × (\d+) vrstev = (\d+) intervenčních buněk$/i, "$1 positions × $2 layers = $3 intervention cells"],
    [/^(\d+) patchů · (\d+) vrstev · (\d+) pozic · (.+)$/i, "$1 patches · $2 layers · $3 positions · $4"],
    [/^L(\d+), modelová pozice (\d+): A „(.+)“ → B „(.+)“\.$/i, "L$1, model position $2: A “$3” → B “$4”."],
    [/^([\d.,]+ \/ [\d.,]+) bitu$/i, "$1 bits"],
    [/^(.+ · JS\(A,B\) [\d.,]+) bitu$/i, "$1 bits"],
    [/^Zásah (.+?) na L(\d+) změnil distribuci dalšího tokenu o JS (.+?); nejvíc posílil token (.+?) a pole se zúžilo o (.+)\.$/i, "Intervention $1 at L$2 changed the next-token distribution by a JS divergence of $3; it most increased token $4 and narrowed the field by $5."],
    [/^Zásah (.+?) na L(\d+) změnil distribuci dalšího tokenu o JS (.+?); nejvíc posílil token (.+?) a pole se rozšířilo o (.+)\.$/i, "Intervention $1 at L$2 changed the next-token distribution by a JS divergence of $3; it most increased token $4 and widened the field by $5."],
    [/^Zásah (.+?) na L(\d+) změnil distribuci dalšího tokenu o JS (.+?); nejvíc posílil token (.+?) a šířka pole zůstala prakticky stejná\.$/i, "Intervention $1 at L$2 changed the next-token distribution by a JS divergence of $3; it most increased token $4, while field width remained virtually unchanged."],
    [/^Zásah (.+?) na L(\d+) změnil distribuci dalšího tokenu o JS (.+?); pole se zúžilo o (.+)\.$/i, "Intervention $1 at L$2 changed the next-token distribution by a JS divergence of $3 and narrowed the field by $4."],
    [/^Zásah (.+?) na L(\d+) změnil distribuci dalšího tokenu o JS (.+?); pole se rozšířilo o (.+)\.$/i, "Intervention $1 at L$2 changed the next-token distribution by a JS divergence of $3 and widened the field by $4."],
    [/^Zásah (.+?) na L(\d+) změnil distribuci dalšího tokenu o JS (.+?); šířka pole zůstala prakticky stejná\.$/i, "Intervention $1 at L$2 changed the next-token distribution by a JS divergence of $3; field width remained virtually unchanged."],
    [/^Identitní kontrola na L(\d+) ponechala distribuci bez zamýšleného zásahu; JS divergence je (.+)\.$/i, "The identity control at L$1 left the distribution without an intended intervention; JS divergence is $2."],
    [/^Společný prefix: (.+)$/i, "Shared prefix: $1"],
    [/^Hotovo — (\d+) tokenů nad shodným kontextem v každém kroku za (.+)\.$/i, "Done — $1 tokens under an identical context at every step in $2."],
    [/^Hotovo — (\d+) tokenů nad shodným kontextem v každém kroku\.$/i, "Done — $1 tokens under an identical context at every step."],
    [/^Proti směru · α (.+)$/i, "Against direction · α $1"],
    [/^Po směru · α (.+)$/i, "Along direction · α $1"],
    [/^Překvapení textu · (.+)$/i, "Text surprisal · $1"],
    [/^nejednoznačných (\d+)$/i, "$1 ambiguous"],
    [/^nejednoznačné indexy (.+)$/i, "ambiguous indices $1"]
  ];

  const dynamicPatterns = [
    [/^generace (\d+)$/i, "generation $1"],
    [/^(\d+) záznam$/i, "$1 record"],
    [/^(\d+) záznamů$/i, "$1 records"],
    [/^1 uložený experiment$/i, "1 saved experiment"],
    [/^(\d+) uložené experimenty$/i, "$1 saved experiments"],
    [/^(\d+) uložených experimentů$/i, "$1 saved experiments"],
    [/^(\d+) token$/i, "$1 token"],
    [/^(\d+) tokeny$/i, "$1 tokens"],
    [/^(\d+) tokenů$/i, "$1 tokens"],
    [/^(\d+) tokenů cíle$/i, "$1 target tokens"],
    [/^(\d+) textový token$/i, "$1 text token"],
    [/^(\d+) textové tokeny$/i, "$1 text tokens"],
    [/^(\d+) textových tokenů$/i, "$1 text tokens"],
    [/^(\d+) znak$/i, "$1 character"],
    [/^(\d+) znaky$/i, "$1 characters"],
    [/^(\d+) znaků$/i, "$1 characters"],
    [/^(\d+) bajt$/i, "$1 byte"],
    [/^(\d+) bajty$/i, "$1 bytes"],
    [/^(\d+) bajtů$/i, "$1 bytes"],
    [/^(\d+) vrstva$/i, "$1 layer"],
    [/^(\d+) vrstvy$/i, "$1 layers"],
    [/^(\d+) vrstev$/i, "$1 layers"],
    [/^(\d+) uzel$/i, "$1 node"],
    [/^(\d+) uzly$/i, "$1 nodes"],
    [/^(\d+) uzlů$/i, "$1 nodes"],
    [/^(\d+) sloupec$/i, "$1 column"],
    [/^(\d+) sloupce$/i, "$1 columns"],
    [/^(\d+) sloupců$/i, "$1 columns"],
    [/^(\d+) pár$/i, "$1 pair"],
    [/^(\d+) páry$/i, "$1 pairs"],
    [/^(\d+) párů$/i, "$1 pairs"],
    [/^(\d+) krok$/i, "$1 step"],
    [/^(\d+) kroky$/i, "$1 steps"],
    [/^(\d+) kroků$/i, "$1 steps"],
    [/^(\d+) kandidát$/i, "$1 candidate"],
    [/^(\d+) kandidáti$/i, "$1 candidates"],
    [/^(\d+) kandidátů$/i, "$1 candidates"],
    [/^(\d+) střep$/i, "$1 fragment"],
    [/^(\d+) střepy$/i, "$1 fragments"],
    [/^(\d+) střepů$/i, "$1 fragments"],
    [/^(\d+) položek slovníku$/i, "$1 vocabulary entries"],
    [/^(\d+) forward$/i, "$1 forward pass"],
    [/^(\d+) forwardy$/i, "$1 forward passes"],
    [/^(\d+) forwardů$/i, "$1 forward passes"],
    [/^všech (\d+) trénovacích párů$/i, "all $1 training pairs"],
    [/^(\d+) quality-gated trénovacích párů$/i, "$1 quality-gated training pairs"],
    [/^Top (\d+) tokenů$/i, "Top $1 tokens"],
    [/^Top (\d+) · T (.+)$/i, "Top $1 · T $2"],
    [/^(\d+) z (\d+) vrstev$/i, "$1 of $2 layers"],
    [/^L(\d+) · vrstva$/i, "L$1 · layer"],
    [/^cíl L(\d+)$/i, "target L$1"],
    [/^pozice (\d+)$/i, "position $1"],
    [/^krok (\d+)$/i, "step $1"],
    [/^hloubka (\d+)$/i, "depth $1"],
    [/^pořadí #(\S+)$/i, "rank #$1"],
    [/^rank #(\S+)$/i, "rank #$1"],
    [/^(.*) je připraven$/i, "$1 is ready"],
    [/^(.*) je připravena$/i, "$1 is ready"],
    [/^Načítám (.*)…$/i, "Loading $1…"],
    [/^Přepínám na (.*)…$/i, "Switching to $1…"],
    [/^Stav: (.*)$/i, "Status: $1"],
    [/^Prefix: (.*)$/i, "Prefix: $1"],
    [/^Tokeny: (.*)$/i, "Tokens: $1"],
    [/^Vrstva: (.*)$/i, "Layer: $1"],
    [/^Čas: (.*)$/i, "Time: $1"],
    [/^Hotovo — (.*)$/i, "Done — $1"],
    [/^Chyba: (.*)$/i, "Error: $1"],
    [/^kontext$/i, "context"],
    [/^kontexty$/i, "contexts"],
    [/^trajektorie$/i, "trajectories"],
    [/^střep$/i, "fragment"],
    [/^střepy$/i, "fragments"],
    [/^výskytem$/i, "occurrence"],
    [/^výskytem ani jeho protějškem$/i, "occurrence or its counterpart"]
  ];

  const fragmentsForPage = () => {
    const maps = [
      pageCatalogue(),
      catalogue.common ?? {},
      catalogue.shell ?? {},
      crossPageCatalogue
    ];
    const seen = new Set();
    const entries = [];
    for (const map of maps) {
      for (const [source, translated] of Object.entries(map)) {
        if (
          seen.has(source) ||
          source.length < 18 ||
          source.length > 180 ||
          /[<>${}]/.test(source)
        ) continue;
        seen.add(source);
        entries.push([source, translated]);
      }
    }
    return entries.sort((left, right) => right[0].length - left[0].length);
  };

  let fragmentCache = null;
  let templateCache = null;

  const exactTranslation = (source) => {
    const direct = (
      overrides.get(source) ??
      pageCatalogue()[source] ??
      catalogue.common?.[source] ??
      catalogue.shell?.[source] ??
      crossPageCatalogue[source] ??
      fallbackCatalogue[source]
    );
    return direct ?? source;
  };

  const applyPatterns = (source, patterns) => {
    for (const [pattern, replacement] of patterns) {
      if (pattern.test(source)) return source.replace(pattern, replacement);
    }
    return source;
  };

  const patternTranslation = (source) => applyPatterns(source, dynamicPatterns);

  const escapeRegExp = (value) => value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

  const templatesForPage = () => {
    const maps = [
      Object.fromEntries(overrides),
      pageCatalogue(),
      catalogue.common ?? {},
      catalogue.shell ?? {},
      crossPageCatalogue
    ];
    const seen = new Set();
    const entries = [];
    for (const map of maps) {
      for (const [source, translated] of Object.entries(map)) {
        if (seen.has(source) || !source.includes("__PG_SLOT_")) continue;
        seen.add(source);
        const slots = [...source.matchAll(/__PG_SLOT_(\d+)__/g)];
        const targetSlots = new Set(
          [...translated.matchAll(/__PG_SLOT_(\d+)__/g)].map((match) => match[1])
        );
        if (!slots.length || slots.some((match) => !targetSlots.has(match[1]))) {
          continue;
        }
        let cursor = 0;
        let pattern = "^";
        const captureBySlot = new Map();
        slots.forEach((match, index) => {
          pattern += escapeRegExp(source.slice(cursor, match.index));
          pattern += "([\\s\\S]*?)";
          captureBySlot.set(match[1], index + 1);
          cursor = match.index + match[0].length;
        });
        pattern += `${escapeRegExp(source.slice(cursor))}$`;
        entries.push({
          source,
          translated,
          expression: new RegExp(pattern),
          captureBySlot
        });
      }
    }
    return entries.sort((left, right) => right.source.length - left.source.length);
  };

  const translateTemplate = (source) => {
    templateCache ??= templatesForPage();
    for (const entry of templateCache) {
      const match = entry.expression.exec(source);
      if (!match) continue;
      return entry.translated.replace(
        /__PG_SLOT_(\d+)__/g,
        (_placeholder, slot) => {
          const capture = match[entry.captureBySlot.get(slot)] ?? "";
          const exact = exactTranslation(capture);
          if (exact !== capture) return exact;
          const prioritized = applyPatterns(capture, priorityPatterns);
          return prioritized !== capture
            ? prioritized
            : patternTranslation(capture);
        }
      );
    }
    return source;
  };

  const translateNormalized = (source, {fragments = true} = {}) => {
    if (!source || language === "cs") return source;
    const direct = exactTranslation(source);
    if (direct !== source) return direct;
    if (!/[A-Za-zÁ-ž]/.test(source)) return source;
    const prioritized = applyPatterns(source, priorityPatterns);
    if (prioritized !== source) return prioritized;
    const templated = translateTemplate(source);
    if (templated !== source) return templated;
    const patterned = patternTranslation(source);
    if (patterned !== source) return patterned;
    if (!fragments) return source;
    fragmentCache ??= fragmentsForPage();
    let translated = source;
    let changed = false;
    for (const [fragment, replacement] of fragmentCache) {
      if (!translated.includes(fragment)) continue;
      translated = translated.replaceAll(fragment, replacement);
      changed = true;
    }
    return changed ? translated : source;
  };

  const translate = (value, options) => {
    const source = String(value ?? "");
    if (language === "cs" || !source.trim()) return source;
    const normalized = normalize(source);
    const translated = translateNormalized(normalized, options);
    if (translated === normalized) return source;
    const leading = source.match(/^\s*/)?.[0] ?? "";
    const trailing = source.match(/\s*$/)?.[0] ?? "";
    return `${leading}${translated}${trailing}`;
  };

  const attributePatterns = [
    [/Zaměřit uzel/gi, "Focus node"],
    [/Zahrada promptu/gi, "Garden for prompt"],
    [/\bUzel(?=\s*#?\d)/gi, "Node"],
    [/\bhloubka(?=\s*\d)/gi, "depth"],
    [/Entropie H/gi, "Entropy H"],
    [/Efektivní šířka 2ᴴ/gi, "Effective width 2ᴴ"],
    [/Náskok top‑1 log₂\(p₁\/p₂\)/gi, "Top‑1 log₂(p₁/p₂) lead"],
    [/Masa mimo top 10/gi, "Mass outside top 10"],
    [/Překvapení volby −log₂ p/gi, "Choice surprisal −log₂ p"],
    [/aktuální uzel/gi, "current node"],
    [/poslední vybraný stav bez dalšího přechodu/gi, "the last selected state without another transition"],
    [/Další zvolený token/gi, "Next selected token"],
    [/překvapení/gi, "surprisal"],
    [/pořadí/gi, "rank"],
    [/začátku slova/gi, "word beginning"],
    [/větně podobný úsek/gi, "sentence-like span"],
    [/mezerový úsek/gi, "whitespace-delimited span"],
    [/číselná agregace surprisal/gi, "numerical aggregation of surprisal"],
    [/Tento bod nemá další zvolený přechod\./gi, "This point has no further selected transition."],
    [/Proužek ukazuje H\/log₂\|V\| v rámci celého slovníku\./gi, "The strip shows H/log₂|V| across the full vocabulary."],
    [/Proužek je logaritmický: log₂\(efektivní šířky\)\/log₂\|V\|\./gi, "The strip is logarithmic: log₂(effective width)/log₂|V|."],
    [/Měřítko proužku je ([^.]+)\./gi, "The strip scale is $1."],
    [/0–8 bitů; vyšší hodnoty se vizuálně saturují/gi, "0–8 bits; higher values saturate visually"],
    [/0–16 bitů; vyšší hodnoty se vizuálně saturují/gi, "0–16 bits; higher values saturate visually"],
    [/efektivně/gi, "effectively"],
    [/([\d.,]+) bitu/gi, "$1 bits"],
    [/([\d.,]+) bitů/gi, "$1 bits"],
    [/([\d.,]+) tokenu/gi, "$1 tokens"],
    [/([\d.,]+) tokenů/gi, "$1 tokens"]
  ];

  const translateAttributeValue = (source) => attributePatterns.reduce(
    (value, [pattern, replacement]) => value.replace(pattern, replacement),
    translate(source, {fragments: false})
  );

  const protectedContentSelector = [
    "[data-i18n-ignore]",
    ".archive-layout .prefix",
    ".archive-layout .token",
    ".context-future",
    ".context-prefix",
    ".detail-head h2",
    ".detail-note",
    "body[data-workbench-page=\"surprise\"] #detail-title",
    "body[data-workbench-page=\"tokenization\"] #detail-title",
    ".experiment-prompt",
    ".experiment-text",
    ".experiment-prefix",
    ".field-prefix",
    ".heatmap-token",
    ".level-prefix",
    ".observed-token",
    ".path-text",
    ".prediction-token",
    ".record-button .record-preview",
    ".record-button .record-title",
    "#row-heading",
    ".token-piece",
    ".token-cell",
    ".token-symbol",
    ".trajectory-completion",
    ".trajectory-prefix",
    ".trajectory-token-text",
    "body[data-workbench-page=\"contrast\"] #results-title"
  ].join(", ");

  const ignoredTextParent = (element) => element?.closest?.(
    `script, style, code, pre, textarea, ${protectedContentSelector}`
  );
  const textStates = new WeakMap();
  const attributeStates = new WeakMap();
  let mutating = false;

  const translateTextNode = (node) => {
    if (!node?.parentElement || ignoredTextParent(node.parentElement)) return;
    const current = node.data;
    const previous = textStates.get(node);
    if (language === "en") {
      if (previous && current === previous.en) return;
      const source = previous && current === previous.cs ? previous.cs : current;
      const output = translate(source);
      textStates.set(node, {cs: source, en: output});
      if (output !== current) node.data = output;
    } else if (previous && current === previous.en) {
      node.data = previous.cs;
    }
  };

  const translateAttribute = (element, attribute) => {
    if (!element.hasAttribute(attribute)) return;
    const states = attributeStates.get(element) ?? {};
    const current = element.getAttribute(attribute) ?? "";
    const previous = states[attribute];
    if (language === "en") {
      if (previous && current === previous.en) return;
      const source = previous && current === previous.cs ? previous.cs : current;
      const output = translateAttributeValue(source);
      states[attribute] = {cs: source, en: output};
      attributeStates.set(element, states);
      if (output !== current) element.setAttribute(attribute, output);
    } else if (previous && current === previous.en) {
      element.setAttribute(attribute, previous.cs);
    }
  };

  const translatePristineControl = (element) => {
    if (!(element instanceof HTMLInputElement || element instanceof HTMLTextAreaElement)) {
      return;
    }
    if (["button", "submit", "reset"].includes(element.type)) {
      const current = element.value;
      const output = translate(current);
      if (output !== current) element.value = output;
    }
  };

  const translateElement = (element) => {
    if (
      !(element instanceof Element) ||
      element.closest(protectedContentSelector)
    ) return;
    for (const attribute of [
      "aria-label",
      "aria-description",
      "placeholder",
      "title",
      "data-empty-label"
    ]) {
      translateAttribute(element, attribute);
    }
    if (
      element instanceof HTMLMetaElement &&
      element.name === "description"
    ) {
      translateAttribute(element, "content");
    }
    translatePristineControl(element);
  };

  const translateTree = (root = document) => {
    mutating = true;
    try {
      if (root instanceof Element) translateElement(root);
      const walker = document.createTreeWalker(
        root,
        NodeFilter.SHOW_ELEMENT | NodeFilter.SHOW_TEXT
      );
      let node;
      while ((node = walker.nextNode())) {
        if (node.nodeType === Node.TEXT_NODE) translateTextNode(node);
        else translateElement(node);
      }
      document.documentElement.lang = language;
      document.documentElement.dataset.language = language;
    } finally {
      mutating = false;
    }
  };

  const observer = new MutationObserver((records) => {
    if (mutating || language !== "en") return;
    for (const record of records) {
      if (record.type === "characterData") {
        translateTextNode(record.target);
      } else if (record.type === "attributes") {
        translateAttribute(record.target, record.attributeName);
      } else {
        for (const node of record.addedNodes) {
          if (node.nodeType === Node.TEXT_NODE) translateTextNode(node);
          else if (node.nodeType === Node.ELEMENT_NODE) translateTree(node);
        }
      }
    }
  });

  const start = () => {
    translateTree(document);
    observer.observe(document.documentElement, {
      subtree: true,
      childList: true,
      characterData: true,
      attributes: true,
      attributeFilter: [
        "aria-label",
        "aria-description",
        "placeholder",
        "title",
        "data-empty-label",
        "content"
      ]
    });
  };

  const setLanguage = (nextLanguage) => {
    if (!SUPPORTED.has(nextLanguage)) return false;
    try {
      window.localStorage.setItem(STORAGE_KEY, nextLanguage);
    } catch {
      // Reload still honors a matching ?lang query when storage is blocked.
    }
    language = nextLanguage;
    document.documentElement.lang = language;
    document.documentElement.dataset.language = language;
    return true;
  };

  window.ProbabilityGardenI18n = Object.freeze({
    get language() {
      return language;
    },
    localeTag: () => language === "en" ? "en-GB" : "cs-CZ",
    setLanguage,
    t: translate,
    translateTree,
    storageKey: STORAGE_KEY
  });

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start, {once: true});
  } else {
    start();
  }
})();
