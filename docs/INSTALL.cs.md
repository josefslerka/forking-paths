# Rozcestí — instalace česky

Rozcestí (anglicky Forking Paths) běží na vašem počítači. Prohlížeč zobrazuje rozhraní, Python počítá
výsledky. Není třeba účet ani placené API. Velký jazykový model ale zabírá
několik až desítky GB; není součástí tohoto repozitáře.

Původní příkaz `apertus-workbench` zůstává funkční; nově lze místo něj použít
`forking-paths`. Technické názvy adresářů dat a proměnných `APERTUS_*` se kvůli
kompatibilitě nemění. Přejmenování aplikace neodpojí modely, doplňky ani archiv.

## Nejdřív zkuste rozhraní

Nainstalujte Python 3.11 nebo 3.12. Otevřete terminál v adresáři repozitáře a
postupně spusťte:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
apertus-workbench demo
```

Ve Windows použijte pro aktivaci `.venv\Scripts\Activate.ps1` v PowerShellu.
Potom otevřete [http://127.0.0.1:8765](http://127.0.0.1:8765).
Vpravo nahoře přepínáte češtinu a angličtinu. Server zastavíte Ctrl+C.

**Demo je simulace.** Slouží k prohlédnutí aplikace bez stahování vah.
Čísla z tohoto režimu nelze použít jako výsledky jazykového modelu.

## Skutečný model na Macu s čipem Apple

Do stejného prostředí doinstalujte výpočetní část. Při prvním spuštění se
správné výchozí prostředí vybere podle systému; vlastní registr má přednost:

```sh
python -m pip install -e '.[mlx]'
apertus-workbench serve
```

Otevřete **Správa modelů** v nabídce nebo adresu
[http://127.0.0.1:8765/sprava-modelu](http://127.0.0.1:8765/sprava-modelu).
Bez připravených vah se tato stránka otevře sama. Běžné přidání už nevyžaduje
ruční úpravu konfiguračního souboru.

1. **Váhy už máte:** zvolte Připojit lokální model, vložte celou cestu ke
   složce a klikněte na Zkontrolovat složku. V Electronu je i tlačítko pro
   výběr složky. Zvolte název a potvrďte připojení. Soubory se nekopírují.
2. **Váhy nemáte:** vyberte připravený profil. Nejdřív si prohlédnete odhad
   místa a potřebné zdroje; teprve zaškrtnutí souhlasu a Spustit instalaci
   zahájí stahování. Nabídka zahrnuje Apertus BF16/8bit/4bit/Hybrid a
   EuroLLM 9B 6bit. Hybrid i kvantizace potřebují původní zdrojové váhy.
3. **Ověřit výpočtem** zkusí načtení a jednu distribuci pravděpodobností.
   Nejde o úplnou validaci modelu nebo všech nástrojů.
4. **Použít** načte model hned. **Nastavit jako výchozí** ho vybere pro další
   start. Skrytí a odpojení upravují pouze seznam; váhy na disku nemažou.

Před změnou nastavení vzniká záloha. Běžné změny se projeví bez restartu.
V demu je správa jen ke čtení; pro připojení skutečných modelů použijte
`serve`, nikoli `demo`. Pro otevření správy bez načtení existujících vah lze
přidat `serve --setup-only`.

Terminálové příkazy jsou dál dostupné, například:

```sh
apertus-workbench models download bf16
apertus-workbench models check bf16
apertus-workbench serve
```

### Experimentální doplňky

Ve Správě modelů je samostatná část **Experimentální doplňky**. U TUNED,
J-LENS nebo detektoru vyberte složku původního balíčku, zkontrolujte ji,
potvrďte připojení, pak použijte **Ověřit balíček** a **Zapnout**.
Připojení nic nestahuje ani nekopíruje. Odpojení ponechá soubory na disku.

Každá karta ukazuje kompatibilní modely. Současný J-LENS je pouze pro BF16
Apertus; původní detektor pro odpovídající Hybrid. TUNED se váže ke konkrétnímu
režimu, pro který byl naučen. Modely EuroLLM nebo jiné přesnosti tím automaticky
nezískají tyto nástroje. Před prvním výpočtem mohou proběhnout další kontroly
obsahu vah či přenosová sonda.

Nastavení je v `experimental-addons.json` v uživatelských datech, včetně
oddělených záloh. Není v Gitu. Starší detektor navíc vyžaduje zamčené pomocné
soubory, na které odkazuje jeho manifest. Když chybí nebo se změnily, nelze ho
zapnout pouhým přepínačem. Ověření balíčku není důkazem kvality metody ani
spolehlivou identifikací autorství konkrétního textu.

### Velikost modelu a úspornější varianty

BF16 váhy mají přibližně 16 GB. Počítejte s další pamětí pro samotný výpočet a
s místem na disku pro stahování. Osmibitovou variantu připravíte zvlášť:

```sh
apertus-workbench models convert 8bit
apertus-workbench serve --initial-model 8bit
```

Hybrid používá osmibitové vrstvy, ale původní BF16 vstup a výstup. Potřebuje
oba adresáře, nikoli jen osmibitové váhy:

```sh
apertus-workbench models check hybrid
apertus-workbench serve --initial-model hybrid
```

Nic se nestahuje automaticky při startu. Existující adresář modelu instalátor
nepřepíše. Volba `models check` kontroluje soubory a nastavení, ne jazykovou
kvalitu modelu ani numerickou shodu s jinou přesností.

## Jiný počítač nebo PyTorch

```sh
python -m pip install -e '.[torch]'
forking-paths serve --device auto
```

Při prvním spuštění na Linuxu, Windows a Intel Macu se bez vlastního registru
vybere PyTorch a nabídnou se pouze původní HF profily. Připojte váhy ve formátu
safetensors přes Správu modelů. Musí jít o kauzální jazykový model podporovaný
knihovnou Transformers. Připravené stahované profily zde jsou Apertus BF16 a
původní EuroLLM 9B HF; kvantizované MLX varianty pro tento backend nejsou.
Případně nastavte cestu ručně v `models.toml` a ověřte ji příkazem:

```sh
apertus-workbench models check
apertus-workbench serve
```

Tento režim umí tokenizaci, pravděpodobnosti, surprisal, kontrasty kontextů a
porovnání modelů. Pohledy dovnitř vrstev zatím vyžadují podporovanou
architekturu v MLX. Soubor z Ollamy/GGUF ani MLX kvantizaci nelze jednoduše
použít jako PyTorch checkpoint. Na CPU mohou velké modely běžet velmi pomalu.

### Linux a hlášení o MLX

Starší výchozí konfigurace mohla na Ubuntu nabídnout MLX a 4bit instalaci.
Jde o chybu průvodce, ne o potřebu pořídit si Mac. Základní pravděpodobnostní
nástroje na Linuxu používají PyTorch; experimentální vnitřní nástroje a naše
MLX kvantizace mají nadále vlastní platformní omezení.

1. V adresáři projektu aktualizujte zdroje pomocí `git pull --ff-only`.
2. Aktivujte prostředí, ze kterého aplikaci spouštíte, a nainstalujte
   `python -m pip install -e '.[torch]'`.
3. Ukončete dosavadní server pomocí Ctrl+C a spusťte
   `forking-paths serve --setup-only`.
4. Pokud Správa modelů ukazuje staré MLX nastavení, zvolte **Přejít na
   PyTorch**. Přepnutí zálohuje původní registr a odpojí jen nepřenositelné
   MLX profily. Váhy na disku nemaže ani nestahuje.
5. U již staženého původního **Apertus 8B · BF16** použijte **Použít**.
   Jeho původní cesta zůstává zachovaná, takže není potřeba znovu stahovat váhy.

Instalační dialog vždy ukazuje chybějící knihovny dřív, než spustí stahování.
Příkaz obsahuje přesný Python aktuálního serveru: instalace přes jiné `pip`
může skončit úspěšně, ale aplikaci nepomůže. Po instalaci knihoven server
restartujte; obnova stránky není restartem Pythonu.

Volba `--device auto` použije dostupné CUDA, jinak CPU. Nezaručuje, že každý
model půjde načíst do paměti daného počítače. Při nedostatku paměti zvolte
menší kompatibilní HF model. Naše 4bit/8bit/Hybrid recepty nejsou PyTorch
kvantizace. MLX/CUDA na Linuxu tato verze aplikace zatím neověřuje ani nenabízí.

## Přidání modelu a běžné potíže

Nový model přidáte formulářem nebo tabulkou `[models.nazev]` se jménem,
cestou a adaptérem. Nemusíte měnit server. Podrobnosti a příklady jsou v
[návodu k registru](MODELS.md).

Při ruční změně registru nabídne správa tlačítko **Použít změny ze souboru**.
Změna backendu, cesty nebo adaptéru právě načteného modelu vyžaduje restart.
Backend se volí pro celý server, nelze libovolně míchat MLX a PyTorch v jednom
běhu. Výpočetní knihovny se instalují přes pip; správa je sama nestahuje.

- „Port je obsazený“: použijte `serve --port 8766` a otevřete tuto adresu.
- „Model není dostupný“: spusťte `models check JEHO_ID`; zkontrolujte cestu,
  všechny části vah a správný backend.
- „Chybí modul“: aktivujte `.venv` a doinstalujte zvolený doplněk `.[mlx]`
  nebo `.[torch]`. Balíčky instalujte přes `python -m pip` stejného prostředí.
- Málo paměti: zavřete jiné modely/aplikace nebo zvolte menší model. Aplikace
  sama nezvyšuje systémové paměťové limity a nespouští `sudo`.
- Šedý nástroj: aktivní architektura ho nepodporuje nebo chybí samostatně
  kalibrovaný artefakt. Přidání názvu do registru není jeho validací.

Umístění vlastních dat vypíše `apertus-workbench paths`. Zdrojový repozitář
neobsahuje vaše texty, archiv měření ani váhy. Při dalším otevření terminálu
stačí znovu aktivovat `.venv` a spustit `apertus-workbench serve`.

Elektronové okno je volitelné; pro samotné používání stačí prohlížeč.
Viz [desktopový návod](DESKTOP.md). Server nesdílejte veřejně — tato verze
nemá přihlašování ani oddělení uživatelských dat.
