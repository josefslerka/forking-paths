# Instalace česky

Workbench běží na vašem počítači. Prohlížeč zobrazuje rozhraní, Python počítá
výsledky. Není třeba účet ani placené API. Velký jazykový model ale zabírá
několik až desítky GB; není součástí tohoto repozitáře.

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

Do stejného prostředí doinstalujte výpočetní část a zkopírujte vzor nastavení:

```sh
python -m pip install -e '.[mlx]'
cp models.example.toml models.toml
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
cp models.torch.example.toml models.toml
```

Spusťte `apertus-workbench serve` a připojte původní HF váhy ve formátu
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
