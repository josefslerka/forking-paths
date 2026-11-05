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
apertus-workbench models list
```

Soubor `models.toml` je váš seznam modelů. Máte-li už stažené kompatibilní
váhy, doplňte k modelu řádek `path = "/cesta/k/modelu"`. Nic se nekopíruje.
Pokud váhy nemáte, tyto příkazy stáhnou konkrétní oficiální verzi Apertusu a
spustí aplikaci:

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

Do `models.toml` doplňte cestu k původním HF vahám ve formátu safetensors.
Musí jít o kauzální jazykový model podporovaný knihovnou Transformers.
Pak spusťte:

```sh
apertus-workbench models check
apertus-workbench serve
```

Tento režim umí tokenizaci, pravděpodobnosti, surprisal, kontrasty kontextů a
porovnání modelů. Pohledy dovnitř vrstev zatím vyžadují podporovanou
architekturu v MLX. Soubor z Ollamy/GGUF ani MLX kvantizaci nelze jednoduše
použít jako PyTorch checkpoint. Na CPU mohou velké modely běžet velmi pomalu.

## Přidání modelu a běžné potíže

Nový model přidáte tabulkou `[models.nazev]` se jménem, cestou a adaptérem.
Nemusíte měnit server. Podrobnosti a příklady jsou v
[návodu k registru](MODELS.md).

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
