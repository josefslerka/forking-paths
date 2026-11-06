[English](README.md) · [Čeština](README.cs.md)

# Rozcestí / Forking Paths

**Výzkumná a výuková laboratoř jazykových modelů.**

Co model očekává za daným textem? Jak jeho očekávání změní čárka, jiný pád
nebo odlišný kontext? Rozcestí umožňuje tyto otázky zkoumat na konkrétních
textech a vidět pravděpodobnosti, nikoli jen hotovou odpověď chatbota.
Je určené také humanitním badatelům a psycholingvistům, ne pouze vývojářům.

Aplikace se česky jmenuje **Rozcestí**, anglicky **Forking Paths**. Apertus
a EuroLLM jsou podporované modely, nikoli název aplikace. Pohled s větvením
možných pokračování se dál jmenuje **Pravděpodobnostní zahrada**.

**Raná alfa, která běží lokálně.** Není třeba účet ani cloudové inferenční API.
Váhy modelů, soukromé texty, korpusy, naskenované dokumenty ani naučené
výzkumné doplňky nejsou součástí repozitáře. Při startu se žádný model
automaticky nestahuje.

## Co lze zkoumat

- **Tokenizace:** na jaké vstupní jednotky vybraný model text rozdělí.
- **Pravděpodobnostní zahrada:** jaká pokračování model nabízí a jak se při
  procházení jednotlivých cest mění rozdělení pravděpodobností.
- **Překvapení textu:** pravděpodobnost, surprisal a entropie v hotovém textu;
  také čtení po slovech, rozkliknutí tokenů a export tabulky.
- **Kontrast kontextů:** jak dva různé začátky mění pravděpodobnost stejného
  zadaného pokračování i celé rozdělení dalších tokenů.
- **Porovnání modelů:** dvě modelová měření stejného textu nebo experimentu.
- **Reprezentace a vrstvy:** embeddingy, kontextové stavy, RAW čtečka,
  rozpis preference, kauzální přenos a steering na podporovaných MLX
  architekturách. Podpora není automatická pro každý registrovaný model.
- **Archiv měření:** lokální ukládání experimentů s nastavením a původem dat.

Aplikace není chatovací služba, model lidských čtecích časů ani důkaz, že model
rozumí textu stejně jako člověk. Výzkumný detektor také není důkazem autorství
nebo původu konkrétního textu. Viz [metody a omezení](docs/METHODS.md).

## 1. Zkuste rozhraní bez modelu

Potřebujete Git a Python **3.11 nebo 3.12**. V terminálu spusťte:

```sh
git clone https://github.com/josefslerka/forking-paths.git
cd forking-paths
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
forking-paths demo
```

Ve Windows aktivujte prostředí příkazem `.venv\Scripts\Activate.ps1`
v PowerShellu namísto `source .venv/bin/activate`.

Otevřete [http://127.0.0.1:8765](http://127.0.0.1:8765). Vpravo nahoře
můžete přepnout češtinu a angličtinu. Server zastavíte pomocí Ctrl+C.
Pokud je port obsazený, spusťte `forking-paths demo --port 8766`.

**Čísla v demu jsou syntetická, ne skutečná měření Apertusu nebo EuroLLM.**
Demo slouží k ověření instalace a seznámení s ovládáním, ne k jazykovým závěrům.
Správa modelů je v tomto režimu jen ke čtení.

## 2. Připojte skutečný model

Pro jeden server zvolte jeden výpočetní backend. Také modely v párovém
porovnání musí používat stejný backend. Oba doplňky instalovat nemusíte.
Při prvním spuštění bez vlastního registru aplikace sama volí **MLX na macOS
s čipem Apple**, jinde **PyTorch**. Existující nastavení nepřepisuje potichu;
u nevhodného MLX nastavení nabídne **Přejít na PyTorch**, se zálohou registru
a zachováním původních HF cest i všech souborů vah.

### Mac s čipem Apple: MLX

Do stejného virtuálního prostředí doinstalujte výpočetní část:

```sh
python -m pip install -e '.[mlx]'
forking-paths serve
```

Bez připravených vah aplikace sama otevře **Správu modelů**. Jinak ji najdete
v nabídce nebo na [http://127.0.0.1:8765/sprava-modelu](http://127.0.0.1:8765/sprava-modelu).

1. **Váhy už máte:** vyberte složku modelu, nechte ji zkontrolovat a potvrďte
   připojení. V Electronu lze složku vybrat nativním dialogem. Váhy se nekopírují.
2. **Váhy potřebujete:** vyberte připravený profil, prohlédněte si zdroje,
   přibližnou velikost a kroky instalace. Stahování začne až po vašem souhlasu.
   Pro MLX jsou připravené Apertus BF16/8bit/4bit/Hybrid a EuroLLM 9B 6bit.
3. **Ověřit výpočtem** zkusí načtení a jednu distribuci. Není to úplná validace
   modelu ani všech metod. **Použít** načte model hned; **Nastavit jako výchozí**
   jej vybere při příštím spuštění.

Skrytí a odpojení upravují pouze registr: **váhy na disku nemažou**.
Před úpravou nastavení vzniká záloha. Správa sama neinstaluje Python balíčky
ani libovolný kód.

Samotné BF16 váhy Apertusu mají přibližně 16 GB; výpočet vyžaduje další paměť
a instalace další místo na disku. Hybrid potřebuje osmibitové tělo i původní
BF16 váhy. Kvantizované profily se připravují z původních zdrojů; nejde o malý
doplněk prohlížeče. Viz [úplný instalační návod](docs/INSTALL.cs.md)
a [připravené profily](docs/MODELS.md#included-presets).

### Linux, Windows nebo jiný počítač: PyTorch

```sh
python -m pip install -e '.[torch]'
forking-paths serve --device auto
```

Ve Správě modelů připojte původní HF váhy ve formátu safetensors pro kauzální
jazykový model podporovaný knihovnou Transformers. Připravené profily pro
tento backend jsou Apertus BF16 a původní EuroLLM 9B HF. Soubory GGUF z Ollamy
ani kvantizované MLX checkpointy zde nejsou zaměnitelné s HF safetensors.

`auto` použije CUDA, pokud je dostupná, jinak CPU. Na CPU se používá FP32;
velký model může být pomalý a náročný na paměť. Tento backend nyní poskytuje
základní pravděpodobnostní nástroje, nikoli nástroje pro vnitřní vrstvy.
Konkrétní instalace PyTorch pro GPU závisí na vašem hardwaru.
4bit, 8bit a Hybrid v nabídce jsou profily našeho MLX režimu na Apple silicon,
ne přenositelné PyTorch kvantizace. Linuxové MLX/CUDA v této verzi nezpřístupňujeme.

Chybějící knihovny a chyby instalace se ukazují přímo v instalačním dialogu.
Použijte uvedený příkaz se stejným Pythonem, ve kterém běží server, a potom
server restartujte. Už stažené původní BF16 váhy lze po přepnutí na PyTorch
použít znovu. Viz [řešení staršího MLX nastavení na Linuxu](docs/INSTALL.cs.md#linux-a-hlášení-o-mlx).

## Další modely a experimentální doplňky

Nový model přidáte přes **Správa modelů → Připojit lokální model** nebo
tabulkou v lokálním `models.toml`. Server není nutné měnit kvůli každému novému
názvu modelu. Známé architektury mají ověřené adaptéry; neznámá architektura
začíná základním adaptérem `causal-lm`. Nové pohledy do vnitřních stavů mohou
vyžadovat implementaci adaptéru a numerické testy. Registr sám takovou podporu
nevytvoří. Viz [návod k registru a adaptérům](docs/MODELS.md).

V části **Experimentální doplňky** můžete připojit kompatibilní balíček TUNED,
J-LENS nebo detektoru: **připojit složku → ověřit balíček → zapnout**.
Každý balíček je vázaný na konkrétní model a přesnost. Současný J-LENS je
pouze pro BF16 Apertus, původní detektor pro odpovídající Hybrid. Tyto soukromé
výzkumné balíčky se s aplikací nešíří. Detektor navíc vyžaduje své zamčené
pomocné soubory; nestačí přenést samotný manifest.

## Prohlížeč nebo Electron

Pro všechny výzkumné nástroje stačí prohlížeč. Volitelný Electron přidává
samostatné okno, běžné kopírování a vkládání, výběr složek a správu lokálního
serveru. Nativní aplikace se jmenuje **Forking Paths**. Python ani váhy nejsou
součástí okna; podepsaný a notarizovaný veřejný instalátor zatím neposkytujeme.
Viz [desktopový návod](docs/DESKTOP.md).

## Vaše data a bezpečnost

Archiv, texty a připojení modelů patří do vašich lokálních dat, ne do Gitu.
Jejich umístění vypíše:

```sh
forking-paths paths
```

Lze je změnit pomocí `APERTUS_DATA_DIR`, `APERTUS_CACHE_DIR`,
`APERTUS_MODEL_REGISTRY` nebo přepínačů `--registry` a `--data-dir`.
Pro kompatibilitu zůstává původní Python namespace, název distribuce,
technické názvy adresářů a proměnných `APERTUS_*`. Příkazy `forking-paths`
a `apertus-workbench` spouštějí stejnou aplikaci. Přejmenování neodpojuje
modely ani nepřesouvá archiv.

**Veřejný je zdrojový kód, nikoli běžící server.** Aplikace je určená jednomu
uživateli na místním počítači, naslouchá pouze na loopback adrese a nemá
přihlašování ani oddělení uživatelů. Nevystavujte ji na internet reverzní proxy,
tunelem nebo jako sdílenou službu. Exporty, logy a archivy před sdílením
zkontrolujte: mohou obsahovat vaše texty a lokální cesty.
Viz [bezpečnost a soukromí](SECURITY.md).

## Vývoj, testy a licence

```sh
python -m pip install -e '.[dev]'
python -m pytest -q
```

Základní testy nestahují váhy. Prohlížečové testy používají syntetická data;
vybrané numerické testy a soukromé kalibrační kontroly jsou samostatné.
Ověření konkrétních prostředí a přeskočené kontroly jsou popsané v
[záznamu ověření](docs/VALIDATION.md). Postup pro přispěvatele včetně
prohlížečových testů je v [CONTRIBUTING.md](CONTRIBUTING.md).

Vlastní kód a původní rozhraní jsou pod licencí **Apache 2.0**:
[LICENSE](LICENSE), [NOTICE](NOTICE). Váhy modelů a závislosti mají vlastní
licence a oznámení: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
Projekt není oficiálním produktem poskytovatelů modelů.

Při publikaci výsledků uveďte checkpoint a revizi modelu, backend, přesnost
či kvantizaci, verzi aplikace, přesné podněty a způsob skórování.
Základní údaje pro citaci najdete v [CITATION.cff](CITATION.cff).
