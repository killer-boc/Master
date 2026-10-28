# OCR text detection na CanMV K230

MicroPython projekt pre dosku **CanMV K230 (Kendryte K230)** s NPU/KPU: detekcia
textu v obraze z kamery v reálnom čase a voliteľne aj rozpoznanie textu (OCR).

Vychádza z oficiálnych CanMV príkladov (`resources/examples/05-AI-Demo/ocr_det.py`,
`ocr_rec.py`) a z firmware knižníc `libs.AIBase`, `libs.AI2D`, `libs.PipeLine` a
modulu `aicube`. Nepotrebuje žiadne externé Python balíčky — všetko je súčasťou
CanMV firmware.

## Čo to robí

```
sensor (RGB888 planar) -> AI2D (letterbox pad + bilinear resize)
        -> PP-OCR DB detection kmodel (KPU)
        -> aicube.ocr_post_process  (mask threshold, box threshold, NMS)
        -> filtre + zoradenie boxov (zhora dole, zľava doprava)
        -> kreslenie polygónov na OSD vrstvu + log do konzoly
```

`ocr_text_read.py` navyše každý detegovaný výrez pošle do recognition modelu
(CTC) a vypíše/nakreslí rozpoznaný reťazec.

## Požiadavky

* CanMV K230 / CanMV K230D s firmware, ktorý obsahuje:
  * `libs/AIBase.py`, `libs/AI2D.py`, `libs/PipeLine.py`, `libs/Utils.py`
    (v firmware na SD karte v `/sdcard/app/libs` alebo `/sdcard/examples/libs`),
  * moduly `nncase_runtime` a `aicube` (sú súčasťou firmware image),
* kameru a displej kompatibilný s `PipeLine` (`display_mode="auto"` vyberie
  predvolený ovládač dosky, napr. `hdmi`, `lcd`, `st7701`, ...).

## Súbory na SD karte

Modely `ocr_det_int16.kmodel`, `ocr_rec_int16.kmodel` a `dict.txt` dodáva CanMV
firmware v príkladoch. Skopírujte skripty vedľa nich (alebo upravte cesty v
konfigurácii na začiatku súborov):

```
/sdcard/examples/
├── ocr_text_detect.py          # tento repozitár
├── ocr_text_read.py            # tento repozitár (importuje ocr_text_detect)
├── kmodel/
│   ├── ocr_det_int16.kmodel    # z firmware / export z Canaan platformy
│   └── ocr_rec_int16.kmodel    # len pre ocr_text_read.py
└── utils/
    └── dict.txt                # slovník znakov pre recognition model
```

Ak vaša verzia firmware má modely v `/sdcard/examples/ai_test_kmodel/`, zmeňte
`DET_KMODEL_PATH` resp. `REC_KMODEL_PATH` v hlavičke skriptu.

## Spustenie

1. Skopírujte `ocr_text_detect.py` na SD kartu (alebo priamo do CanMV IDE).
2. V CanMV IDE otvorte súbor a stlačte **Run**, alebo v REPL:

```python
import ocr_text_detect
ocr_text_detect.main()
```

3. Detegované textové oblasti sa kreslia žltými polygónmi, do konzoly sa každých
   `PRINT_INTERVAL` snímok vypíše počet boxov a ich súradnice:

```
[ocr_text_detect] frame=30 fps=18.7 boxes=3
    quad: [400, 120, 560, 120, 560, 140, 400, 140]
    quad: [40, 200, 240, 200, 240, 220, 40, 220]
```

4. Zastavenie: **Stop** v IDE alebo `Ctrl-C`. `finally` blok vždy uvoľní KPU,
   senzor a displej (`deinit()` + `PipeLine.destroy()`), takže dalšie spustenie
   funguje bez resetu dosky.

Pre rozpoznanie textu (detekcia + čítanie):

```python
import ocr_text_read
ocr_text_read.main()
```

## Konfigurácia

`ocr_text_detect.py` (hlavička súboru):

| Konštanta | Význam |
| --- | --- |
| `DET_KMODEL_PATH` | cesta k detekčnému kmodelu |
| `DET_INPUT_SIZE` | vstup modelu `[w, h]`, musí sedieť s kmodelom (predvolene `[640, 640]`) |
| `RGB888P_SIZE` | rozlíšenie posielané do AI `[w, h]`; menšie = rýchlejšie |
| `DISPLAY_MODE`, `DISPLAY_SIZE` | `"auto"` / `"hdmi"` / `"lcd"` / ... a voliteľné rozlíšenie |
| `MASK_THRESHOLD` | prah pixelovej pravdepodobnosti "text" (DB maska) |
| `BOX_THRESHOLD` | prah priemernej pravdepodobnosti boxu |
| `MIN_BOX_AREA` | minimálna plocha boxu v pixeloch AI snímky (filter šumu) |
| `MAX_BOXES` | max. počet boxov na snímku (0 = bez limitu) |
| `BOX_COLOR`, `BOX_THICKNESS` | farba (ARGB) a hrúbka obrysov |
| `PRINT_INTERVAL` | každá N-tá snímka sa loguje (0 = nikdy) |

`ocr_text_read.py` pridáva `REC_KMODEL_PATH`, `REC_INPUT_SIZE` (`[512, 32]`),
`DICT_PATH`, `BLANK_INDEX`, `OUTPUT_FILE` (ukladanie textu na SD), `FONT_PATH`
(TTF pre čínsky text, napr. `/sdcard/res/font/AlibabaPuHuiTi-3-45-Light.ttf`).

### Ladenie presnosti

* **Nič sa nedeteguje** – znížte `MASK_THRESHOLD` (napr. 0.2) a `BOX_THRESHOLD`
  (napr. 0.2), zväčšite `RGB888P_SIZE` (napr. `[1280, 720]`), odstráňte
  `MIN_BOX_AREA` limit.
* **Príliš veľa falošných boxov** – zvýšte oba prahy a `MIN_BOX_AREA`.
* **Detekcia je pomalá** – zmenšite `RGB888P_SIZE` (napr. `[480, 270]`) alebo
  znížte `MAX_BOXES`; detekcia beží na 640×640 modeli, čo je najdrahší krok.
* **Text je posunutý o znak / chýbajú znaky** – v `ocr_text_read.py` nastavte
  `BLANK_INDEX = 0` (niektoré exporty majú CTC blank na indexe 0 namiesto
  posledného).

## Offline testy (na PC, bez dosky)

Skripty sa dajú overiť bez hardvéru — `tests/emulate_board.py` nahradí
firmware moduly (`libs.*`, `nncase_runtime`, `aicube`, `media.*`, `ulab.numpy`)
ľahkými stubmi a spustí reálny kód detekcie aj rozpoznávania:

```bash
python3 -m pip install -r tests/requirements.txt
python3 tests/emulate_board.py
```

Overuje sa: import a beh oboch skriptov, tvary AI2D vstupov/výstupov, filtrovanie
príliš malých boxov, orezanie súradníc mimo snímky, poradie čítania boxov,
CTC dekódovanie (spájanie opakovaní + mazanie blank tokenov) a uvoľnenie
prostriedkov (`deinit` + `PipeLine.destroy`).

Očakávaný výstup:

```
ok  geometry helpers
ok  detector
ok  recognizer
ok  ocr_text_detect.main()
ok  ocr_text_read.main()
all offline checks passed
```

## Štruktúra repozitára

```
ocr_text_detect.py     # detekcia textu z kamery (hlavný skript)
ocr_text_read.py       # detekcia + rozpoznanie textu
README.md
tests/emulate_board.py # offline smoke test so stubmi firmware modulov
tests/requirements.txt # numpy (len pre offline test)
```

## Známe obmedzenia

* Detekčný model je PP-OCR DB (`ocr_det_int16.kmodel`) z CanMV firmware; iný
  kmodel vyžaduje zhodný vstup (`DET_INPUT_SIZE`) a rovnaký typ výstupu
  (pravdepodobnostná maska pre `aicube.ocr_post_process`).
* `aicube.ocr_post_process` vracia výrezy v rozlíšení modelu; recognition model
  preto očakáva `REC_INPUT_SIZE = [512, 32]` (šírka × výška).
* OSD kreslenie orezáva súradnice do rozlíšenia displeja až pri vykreslení;
  boxy sú vždy najprv oreazané do rozlíšenia AI snímky.
* Zdrojová inšpirácia a API: oficiálne príklady CanMV K230
  (`resources/examples/05-AI-Demo/ocr_det.py`, `ocr_rec.py`), licencia CanMV SDK.
