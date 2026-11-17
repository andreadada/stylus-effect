# Stylus Effect

Generatore di incisioni vettoriali da fotografie. Ripristinata la pipeline **v7.1** originale: contorni spezzati effetto matita, piccoli gruppi di tagli ravvicinati lungo le curvature, poche linee interne e output trasparente.

**Importante:** i file di esempio e le fotografie non sono inclusi nel repository. I test creano immagini sintetiche temporanee.

## Installazione

Richiede Python 3.10+.

```powershell
git clone https://github.com/andreadada/stylus-effect.git
cd stylus-effect
python -m pip install -r requirements.txt
```

Se il progetto è già presente in PyCharm, aggiorna i file come descritto nella sezione **Aggiornare il progetto esistente**.

## Comando rapido

Inserisci una foto in `input/cornicello.jpg`, poi esegui:

```powershell
python photo_to_svg.py --input input/cornicello.jpg --output output
```

Il preset predefinito riproduce lo stile inciso della v7.1.

### Il comando originale, senza modifiche

```powershell
python photo_to_svg.py --input input/cornicello.jpg --output output --preset balanced --style accent --accent-local-density 4.2 --outline-style cut --outline-cut-density 1.6 --outline-cut-length 11 --export-transparent --svg-background
```

`--svg-background` aggiunge il fondo scuro **solo allo SVG**. Omettilo per uno SVG con sfondo trasparente.

## Output

Per `cornicello.jpg` vengono generati in `output/`:

- `cornicello.svg` — disegno vettoriale modificabile, trasparente salvo `--svg-background`.
- `cornicello_art.png` — incisione su fondo antracite, **da aprire per vedere lo stile**.
- `cornicello_art_transparent.png` — PNG trasparente con i soli tratti, da sovrapporre a qualsiasi sfondo.
- `cornicello_preview.png` — confronto originale / contorni / risultato.

I pixel interamente trasparenti nel PNG hanno RGB 0, evitando il rettangolo grigio della precedente versione.

## Esempi di comandi PowerShell

```powershell
# Tutte le foto contenute in input/
python photo_to_svg.py

# Gruppi più fitti, senza aggiungere nuove zone
python photo_to_svg.py --input input/cornicello.jpg --accent-local-density 5.5

# Bordo con più tagli irregolari
python photo_to_svg.py --input input/cornicello.jpg --outline-cut-density 2

# Più gruppi di incisione
python photo_to_svg.py --input input/cornicello.jpg --accent-clusters 8

# Meno linee interne
python photo_to_svg.py --input input/cornicello.jpg --accent-inner-lines 1

# Riconoscimento dello sfondo bianco
python photo_to_svg.py --input input/cornicello.jpg --auto-mask white

# Isola un soggetto centrale su uno sfondo non uniforme
python photo_to_svg.py --input input/cornicello.jpg --grabcut

# Maschera personalizzata: soggetto bianco, sfondo nero
python photo_to_svg.py --input input/cornicello.jpg --mask input/maschera.png

# Anteprima e mappe per controllare la segmentazione
python photo_to_svg.py --input input/cornicello.jpg --debug

# Solo il tratto SVG e il PNG trasparente, senza foglio di confronto
python photo_to_svg.py --input input/cornicello.jpg --no-preview
```

Le fotografie con sfondo complesso possono richiedere `--grabcut` o una maschera manuale. La versione ripristinata non implementa il modello opzionale `--segment rembg` della riscrittura, che aveva modificato in modo indesiderato la resa artistica.

## Interfaccia grafica

```powershell
python photo_to_svg_gui.py
```

La GUI è costruita dagli argomenti di `photo_to_svg.py`, quindi permette di regolare parametri, slider e percorsi senza modificare il codice.

## Aggiornare il progetto esistente in PyCharm (PowerShell)

Esegui questi comandi nella cartella del progetto esistente. Le cartelle `input/`, `output/` e `.venv/` non vengono toccate.

```powershell
Copy-Item photo_to_svg.py photo_to_svg_backup.py -Force
Copy-Item photo_to_svg_gui.py photo_to_svg_gui_backup.py -Force
$base = "https://raw.githubusercontent.com/andreadada/stylus-effect/main"
Invoke-WebRequest "$base/photo_to_svg.py" -OutFile photo_to_svg.py
Invoke-WebRequest "$base/photo_to_svg_gui.py" -OutFile photo_to_svg_gui.py
Invoke-WebRequest "$base/requirements.txt" -OutFile requirements.txt
python -m pip install -r requirements.txt
```

## Test

```powershell
python -m unittest discover -s tests -v
```

Non vengono pubblicate immagini o file generati. `input/` e `output/` sono ignorati da Git.

---
La pipeline utilizza OpenCV + NumPy in locale, senza API o connessioni obbligatorie.
