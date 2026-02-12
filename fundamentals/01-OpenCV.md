# Základy OpenCV - Shrnutí

## 0. Příprava prostředí

### Vytvoření virtuálního prostředí (Virtual Environment)
Pro izolaci projektu a závislostí je doporučeno vytvořit virtuální prostředí.
```bash
python -m venv .venv
```

### Aktivace prostředí (Linux/macOS)
```bash
source .venv/bin/activate
# Deaktivace se provádí příkazem: deactivate
```

### Instalace knihoven
Pro začátek potřebujeme `opencv-python` (pro prací s obrazem) a `matplotlib` (pro grafy).
```bash
pip install opencv-python matplotlib
```

### Nastavení VS Code (Python Interpret)
Aby VS Code používal knihovny z virtuálního prostředí (a nehlásil "Module not found"):
1. Otevřete paletu příkazů: **Ctrl + Shift + P** (nebo F1).
2. Napište a vyberte: `Python: Select Interpreter`.
3. V seznamu vyberte položku, která obsahuje cestu k vašemu prostředí (např. `./.venv/bin/python` nebo `.venv (v environment)`).
   - Pokud se prostředí nezobrazí, zvolte "Enter interpreter path" a naved'te jej na soubor `python` ve složce `.venv/bin`.

---

Tento dokument shrnuje základní operace v knihovně OpenCV pro Python (`cv2`).

## 1. Reprezentace obrazu
Obraz je v OpenCV reprezentován jako **NumPy matice** (`numpy.ndarray`).
- **Rozměry:** `(výška, šířka, kanály)`
- **Datový typ:** `uint8` (hodnoty 0-255)
- **Barevný model:** **BGR** (Blue-Green-Red), nikoliv RGB.

### Vytvoření prázdného obrazu
```python
import numpy as np
import cv2 as cv

# Černý obraz 500x300 pixelů, 3 kanály
img = np.zeros((500, 300, 3), np.uint8)
```

## 2. Přístup k datům a modifikace
K pixelům se přistupuje pomocí indexování `img[y, x]`.

### Modifikace pixelů
```python
# Změna barvy jednoho pixelu na bílou
img[50, 50] = (255, 255, 255)
```

### Slicing (Řezy) a ROI (Region of Interest)
Pro efektivní práci s oblastmi se používá slicing NumPy.
```python
# Obarvení řádku 50 na červenou
img[50, :] = (0, 0, 255)

# Výběr a kopírování oblasti (ROI)
# img[y1:y2, x1:x2]
roi = img[0:10, 0:10].copy()
```

## 3. Kreslení do obrazu
Funkce pro kreslení přímo modifikují zadaný obraz.
Souřadnice se zadávají jako `(x, y)`.

```python
# Úsečka (obraz, start, konec, barva, tloušťka)
cv.line(img, (0, 0), (100, 100), (255, 0, 0), 5)

# Obdélník (obraz, levý-horní, pravý-dolní, barva, tloušťka)
cv.rectangle(img, (50, 50), (150, 150), (0, 255, 0), 3)

# Kružnice (obraz, střed, poloměr, barva, tloušťka (-1 = výplň))
cv.circle(img, (200, 200), 30, (0, 0, 255), -1)

# Text (obraz, text, pozice, font, měřítko, barva, tloušťka)
cv.putText(img, 'OpenCV', (10, 400), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
```

## 4. Operace s obrazem (I/O a zpracování)

### Načtení a uložení
```python
# Načtení obrazu (1 = barevně, 0 = šedotón)
img = cv.imread('vstup.png', 1)

# Uložení obrazu
cv.imwrite('vystup.jpg', img)
```

### Změna velikosti (Resize)
```python
# Změna velikosti na polovinu pomocí faktorů fx, fy
resized = cv.resize(img, None, fx=0.5, fy=0.5)
```

### Konverze barev
```python
# BGR -> Grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# BGR -> HSV
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
```

> **Poznámka:** HSV (Hue, Saturation, Value) je barevný model často používaný pro segmentaci barev, protože odděluje informaci o barvě (Hue) od jasu (Value).

## 5. Práce s videem
Pro čtení z kamery nebo videa se používá `VideoCapture`.

```python
cap = cv.VideoCapture(0) # 0 = výchozí kamera

while True:
    ret, frame = cap.read() # Čtení snímku
    if not ret: break

    # Zpracování snímku...
    
    cv.imshow('Kamera', frame)
    
    # Ukončení klávesou 'q' (čekání 1ms)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
```
