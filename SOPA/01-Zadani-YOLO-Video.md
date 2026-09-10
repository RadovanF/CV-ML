# Zadání 01: Detekce dopravních objektů ve videu (YOLO)

---

## Cíl úlohy
Načíst vstupní videozáznam silničního provozu, pomocí neuronové sítě **YOLO** v něm detekovat objekty a omezit výsledky pouze na vybrané dopravní třídy (osoby, vozidla, cyklisté). Získané detekce vizualizovat přímo v obraze pomocí knihovny **OpenCV**.

---

## 1. Příprava prostředí
K řešení jsou potřeba knihovny `ultralytics` (YOLO) a `opencv-python`.

```bash
pip install ultralytics opencv-python
```

---

## 2. Vybrané třídy z datasetu COCO

| ID třídy | Název v COCO | Český název | Doporučená barva (BGR) |
| :---: | :--- | :--- | :--- |
| **0** | `person` | Osoba | `(0, 255, 0)` – zelená |
| **1** | `bicycle` | Jízdní kolo | `(255, 255, 0)` – azurová |
| **2** | `car` | Osobní automobil | `(255, 0, 0)` – modrá |
| **3** | `motorcycle` | Motocykl | `(0, 165, 255)` – oranžová |
| **5** | `bus` | Autobus | `(0, 0, 255)` – červená |

---

## 3. Požadavky na řešení (Úkoly)

1. **Inference s filtrací:** Využít parametr `classes` při volání modelu a omezit detekci pouze na třídy `[0, 1, 2, 3, 5]`.
2. **Vlastní vykreslení rámečků:** Místo výchozího zobrazení projít nalezené objekty v `results[0].boxes` a vykreslit:
   - Rámeček objektu danou barvou (`cv2.rectangle`).
   - Popisek s českým názvem třídy a hodnotou jistoty (`cv2.putText`).
3. **BONUS: Statistika v obraze:** Spočítat počty jednotlivých detekovaných objektů v aktuálním snímku a zobrazit souhrnný text v horním/dolním pruhu videa.
---

## 4. Spustitelná šablona kódu

Níže uvedený skript načte video, spustí model a vypisuje základní informace o detekcích do příkazové řádky. Vaším úkolem je doplnit vyznačené části (`TODO`):

```python
import cv2
from ultralytics import YOLO

# 1. Definice požadovaných tříd a barev
TARGET_CLASSES = {
    0: ["Osoba", (0, 255, 0)],
    1: ["Jizdni kolo", (255, 255, 0)],
    2: ["Automobil", (255, 0, 0)],
    3: ["Motocykl", (0, 165, 255)],
    5: ["Autobus", (0, 0, 255)],
}

# 2. Inicializace modelu a videa
model = YOLO("yolo26n.pt")
cap = cv2.VideoCapture("vstupni_video.mp4")

if not cap.isOpened():
    print("Chyba: Nelze otevrit video.")
    exit(1)

cv2.namedWindow("YOLO Detekce", cv2.WINDOW_NORMAL)
frame_id = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        if not ret:
            break

    frame_id += 1

    # 3. Spuštění modelu
    # TODO: Pridejte parametr classes pro filtraci pouze vybranych trid
    # TODO: Nastavte minimalni prah spolehlivosti conf=0.25
    results = model.predict(frame, verbose=False)
    boxes = results[0].boxes

    # Ukázkový výpis detekcí do terminálu
    print(f"--- Snimek {frame_id}: Nalezeno {len(boxes)} objektu ---")
    for box in boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        print(f"  Trida ID: {cls_id}, Jistota: {conf:.2f}, Souradnice: [{x1}, {y1}, {x2}, {y2}]")

        # TODO: vykreslete ramecek kolem objektu pomoci cv2.rectangle()
        # TODO: Pridejte textovy popisek tridy pomoci cv2.putText()

    # TODO BONUS: Spocitejte pocty objektu pro kazdou tridu a vykreslete informacni pruh do dolni casti obrazu

    # Zobrazeni snimku
    cv2.imshow("YOLO Detekce", frame)

    key = cv2.waitKey(1)
    if key in [ord("q"), ord("Q")]:
        break

cap.release()
cv2.destroyAllWindows()
```

---

## 5. Ukázka výsledného řešení

Ukázkové video výstupu: [vystup_01_yolo_video.mp4](vystup_01_yolo_video.mp4)

<video src="vystup_01_yolo_video.mp4" controls="controls" width="100%"></video>

