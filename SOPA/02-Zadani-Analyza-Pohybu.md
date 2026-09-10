# Zadání 02: Odhad lidské pózy a počítadlo dřepů (YOLO Pose)

---

## Cíl úlohy
Využít neuronovou síť pro odhad polohy lidského těla (**Human Pose Estimation**), ze záznamu pohybu extrahovat 17 klíčových anatomických bodů a vytvořit automatické počítadlo provedených dřepů (nápověda: úhel v kolenním kloubu nebo vzdálenost mezi kyčlí a kotníkem).

---

## 1. Příprava prostředí
K řešení jsou potřeba knihovny `ultralytics` (YOLO) a `opencv-python`.

```bash
pip install ultralytics opencv-python
```

---

## 2. Klíčové body modelu YOLO Pose
Model vrací pro každou osobu 17 bodů o souřadnicích `[x, y]` a jistotě `conf`:

* **0:** nos, **1–2:** oči, **3–4:** uši
* **5–6:** ramena, **7–8:** lokty, **9–10:** zápěstí
* **11:** levá kyčel, **12:** pravá kyčel
* **13:** levé koleno, **14:** pravé koleno
* **15:** levý kotník, **16:** pravý kotník

Pro analýzu nohou slouží indexy:
* **Pravá noha:** kyčel (12) – koleno (14) – kotník (16)
* **Levá noha:** kyčel (11) – koleno (13) – kotník (15)

---

## 3. Postup

1. **Výpočet úhlu:** Napsat funkci pro výpočet úhlu mezi třemi body a aplikovat ji na souřadnice kyčle, kolene a kotníku.
2. **Počítadlo dřepů:** Využít proměnnou pro sledování fáze (`nahore` / `dole`) a přičítat opakování pouze při dokončení celého pohybu.
3. **Vizualizace v obraze:** Vykreslit naměřený úhel ke koleni a velkým číslem zobrazit celkový počet dřepů na obrazovce.

---

## 3. Spustitelná šablona kódu

Níže uvedený skript načte video, spustí model a vypisuje souřadnice kloubů do příkazové řádky. Vaším úkolem je doplnit další části:

```python
import cv2
import numpy as np
from ultralytics import YOLO

# Inicializace modelu a videa
model = YOLO("yolo26n-pose.pt")
cap = cv2.VideoCapture("vstupni_video_pohybu.mp4")

if not cap.isOpened():
    print("Chyba: Nelze otevrit video.")
    exit(1)

pocet_drepu = 0
faze = "nahore"  # pamatujeme si, zda je clovek "nahore" nebo "dole"

cv2.namedWindow("Analyza pohybu", cv2.WINDOW_NORMAL)
frame_id = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        if not ret:
            break

    frame_id += 1

    # Spuštění odhadu pózy
    results = model(frame, verbose=False)
    result = results[0]

    if result.keypoints is not None and len(result.keypoints.xy) > 0:
        pts = result.keypoints.xy[0].cpu().numpy()

        if len(pts) >= 17:
            # Body pro pravou nohu: kycel (12), koleno (14), kotnik (16)
            r_hip, r_knee, r_ankle = pts[12], pts[14], pts[16]

            # Ukázkový výpis souřadnic do terminálu
            print(f"Snimek {frame_id}: Kycel={r_hip.astype(int)}, Koleno={r_knee.astype(int)}, Kotnik={r_ankle.astype(int)}")

    # Zobrazení výchozí kostry pro kontrolu
    display_frame = result.plot()
    cv2.imshow("Analyza pohybu", display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key in [ord("q"), ord("Q"), 27]:
        break

cap.release()
cv2.destroyAllWindows()
```

---

## 5. Ukázka výsledného řešení

Ukázkové video výstupu: [vystup_02_analyza_pohybu.mp4](vystup_02_analyza_pohybu.mp4)

<video src="vystup_02_analyza_pohybu.mp4" controls="controls" width="100%"></video>

