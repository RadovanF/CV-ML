# Detektory objektů založené na Transformerech

**(PRACOVNÍ VERZE v1)**

Navazuje na přednášky o detekci (R-CNN, YOLO) a o Vision Transformeru (ViT) - tentokrát z pohledu transformer + lokalizace objektů.

---

## Přehled

| Model | Rok | Co dělá jinak | Ukázka |
|---|---|---|---|
| **DETR** | 2020 | Obrázek zpracuje klasická CNN. Transformer pak dostane 100 dotazů a každý z nich vrátí buď jeden objekt (třída + bounding box), nebo „nic tu není“. Nepotřebuje anchory ani NMS. | Hugging Face |
| **YOLOS** | 2021 | Vezme obyčejný ViT z minulé přednášky a místo jednoho CLS tokenu mu přidá 100 „detekčních tokenů“. Každý z nich se naučí najít jeden objekt. Žádná CNN, žádný decoder. | Hugging Face |
| **ViTDet** | 2022 | ViT slouží jen jako backbone, tedy extraktor příznaků. Na detekci se pak použije klasická hlava z rodiny R-CNN. Ukazuje, že obyčejný ViT je dobrý základ i pro detekci. | – |
| **RT-DETR** | 2023 | DETR zrychlený tak, aby běžel v reálném čase a konkuroval YOLO. V Ultralytics se používá i trénuje stejně jako YOLO. | Ultralytics |
| **RF-DETR** | 2025 | Real-time DETR s velkým předtrénovaným ViT (DINOv2). Z jednoho tréninku umí vybrat rychlejší nebo přesnější variantu podle potřeby. Dobře se dotrénuje na vlastní data. | `rfdetr` (Roboflow) |

---

## 0. Pojmy, které budeme potřebovat

### 0.1 Backbone, neck, head (opakování z YOLO)
- **Backbone** – extrahuje příznaky z obrázku (např. ResNet, nebo nově ViT).
- **Neck** – kombinuje příznaky z různých měřítek (např. FPN).
- **Head** – z příznaků predikuje třídy a bounding boxy.

U detektorů v této přednášce může být Transformer na různých místech: jen v backbonu (ViTDet), jen v hlavě (DETR, RT-DETR) nebo všude (YOLOS, RF-DETR).

### 0.2 Encoder a decoder v Transformeru
Pozor, v Transformerech neznamená encoder-decoder totéž co u autoencoderu nebo U-Netu. Tam encoder data **stlačí** do malého „hrdla“ (bottleneck) a decoder je zase rozbalí. V Transformeru se nic nestlačuje:

- **Encoder** – tokeny si „povídají mezi sebou“ (self-attention). Kolik tokenů jde dovnitř, tolik jich vyjde ven, jen jsou obohacené o kontext. Ve ViT jde dovnitř 197 × 768 a ven zase 197 × 768.

### 0.3 Object queries (dotazy na objekty)
Naučitelné vektory, typicky jich je 100–300. Fungují podobně jako CLS token ve ViT, jen jich je víc a každý se během tréninku „specializuje“ na hledání jednoho objektu. Výstupem každého dotazu je **třída + box**, nebo třída **∅ (žádný objekt)**.

### 0.4 Hungarian matching – proč není potřeba NMS
- **Klasické YOLO** (v1 až v9, v11, v12): jeden objekt může predikovat víc kandidátů najednou (sousední buňky, více boxů nebo anchorů na buňku), proto se duplicitní boxy musí na konci odfiltrovat pomocí **NMS**. Novější YOLOv10 a YOLO26 se NMS zbavily podobnou myšlenkou jako DETR: při tréninku přiřazují jednomu objektu jedinou predikci.
- **DETR:** model vrátí pevný počet predikcí (např. 100). Při tréninku se **Hungarian algoritmem** najde přiřazení „1 skutečný objekt ↔ právě 1 predikce“ s nejmenší celkovou chybou (shoda třídy + vzdálenost boxů). Zbylé predikce se musí naučit říkat „∅“.
- Za každý objekt je tedy zodpovědná **jen jedna predikce** a model je za duplicity při tréninku penalizován. Proto **NMS odpadá**.
- Jinými slovy: odstraňování duplicit už **není ručně napsaný krok v postprocessingu, ale síť se ho naučí sama** během tréninku.

---

## 1. DETR – DEtection TRansformer
*Carion et al., Facebook AI, 2020 – „End-to-End Object Detection with Transformers“*

**Architektura:**
1. **CNN backbone** (ResNet-50) vytvoří mapu příznaků obrázku.
2. Mapa se rozloží na sekvenci tokenů (každá pozice = jeden token) a přičte se poziční kódování, stejně jako u ViT.
3. **Transformer encoder (self-attention)** – tokeny obrázku se dívají navzájem na sebe.
4. **Transformer decoder (cross-attention)** – 100 object queries se dívá do tokenů obrázku a každý si vytáhne jeden objekt.
5. Každý výstup decoderu → malá FFN → **třída + box** (nebo ∅).

**Co je nové:** žádné anchory, žádné NMS, žádné ručně laděné kroky. Celý detektor je jedna síť trénovaná end-to-end.

**Slabiny:** trénuje se velmi dlouho (stovky epoch) a hůř detekuje malé objekty. Většina pozdějších variant (Deformable DETR, RT-DETR, RF-DETR) řeší právě tyto dva problémy.

### Ukázka – DETR přes Hugging Face
```python
from transformers import pipeline

detector = pipeline("object-detection", model="facebook/detr-resnet-50")
url = "http://images.cocodataset.org/val2017/000000039769.jpg"

for d in detector(url):
    print(f"{d['label']:10} {d['score']:.2f} {d['box']}")
```

---

## 2. YOLOS – You Only Look at One Sequence
*Fang et al., 2021 – „You Only Look at One Sequence: Rethinking Transformer in Vision through Object Detection“*

**Myšlenka:** jak daleko se dostane **čistý ViT** bez CNN a bez znalosti 2D struktury obrázku?

**Úpravy oproti ViT (z minulé přednášky):**
1. CLS token se **odstraní** a místo něj se přidá **100 naučitelných [DET] tokenů**.
2. Sekvence = 196 patch tokenů + 100 [DET] tokenů, ke všem se přičtou poziční embeddingy (rovnice 1 z ViT, jen s [DET] místo [CLS]).
3. Projde **jen encoderem ViT**, žádný decoder. Patch tokeny i [DET] tokeny se berou rovnocenně.
4. Každý [DET] token → MLP → třída + box. Trénuje se Hungarian matchingem jako DETR.

**Shrnutí:** YOLOS je ViT, který má místo jednoho CLS tokenu 100 [DET] tokenů. Přesností nekonkuruje specializovaným detektorům. Ukazuje ale, že obyčejný ViT zvládne detekci téměř bez úprav.

**YOLOS vs. DETR:** YOLOS = ViT, ve kterém je CLS token stokrát a každý hledá jeden objekt. DETR = CNN + Transformer s decoderem.

### Ukázka – YOLOS přes Hugging Face
```python
from transformers import pipeline

detector = pipeline("object-detection", model="hustvl/yolos-tiny")   # nebo hustvl/yolos-small
url = "http://images.cocodataset.org/val2017/000000039769.jpg"

for d in detector(url):
    print(f"{d['label']:10} {d['score']:.2f} {d['box']}")
```

---

## 3. ViTDet – ViT jako backbone pro detekci
*Li, Mao, Girshick, He, Facebook AI Research, 2022 – „Exploring Plain Vision Transformer Backbones for Object Detection“*

**Problém:** detektory byly navržené pro CNN backbony, které přirozeně vytvářejí příznaky ve více měřítkách (pyramidu). Obyčejný ViT má po celou dobu **jedno měřítko** (mřížka 14 × 14 tokenů). Je vůbec použitelný pro detekci malých i velkých objektů?

**Řešení – minimum úprav:**
1. **Jednoduchá pyramida příznaků:** z poslední mapy ViT se pomocí zvětšení a zmenšení (deconvoluce, pooling) vyrobí 4 měřítka. Klasické FPN není potřeba.
2. **Window attention:** u velkých obrázků je attention mezi všemi tokeny drahá, proto se počítá jen uvnitř oken. K tomu stačí **pár bloků s globální attention**, které propojí informace mezi okny.
3. Detekční hlava je klasická z rodiny **R-CNN** (Mask R-CNN / Cascade Mask R-CNN), tedy navazuje na přednášku o R-CNN. **NMS tu zůstává.**
4. Backbone je předtrénovaný jako **MAE** (Masked Autoencoder): zakryje se 75 % patchů a model je dopočítává, stejně jako BERT dopočítává zakrytá slova.

**Výsledek:** dosáhl až 61,3 AP (box) na COCO, srovnatelně s nejlepšími hierarchickými backbony. Transformer tu je **jen v backbonu**, zbytek detektoru je klasický.

---

## 4. RT-DETR – Real-Time DETR
*Zhao et al., Baidu, 2023 (CVPR 2024) – „DETRs Beat YOLOs on Real-time Object Detection“*

**Cíl:** zachovat výhody DETR (bez NMS, end-to-end) a zrychlit ho na úroveň YOLO.

**Hlavní triky:**
- **Hybridní encoder:** drahá self-attention se počítá jen na nejmenší (nejhrubší) mapě příznaků. Spojení měřítek dělají levné konvoluce. Většina výpočtu DETR byla právě v encoderu.
- **Chytrý výběr dotazů:** každé místo v příznacích dostane skóre, jak pravděpodobně tam je objekt. 300 míst s nejvyšším skóre se stane výchozími dotazy pro decoder, který jejich boxy zpřesní.
- **Nastavitelná rychlost:** při inferenci lze ubrat vrstvy decoderu a zrychlit model **bez nového tréninku**.

### Ukázka – inference v Ultralytics
```python
from ultralytics import RTDETR

model = RTDETR("rtdetr-l.pt")
r = model("http://images.cocodataset.org/val2017/000000039769.jpg")[0]

for b in r.boxes:
    print(f"{r.names[int(b.cls)]:10} {float(b.conf):.2f} {[round(v) for v in b.xyxy[0].tolist()]}")
```

### Ukázka – trénink na vlastních datech (stejně jako YOLO)
```python
from ultralytics import RTDETR

model = RTDETR("rtdetr-l.pt")
model.train(data="muj_dataset.yaml", epochs=50, imgsz=640)   # stejný yaml a formát anotací jako u YOLO
```

---

## 5. RF-DETR – Roboflow DETR
*Robinson et al., Roboflow + Carnegie Mellon University, 2025 (ICLR 2026) – „RF-DETR: Neural Architecture Search for Real-Time Detection Transformers“*

**Problém:** moderní detektory (včetně YOLO) jsou podle autorů „přeučené na COCO“, tedy vyladěné na jeden benchmark, a na reálných datech s jinými třídami a jiným počtem objektů fungují hůř.

**Hlavní myšlenky:**
1. **Velký předtrénovaný ViT backbone – DINOv2.** Je předtrénovaný bez anotací na obrovském množství obrázků, takže „zná svět“ a dobře se přenáší na nové domény (medicína, průmysl, zemědělství…).
2. Staví na LW-DETR. Střídá **okénkovou a globální attention** (podobně jako ViTDet) a v decoderu používá **deformable cross-attention**, kde se každý dotaz dívá jen na několik vybraných míst obrázku, ne na všechny tokeny.
3. **Jeden trénink → mnoho variant (weight-sharing NAS):** během tréninku se náhodně mění velikost patche, rozlišení, počet vrstev decoderu, počet dotazů a počet oken. Po tréninku si lze vybrat variantu podle požadované rychlosti **bez dalšího tréninku**.
4. Existuje i varianta pro segmentaci: **RF-DETR-Seg**.

### Ukázka – inference
Dokumentace: https://github.com/roboflow/rf-detr
```python
# pip install rfdetr
from rfdetr import RFDETRBase                     # novější verze: RFDETRNano, RFDETRSmall, RFDETRMedium, …
from rfdetr.assets.coco_classes import COCO_CLASSES

model = RFDETRBase()
detections = model.predict("000000039769.jpg", threshold=0.5)    # výstup = supervision.Detections

for cid, conf, box in zip(detections.class_id, detections.confidence, detections.xyxy):
    print(f"{COCO_CLASSES[cid]:10} {conf:.2f} {box.round()}")
```

### Ukázka – trénink na vlastních datech
```python
from rfdetr import RFDETRBase

model = RFDETRBase()
model.train(
    dataset_dir="muj_dataset",
    epochs=50,
    batch_size=4,
    grad_accum_steps=4,
    lr=1e-4,
    output_dir="vystup",
)
```
Použití je podobné jako u Ultralytics YOLO.
