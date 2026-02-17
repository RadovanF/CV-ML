# Detekce a rozpoznávání objektů – Tradiční metody (Haar, HOG, LBP, SVM)

Tento text slouží jako studijní materiál k základům detekce a rozpoznávání objektů v počítačovém vidění. Pokrývá tradiční metody: detekci obličejů pomocí Haarových příznaků (Viola-Jones), deskriptory HOG a LBP, klasifikátory SVM a AdaBoost a rozpoznávání obličejů pomocí LBP histogramů.

## 1. Co je to detekce objektů?

Detekce objektů je klíčová úloha počítačového vidění. Na rozdíl od prosté klasifikace (která říká *co* je na obrázku) detekce odpovídá na dvě otázky:
1.  **Co** je to za objekt? (Klasifikace)
2.  **Kde** se objekt nachází? (Lokalizace)

Výstupem je obvykle ohraničující rámeček (bounding box), který definuje pozici a měřítko objektu v obraze.

---

## 2. Přístup posuvného okna (Sliding Window)

Jednou z tradičních a intuitivních metod pro detekci je technika **posuvného okna**.

### Jak to funguje?
1.  **Skenování:** Po vstupním obraze se posouvá obdélníkové okno pevné velikosti.
2.  **Měřítko:** Abychom našli objekty různých velikostí, proces se opakuje pro zmenšené verze obrazu (nebo se zvětšuje okno).
3.  **Extrakce a klasifikace:** Z každého výřezu (pod-okna) se vypočítají příznaky (features), které se předají klasifikátoru (např. SVM). Ten rozhodne, zda je objekt přítomen.

**Nevýhody:**
* **Výpočetní náročnost:** Generuje se obrovské množství výřezů, které je nutné klasifikovat.
* **Vícenásobné detekce:** Často dochází k několika pozitivním detekcím kolem jednoho objektu, které je nutné následně sloučit.

---

## 3. Detekce obličejů: Viola-Jones detektor

Tento detekční přístup (framework) pro detekci objektů (autoři Viola-Jones, 2001), se stal svého času standardem pro detekci obličejů v reálném čase.

Metoda stojí na čtyřech hlavních pilířích:

### A. Haarovy příznaky (Haar Features)
Místo práce s jednotlivými pixely využívá algoritmus rozdíly v jasu mezi sousedními obdélníkovými oblastmi. Tyto příznaky připomínají Haarovy vlnky.

* **Princip:** $Hodnota = \sum(černé\_pixely) - \sum(bílé\_pixely)$
* **Typy příznaků:**
    * *Hranové (Edge):* Detekují přechod mezi tmavou a světlou oblastí (např. obočí vs. čelo).
    * *Liniové (Line):* Detekují tmavý pruh mezi dvěma světlými (např. rty) nebo naopak (např. hřbet nosu).
    * *Čtyř-obdélníkové:* Pro diagonální detaily.

### B. Integrální obraz (Integral Image)
Aby bylo možné počítat sumy pixelů v obdélnících extrémně rychle, obraz se převede na tzv. integrální obraz.
* Každý bod v integrálním obraze obsahuje součet všech pixelů nacházejících se **nalevo a nahoře** od něj.
* Díky tomu lze součet libovolného obdélníku spočítat pomocí pouhých **4 odkazů do paměti** (rohů obdélníku), bez ohledu na jeho velikost.
* To umožňuje vyhodnocovat Haarovy příznaky v konstantním čase.

### C. AdaBoost (Adaptive Boosting)
Algoritmus generuje pro každý obrázek stovky tisíc možných Haarových příznaků (různé velikosti a pozice), ale většina z nich je pro detekci obličeje zbytečná. AdaBoost tento problém řeší následovně:

* **Cíl:** Vybrat z obrovského množství jen ty nejdůležitější příznaky a vytvořit z nich jeden silný klasifikátor.
* **Princip slabých klasifikátorů:** AdaBoost kombinuje mnoho tzv. "slabých klasifikátorů". Slabý klasifikátor je velmi jednoduchý (např. rozhoduje jen podle jednoho jediného příznaku, třeba "je oblast očí tmavší než tváře?"). Sám o sobě funguje jen o trochu lépe než náhodný tip, ale v týmu je silný.
* **Učení se z chyb (Iterativní proces):**
    1.  V prvním kole mají všechny trénovací obrázky stejnou váhu (důležitost).
    2.  Vybere se první slabý klasifikátor, který dělá nejméně chyb.
    3.  **Klíčový moment:** Obrázkům, které tento klasifikátor určil špatně, se **zvýší váha**.
    4.  V dalším kole musí nový slabý klasifikátor věnovat větší pozornost těmto "těžkým" příkladům, protože mají vyšší váhu.
* **Výsledek:** Finální silný klasifikátor je váženým součtem těchto slabých klasifikátorů. Ti přesnější mají ve finálním hlasování silnější slovo.

### D. Kaskádový klasifikátor (Cascade of Classifiers)
Pro maximální rychlost se klasifikátory řadí do kaskády (série fází).
1.  **První fáze:** Obsahuje jen několik nejzásadnějších příznaků. Pokud okno neprojde touto fází, je okamžitě zamítnuto a dál se nepočítá.
2.  **Další fáze:** Jsou složitější a přesnější. Aplikují se jen na okna, která prošla předchozími fázemi.
* **Výsledek:** Většina obrazu (pozadí) je zahozena velmi rychle, výpočetní výkon se soustředí jen na nadějné oblasti.

---

## 4. Další metody detekce a klasifikace

Kromě Viola-Jones existují další techniky, které se často používají, například pro detekci chodců (Pedestrian Detection) nebo rozpoznávání textur.

### HOG (Histograms of Oriented Gradients)
Tato metoda se zaměřuje na tvar a strukturu objektu pomocí rozložení směrů gradientů (hran).
* **Princip:** Obraz se rozdělí na malé buňky (cells). V každé buňce se spočítá histogram orientací gradientů.
* **Normalizace:** Histogramy se normalizují přes větší bloky, což zajišťuje odolnost vůči změnám osvětlení a stínům.
* **Využití:** Často se používá v kombinaci s SVM klasifikátorem pro detekci osob, kde je tvar těla spolehlivějším vodítkem než barva.

### LBP (Local Binary Patterns)
LBP je deskriptor textur, který zachycuje lokální struktury obrazu (hrany, rohy, ploché oblasti) porovnáním každého pixelu s jeho okolím.

* **Princip:**
    1. Pro každý pixel se vezme okolí, např. 3×3 (8 sousedů).
    2. Každý soused se porovná s centrálním pixelem: pokud je intenzita souseda ≥ centrální pixel, přiřadí se `1`, jinak `0`.
    3. Výsledkem je 8bitové binární číslo, které se převede na decimální hodnotu – tzv. **LBP kód**.
    4. Směr čtení sousedů (po/proti směru hodinových ručiček) musí být konzistentní pro celý obraz.

* **Vlastnosti:**
    * Odolnost vůči monotónním změnám jasu.
    * Výpočetně nenáročný – vhodný pro embedded systémy a zařízení bez GPU.
    * Výstupem je histogram LBP kódů, který lze použít jako vektor příznaků pro klasifikaci (SVM, k-NN).

* **Rozšíření:** Existují varianty s jiným počtem sousedů a poloměrem (např. LBP s 16 sousedy na poloměru 2), a tzv. uniform patterns, které omezují počet příznaků na ty nejčastější.

### SVM (Support Vector Machines)
SVM je algoritmus strojového učení používaný pro klasifikaci (rozhodnutí, zda je ve výřezu objekt nebo ne).
* **Princip:** Snaží se najít optimální hranici (nadrovinu), která odděluje data dvou tříd (např. "obličej" a "ne-obličej") s co největším odstupem (marginem).
* **Support Vectors:** Jsou to klíčové body dat, které leží nejblíže dělící hranici a definují její polohu.
* **Jádrový trik (Kernel trick):** Pokud data nelze oddělit rovnou čárou, SVM je umí převést do vyšší dimenze, kde oddělitelná jsou.

### Deep Learning (Hluboké učení)
V současnosti v oboru dominují hluboké neuronové sítě. Oproti tradičním metodám (kde je nutné ručně definovat příznaky jako Haar nebo HOG) se sítě učí příznaky automaticky přímo z dat.
* **CNNs (Konvoluční neuronové sítě):** Základ detekce a rozpoznávání v deep learning přístupu.
* **YOLO / SSD:** Detektory schopné najít objekty v jediném průchodu sítí.

> **Poznámka:** Deep learning přístupy přesahují rozsah tohoto textu (budou řešeny v dalších materiálech). Zde jsou uvedeny pro kontext a srovnání s tradičními metodami.

---

## 5. Rozpoznávání obličejů (Face Recognition)

Rozpoznávání obličejů je navazující úloha na detekci. Zatímco detekce odpovídá na otázku "kde je obličej?", rozpoznávání odpovídá na otázku "čí je to obličej?".

### Pipeline rozpoznávání obličejů
Proces se skládá ze tří kroků:
1. **Detekce obličeje** – nalezení oblasti obličeje v obraze (např. Viola-Jones, viz sekce 3).
2. **Extrakce příznaků** – z oblasti obličeje se vypočítá vektor příznaků (feature vector), který obličej numericky popisuje.
3. **Rozpoznání** – porovnání vypočítaného vektoru s databází známých obličejů (např. pomocí euklidovské vzdálenosti, chi-square nebo k-NN klasifikátoru).

### LBP pro rozpoznávání obličejů
LBP deskriptor (viz sekce 4) lze využít pro extrakci příznaků obličeje následujícím postupem:

1. **Převod na LBP obraz:** Celý obraz obličeje se transformuje – každý pixel je nahrazen svým LBP kódem.
2. **Rozdělení na oblasti:** Transformovaný obraz se rozdělí na mřížku oblastí (např. 7×7 nebo 8×8).
3. **Výpočet histogramů:** Pro každou oblast se spočítá histogram LBP kódů.
4. **Zřetězení:** Histogramy všech oblastí se zřetězí do jednoho dlouhého vektoru příznaků. Tím se zachová prostorová informace (histogram z oblasti očí je jiný než z oblasti úst).
5. **Porovnání / klasifikace:** Zřetězený histogram slouží jako vstup pro rozpoznání. Lze použít:
    * **Přímé porovnání vzdáleností** mezi histogramem nového obličeje a uloženými histogramy:
        * Euklidovská vzdálenost
        * Chi-square vzdálenost
        * Vzdálenost absolutních hodnot
    * **Trénovací klasifikátory**, které se naučí rozhodovací hranice z trénovacích dat:
        * **k-NN (k-Nearest Neighbors)** – klasifikace podle k nejbližších sousedů v prostoru histogramů (používá např. OpenCV `LBPHFaceRecognizer`)
        * **SVM** – natrénuje se na histogramech jako vektorech příznaků, vhodný pro větší datasety

**Vlastnosti LBP pro rozpoznávání obličejů:**
* Odolnost vůči změnám osvětlení (LBP porovnává relativní rozdíly).
* Rychlý výpočet – vhodné i pro zařízení s omezeným výkonem.
* Rozdělení na oblasti zachovává prostorovou strukturu obličeje.

---

## 6. Ukázky implementace v Pythonu

### 6.1 Detekce obličejů – Haar kaskádový klasifikátor (OpenCV)

```python
import cv2

# 1. Načtení klasifikátoru (XML soubor s natrénovanými daty)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# 2. Načtení obrázku a převod na šedotón (detekce probíhá v šedé škále)
img = cv2.imread("image.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 3. Detekce
# scaleFactor: O kolik se zmenší obraz při každém kroku škálování (např. 1.2 = o 20 %)
# minNeighbors: Kolik sousedních obdélníků musí detekci potvrdit, aby byla platná
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.2,
    minNeighbors=3,
    minSize=(100, 100),
    maxSize=(500, 500)
)

# 4. Vykreslení výsledků
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

cv2.imshow('Detekce obliceje', img)
cv2.waitKey(0)
```

### 6.2 Rozpoznávání obličejů – LBP (OpenCV)

OpenCV poskytuje třídu `LBPHFaceRecognizer` pro rozpoznávání obličejů pomocí LBP histogramů.

```python
import cv2

# Vytvoření rozpoznávače
recognizer = cv2.face.LBPHFaceRecognizer_create(
    radius=1,       # Poloměr LBP okolí
    neighbors=8,    # Počet sousedních bodů
    grid_x=8,       # Počet oblastí ve směru x
    grid_y=8        # Počet oblastí ve směru y
)
```

> **Poznámka:** Modul `cv2.face` je součástí balíčku `opencv-contrib-python`. Instalace: `pip install opencv-contrib-python`

### 6.3 Extrakce LBP příznaků – scikit-image

Alternativně lze LBP příznaky extrahovat pomocí knihovny scikit-image:

```python
from skimage.feature import local_binary_pattern
import cv2
import numpy as np

# 1. Načtení obrázku v šedotónu
img = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)

# 2. Výpočet LBP
radius = 1          # Poloměr okolí
n_points = 8        # Počet sousedních bodů (8 pro základní LBP)
lbp = local_binary_pattern(img, n_points, radius, method='uniform')
# method='uniform' omezí počet příznaků na tzv. uniform patterns
```