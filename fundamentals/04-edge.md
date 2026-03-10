# Úvod do detekce hran a filtrace obrazu

Tento dokument je rychlým úvodem do zpracování obrazu se zaměřením na detekci hran. Jsou zde vysvětleny matematické principy (konvoluce), standardní algoritmy (Sobel, Canny) a propojení s konvolučními neuronovými sítěmi.

---

## 1. Definice hrany a význam detekce

Hrany jsou definovány jako jedny z nejvýznamnějších obrazových příznaků (features) v počítačovém vidění. Jsou jimi reprezentována místa v obraze, kde je detekována náhlá změna intenzity pixelů (jasový skok).

### Význam detekce hran:
* **Redukce dat:** Z původního obrazového pole je zachována pouze základní struktura.
* **Identifikace objektů:** Pomocí hran jsou definovány obrysy objektů, což je využíváno pro jejich následnou klasifikaci.
* **Lokalizace:** Je umožněno přesné určení hranic pro měření rozměrů.

Pokud si obraz představíme jako 3D krajinu (kde výška odpovídá jasu pixelu), hrana je "sráz" (oblast s nejvyšším sklonem). V matematice takové změny hledáme pomocí **derivace** (gradientu). Kde je změna jasu nejstrmější, tam je derivace nejvyšší.

---

## 2. Matematický základ: Diskrétní konvoluce

Operace detekce hran jsou založeny na matematické operaci diskrétní konvoluce. Výpočet je prováděn pomocí konvoluční masky (jádra), např. o velikosti $3 \times 3$, kterou je postupně procházen vstupní obraz.

### Vzorec pro 2D konvoluci
Pro vstupní obraz $f$ a masku $h$ je nová hodnota pixelu obrazu $g$ vypočtena vztahem:

$$g(x, y) = \sum_{i=-k}^{k} \sum_{j=-k}^{k} f(x-i, y-j) \cdot h(i, j)$$

**Princip výpočtu:**
1. Maska je aplikována na definovanou oblast obrazu.
2. Hodnoty v masce jsou vynásobeny odpovídajícími hodnotami pixelů.
3. Všechny tyto součiny jsou sečteny.
4. Výsledná hodnota je zapsána do středového pixelu nového obrazu.

*(Poznámka: V softwarových implementacích je často reálně prováděna operace korelace, kdy maska není před výpočtem středově převrácena. U symetrických masek (které budeme v rámci kurzu používat nejčastěji) je výsledek obou matematických operací totožný).*

### Příklad výpočtu konvoluce

Mějme výřez obrazu $f$ o velikosti $3 \times 3$, který obsahuje svislou světlou hranu, a symetrickou masku $h$. Počítáme novou hodnotu pro **středový pixel**.

**1. Vstupní data**

Výřez obrazu $f$ (světlá hrana uprostřed):

$$
\begin{bmatrix} 
10 & 100 & 10 \\ 
10 & 100 & 10 \\ 
10 & 100 & 10 
\end{bmatrix}
$$

Maska $h$:

$$
\begin{bmatrix} 
0 & -1 & 0 \\ 
-1 & 4 & -1 \\ 
0 & -1 & 0 
\end{bmatrix}
$$

**2. Postup výpočtu**

Protože je maska symetrická, vynásobíme hodnoty obrazu a masky na stejných pozicích a vše sečteme:

$$g = (10 \cdot 0) + (100 \cdot -1) + (10 \cdot 0)$$
$$+ (10 \cdot -1) + (100 \cdot 4) + (10 \cdot -1)$$
$$+ (10 \cdot 0) + (100 \cdot -1) + (10 \cdot 0)$$

**3. Výsledek**

$$g = 0 - 100 + 0 - 10 + 400 - 10 + 0 - 100 + 0$$
$$g = 180$$

Nová hodnota středového pixelu je **180**.

---

## 3. Sobelův detektor

Sobelův operátor je využíván k detekci hran pomocí konvoluce. Jsou aplikovány dvě specifické masky – pro vertikální a horizontální směr.

### Konvoluční masky (Jádra $3 \times 3$)

**1. Horizontální gradient ($G_x$):**
Je jím reagováno na vertikální hrany (změna jasu v ose x).

$$
G_x = \begin{bmatrix} 
-1 & 0 & 1 \\ 
-2 & 0 & 2 \\ 
-1 & 0 & 1 
\end{bmatrix}
$$

**2. Vertikální gradient ($G_y$):**
Je jím reagováno na horizontální hrany (změna jasu v ose y).

$$
G_y = \begin{bmatrix} 
-1 & -2 & -1 \\ 
0 & 0 & 0 \\ 
1 & 2 & 1 
\end{bmatrix}
$$

### Příklad výpočtu
Pro výřez obrazu $A$, kde je vlevo lokalizována tmavší oblast (hodnota 10) a vpravo oblast světlejší (hodnota 200) – tedy svislá hrana:

$$A = \begin{bmatrix} 
10 & 10 & 200 \\ 
10 & 10 & 200 \\ 
10 & 10 & 200 
\end{bmatrix}$$

Aplikace masky $G_x$ (součet součinů):
$$(-1\cdot10) + (0\cdot10) + (1\cdot200) + (-2\cdot10) + (0\cdot10) + (2\cdot200) + (-1\cdot10) + (0\cdot10) + (1\cdot200)$$
$$= -10 + 0 + 200 - 20 + 0 + 400 - 10 + 0 + 200 = \mathbf{760}$$

Získaná vysoká hodnota 760 indikuje přítomnost silné hrany.

### Výpočet vlastností hrany
Hodnoty $G_x$ a $G_y$ jsou následně využívány pro výpočet celkové síly hrany a jejího směru.

1.  **Velikost hrany (Magnituda):**
    $$G = \sqrt{G_x^2 + G_y^2}$$
2.  **Úhel gradientu:**
    $$\theta = \arctan\left(\frac{G_y}{G_x}\right)$$

---

## 4. Cannyho detektor

Cannyho detektor je vícestupňový algoritmus navržený pro precizní lokalizaci hran. Ve srovnání s prostým konvolučním výpočtem je jím dosahováno vyšší odolnosti vůči šumu.

### Krok 1: Redukce šumu
Pro odstranění šumu je na vstupní obraz aplikován Gaussův filtr.

### Krok 2: Výpočet gradientu
Pomocí konvolučních operátorů (typicky Sobelova) je získána magnituda $G$ a směr gradientu $\theta$.

### Krok 3: Potlačení nemaxim (Non-maximum suppression)
Nalezené hrany jsou v této fázi zbytečně tlusté. Algoritmus proto projde šířku každé hrany (k tomu využívá velikost a směr gradientu z předchozího kroku) a nechá jen ten absolutně nejsilnější bod uprostřed - zbytek odstraní. Výsledkem jsou hrany o tloušťce jednoho pixelu.

### Krok 4: Hystereze (Dvojité prahování)
Pro finální klasifikaci hran jsou definovány dva prahy ($T_{min}$ a $T_{max}$):
1.  **Silné hrany ($> T_{max}$):** Jsou okamžitě klasifikovány jako hrany.
2.  **Slabé hrany (mezi $T_{min}$ a $T_{max}$):** Jsou klasifikovány jako hrany výhradně za předpokladu, že je detekováno jejich prostorové spojení se silnou hranou.
3.  **Šum ($< T_{min}$):** Je z výpočtu vyřazen.

---

## 5. Konvoluční neuronové sítě (CNN)

V architekturách konvolučních neuronových sítí nejsou konvoluční masky definovány staticky, nýbrž jsou dynamicky optimalizovány.

* **Učení filtrů:** Váhy v maskách jsou inicializovány náhodně. Během trénovacího procesu jsou tyto hodnoty výpočtem adaptovány tak, aby byla zajištěna detekce požadovaných obrazových příznaků.
* **První vrstvy:** Při analýze struktur natrénovaných sítí je zjištěno, že filtry v počátečních vrstvách vykazují značnou podobnost se standardními detektory hran (včetně Sobelových operátorů). Detekce hran je tímto validována jako optimální výchozí krok pro hlubokou analýzu obrazu.

---

## 6. Filtrace a redukce šumu

Než začneme hledat hrany, je velmi často vhodné obraz vyčistit od šumu, aby program nedetekoval falešné hrany. 
Pro každý typ filtrace nabízí knihovna OpenCV přímo konkrétní funkci.

### Rozostření (blur)

Tato metoda zprůměruje hodnoty všech sousedních pixelů v dané oblasti, což zajistí základní rozostření obrazu. 
Je to rychlý postup, ale může hrany nechtěně příliš rozmazat. 
Příslušná OpenCV funkce`blur()`.

### Gaussovský filtr (GaussianBlur)

Tento filtr počítá vážený průměr okolních pixelů, přičemž body blíže ke středu mají na výsledek větší vliv. 
Tím vzniká přirozenější rozostření, které je shopno do určité míry potlačit drobný šum a detaily. 
Příslušná OpenCV funkce je `GaussianBlur()`.

### Mediánový filtr (medianBlur)

Místo průměrování tento filtr seřadí hodnoty okolních pixelů a vybere z nich prostřední hodnotu (medián). 
S tímto filtrem je tak možné odstranit izolované hodnoty (šum typu "sůl a pepř"). 
Příslušná OpenCV funkce `medianBlur()`.

---

## 7. Praktická ukázka (Python & OpenCV)

Níže je uvedena ukázka vybraných metod s využitím knihovny OpenCV.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Načtení obrazu ve stupních šedi
img = cv2.imread('vstup.jpg', 0)

if img is None:
    print("Chyba: Obraz nebyl detekován.")
else:
    # 2. Filtrace (Gaussovský filtr s maskou 5x5)
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)

    # 3. Sobelova detekce hran
    # Ruční definice matic (jader) pro Sobelův operátor
    kernel_x = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]], dtype=np.float32)

    kernel_y = np.array([[-1, -2, -1],
                         [ 0,  0,  0],
                         [ 1,  2,  1]], dtype=np.float32)

    # Použití funkce filter2D s ručně definovanými maticemi
    # Datový typ cv2.CV_64F je využit pro zachování záporných hodnot derivací
    # (Alternativně lze pro tento výpočet použít přímo funkci cv2.Sobel v OpenCV)
    sobel_x = cv2.filter2D(img_blur, cv2.CV_64F, kernel_x)
    sobel_y = cv2.filter2D(img_blur, cv2.CV_64F, kernel_y)

    # Výpočet magnitudy hrany
    sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    # 4. Cannyho detektor (všechny kroky v rámci jedné funkce)
    # Prahové hodnoty jsou nastaveny na 100 a 200
    edges_canny = cv2.Canny(img, 100, 200)

    # Vizualizace
    plt.figure(figsize=(10,5))
    plt.subplot(121), plt.imshow(sobel_magnitude, cmap='gray'), plt.title('Sobel (Magnituda)')
    plt.subplot(122), plt.imshow(edges_canny, cmap='gray'), plt.title('Canny')
    plt.show()
