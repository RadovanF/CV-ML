# Úvod k jazykovým modelům + cesta k Vision Transformerům

**(PRACOVNÍ VERZE v1)**

1. **Část I: Nadhled** – Intuitivní pochopení principů + ukázka hotových modelů pomocí knihoven např. `gensim` a `transformers`.
2. **Část II: Podrobnější informace** –  Funkčnost jednotlivých modulů na úrovni tenzorových operací (v knihovně `PyTorch`).

---

# Část I

## 1. Základy převodu textu na vektory
Počítačové systémy zpracovávají textová data prostřednictvím numerických hodnot. Převod textu na matematické reprezentace probíhá ve dvou krocích:
1. **Tokenizace:** Vstupní text se rozdělí na menší celky, zvané **tokeny** (slova, podslova či znaky). Každému unikátnímu tokenu z definovaného slovníku je přiřazen jedinečný celočíselný index. Tyto indexy samy o sobě nenesou žádný sémantický význam (např. slova s podobným významem jako "pes" a "vlk" mohou mít velmi vzdálené indexy).
2. **Embeddings (Slovní vnoření):** Numerické indexy jsou promítnuty do vícerozměrného sémantického prostoru jako **vektory** (reálná čísla v definované dimenzi, např. 300 nebo 512). Tyto vektory fungují jako souřadnice v prostoru významu. Slova s podobným významem nebo výskytem v podobných kontextech jsou v tomto prostoru umístěna blízko sebe.

## 2. Statická slovní vnoření: Word2Vec a GloVe
První generace moderních sémantických modelů (kolem let 2013–2014) pracovala se **statickou reprezentací**. Každému slovu ze slovníku byl přiřazen právě jeden pevný vektor bez ohledu na větný kontext. Slovo s více významy (např. "koruna") má v tomto schématu pouze jedinou průměrnou reprezentaci.

### 2.1 Word2Vec
Mikolov et al. (Google, 2013) navrhli modely, kde vektory představují přímo váhy uvnitř neuronové sítě. Tato síť má jednu vnitřní (skrytou) vrstvu. Během tréninku se učí hádat okolní slova, a jakmile trénink skončí, váhy z této vrstvy se uloží. Tím vzniknou pevné (statické) vektory. Model čte text jako posuvné okno a zkouší uhádnout skryté slovo podle sousedů a k nalezení výsledku používá kosinovou podobnost.

Díky lineární struktuře vykazují výsledné vektory schopnost reprezentovat vztahy pomocí aditivních a subtraktivních operací (např. $\text{King} - \text{Man} + \text{Woman} \approx \text{Queen}$).

### 2.2 GloVe (Global Vectors)
Pennington et al. (Stanford, 2014) vytvořili model GloVe, který zjišťuje význam slov tak, že si nejprve vytvoří obrovskou statistickou tabulku ze všech dostupných textů. V ní je přesně spočítáno, kolikrát se každé slovo objevilo v blízkosti jakéhokoliv jiného slova. Místo postupného hádání z okolních slov (jako to dělá Word2Vec) hledá logiku v poměrech těchto výskytů. Porovnává vždy dvě slova vůči třetímu. Z této globální statistické tabulky se následně pomocí matematických operací rovnou vygenerují konečné vektory slov (tzv. embedding), aniž by se musela trénovat složitá neuronová síť.

### Ukázka modelů Word2Vec/GloVe přes Gensim
Knihovna `gensim` nabízí rozhraní pro práci s předtrénovanými statickými vektory.

```python
import gensim.downloader as api

# https://platform.openai.com/tokenizer

# https://tiktokenizer.vercel.app/

# předtrénovaný model Word2Vec.
word2vec_model = api.load("word2vec-google-news-300")

# předtrénovaný model GloVe.
#glove_model = api.load("glove-wiki-gigaword-100")

# nejpodobnější slova pomocí Word2Vec.
print("Word2Vec podobnost pro 'dog':")
print(word2vec_model.most_similar("dog", topn=3))

# rovnice (král - muž + žena) v modelu word2vec_model.
result = word2vec_model.most_similar(
    positive=["king", "woman"], negative=["man"], topn=1
)
print("\nword2vec_model výpočet (king - man + woman):")
print(result)

# ukázka vektoru pro konkrétní slovo glove_model.
#vector = glove_model["cat"]
#print("\nDélka vektoru pro 'cat' glove_model:", len(vector))

# ukázka vektoru pro konkrétní slovo word2vec_model.
vector = word2vec_model["cat"]
print("\nDélka vektoru pro 'cat' word2vec_model:", len(vector))
```

---

## 3. Dynamické reprezentace: Architektura Transformer
V roce 2017 představili Vaswani et al. v práci *Attention Is All You Need* architekturu **Transformer** a mechanismus pozornosti (attention). Zatímco ve statických přístupech znají modely význam slov izolovaně, mechanismus pozornosti umožňuje pochopit větu jako propojený celek. Vektor slova se dynamicky mění podle okolního kontextu (např. slovo "zámek" má jiná čísla ve spojitosti s budovou a jiná s dveřmi). Výsledkem jsou **dynamické (kontextové) vektory**, které mění své souřadnice podle obsahu celé věty.

### Koncept Self-Attention (Q, K, V)
Při výpočtu se pro každý token generují tři lineární projekce, které reprezentují specifické role v procesu hledání souvislostí:
1. **Query (Matice Q - Dotaz):** Co daný token v sekvenci hledá.
2. **Key (Matice K - Klíč):** Jaké informace daný token nabízí ostatním.
3. **Value (Matice V - Hodnota):** Skutečný sémantický obsah tokenu, který se předává dál.

Matice Q se vynásobí s transponovanou maticí K. Výsledkem je matice skóre (vztahů) popisující, jak moc se tokeny zajímají o sebe navzájem.
Matice vztahů se vynásobí původními hodnotami V. Tokeny do sebe "nasají" hodnoty ostatních a výsledná čísla jsou obohacena o kontext.

---

## 4. Poziční kódování a  Token [CLS] 
Protože Transformer zpracovává celou větu paralelně (najednou), postrádá informaci o **pořadí tokenů**. K vyřešení tohoto omezení se může využít:
* **Absolutní poziční kódování:** K sémantickým vektorům se na vstupu přičtou poziční kódy generované pomocí sinusových a kosinusových funkcí s různými frekvencemi.
* **Relativní poziční kódování:** Informace o vzájemné vzdálenosti tokenů se přičítá jako korekční člen (bias) přímo při výpočtu skóre pozornosti.

* **Token [CLS]:** Speciální token vložený na začátek každé sekvence. Vzhledem k tomu, že prochází všemi vrstvami pozornosti, jeho výsledný vektor reprezentuje agregovaný sémantický kontext celé věty. Jinými slovy: pro finální klasifikaci je potřeba jeden vektor zastupující celý obrázek nebo text. Uměle se přidá na úplný začátek, funguje jako "houba" a během výpočtu nasává informace z ostatních tokenů.

---

## 5. Model BERT
Model **BERT** (Devlin et al., Google, 2018) využívá architekturu z Transformeru pro získání hlubokých obousměrných reprezentací.
Model slouží k porozumění napsanému textu. Čte celou větu obousměrně (zleva doprava i zprava doleva najednou) pro získání kompletního kontextu.
V textu se zakryjí náhodná slova a model se učí je z kontextu uhádnout.
*   **Využití:** Třídění dokumentů, klasifikace textu, přesné vyhledávání.

### Ukázka klasifikace textu pomocí BERT
```python
from transformers import pipeline

# 1. Inicializace pipeline pro klasifikaci textu (Text Classification)
# BERT model předtrénovaný pro sentimentální analýzu v češtině.
classifier = pipeline("text-classification", model="Kath997/FERNET-C5-RoBERTa-finetuned-sentiment")

# 2. Definování vstupního textu k analýze (např. recenze produktu)
text = "Tato nová sluchátka mají naprosto špičkový zvuk."
result = classifier(text)[0]

# 3. Výsledek klasifikace
print(f"Vstupní text: '{text}'")
print(f"Klasifikace: '{result['label']}' (pravděpodobnost: {result['score']:.4f})")
```

---

## 6. Model GPT-2
Model **GPT-2** (Radford et al., OpenAI, 2019) využívá také architekturu z Transformeru.
Na rozdíl od modelu BERT je GPT-2 sestaven tak, aby na základě přečteného začátku vymýšlel jeho další pokračování.
Čte text jednosměrně zleva doprava.
Trénuje se neustálým hádáním, jaké jediné slovo má logicky následovat. 
Na výstupu sítě získá každý token procentuální šanci a vybere se slovo s nejvyšší pravděpodobností.
*   **Využití:** Psaní článků, konverzační boti nebo překlady.

### Ukázka generování textu pomocí GPT-2
```python
from transformers import pipeline

# 1. Inicializace pipeline pro generování textu s modelem GPT-2
generator = pipeline("text-generation", model="gpt2")

# 2. Spuštění generování na základě počátečního kontextu (promptu)
prompt = "The future of language models lies in"
result = generator(prompt, max_length=30, num_return_sequences=1)

# 3. Výpis vygenerovaného textu
print("Vygenerovaný text z modelu GPT-2:")
print(result[0]['generated_text'])


# 4. Ukázka pro generování textu s českým modelem.
generator = pipeline("text-generation", model="spital/gpt2-small-czech-cs")

prompt = "Budoucnost umělé inteligence je"

result = generator(prompt, max_new_tokens=30, num_return_sequences=1)

print(result[0]['generated_text'])
```

## 7. Aplikace v počítačovém vidění: Vision Transformer (ViT)
Dosovitskiy et al. (Google Brain, 2020) ukázali, že standardní Transformer lze aplikovat přímo na obrazová data (podobně jako model BERT). Model **Vision Transformer (ViT)** zpracovává obrazové fragmenty jako slova v textové sekvenci následujícím způsobem:
1. **Rozdělení obrazu na části (patche):** Obrázek je rozdělen na mřížku 2D "patchů" (např. 16x16 pixelů), které plní funkci sémantických tokenů (slov).
2. **Lineární projekce:** Každý patch je zploštěn a lineárně promítnut do dimenze Transformeru.
3. **Poziční vnoření a token [CLS]:** K patchům se přičtou naučitelná 1D poziční vnoření a na začátek sekvence se vloží speciální token **[CLS]** pro klasifikační hlavu.

### ViT vs. CNN (např. VGG)
Tradiční konvoluční sítě (CNN) vidí obraz spíše lokálně (učí se lokální vazby) díky konvolučním jádrům. ViT naproti tomu **nemá téměř žádné lokální apriorní předpoklady (předsudky)** o 2D obrazu a prostorové uspořádání se učí zcela od nuly z velkého množství dat.

### Ukázka klasifikace obrazu pomocí ViT
```python
from transformers import pipeline
import requests

# Modelem ViT Small.
image_classifier = pipeline(model="timm/vit_small_patch16_224.augreg_in21k_ft_in1k")

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/timm/cat.jpg"

# Klasifikace
outputs = image_classifier(url)

# Výsledky.
for output in outputs:
    print(f"Label: {output['label'] :20} Score: {output['score'] :0.2f}")   
```
