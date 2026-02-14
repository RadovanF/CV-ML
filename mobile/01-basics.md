
---

# Anatomie webu a role Ionicu

> **Cílem této části je stručný popis základnů HTML, CSS, JS a vysvětlení mechanismu, jakým do tohoto prostředí vstupuje knihovna Ionic.**

---

## 1. Tři vrstvy webu

Webový prohlížeč funguje jako renderovací engine, který zpracovává tři typy vstupů. Každý má svou definovanou roli:

1. **HTML (Structure)**: Definuje sémantiku a hierarchii prvků v DOM (Document Object Model).
2. **CSS (Presentation)**: Definuje vizuální vlastnosti prvků (barvy, rozměry, pozice).
3. **JavaScript (Behavior)**: Definuje logiku a interakci. Umožňuje manipulaci s DOMem v reálném čase.

### Rozdíl v přístupu: Native HTML vs. Ionic

Při použití čistého HTML/CSS je vývojář zodpovědný za definici všech vizuálních stavů prvku. Při použití Ionicu se vkládají tzv. **Web Components** (vlastní HTML značky), které mají styly a chování zapouzdřené uvnitř.

| Vrstva | Čisté HTML/CSS | Ionic (Web Components) |
| --- | --- | --- |
| **Vykreslení** | Prohlížeč vykreslí prvek podle CSS definovaného vývojářem. | Prohlížeč vykreslí interní strukturu definovanou knihovnou. |
| **Styling** | Nutná manuální definice tříd (classes). | Styly jsou načteny automaticky podle platformy (iOS/Android). |

---

## Příklad 1: Standardní HTML, CSS a JS


1. **První tlačítko** nemá žádné CSS třídy. Prohlížeč jej vykreslí s výchozím vzhledem operačního systému.
2. **Druhé tlačítko** má přiřazenou třídu `.custom-button`. Všechny vizuální vlastnosti (barva, rámeček, padding) musely být explicitně definovány v bloku `<style>`.
3. **Interakce** je řešena pomocí nativní funkce `window.alert()`. Tato funkce zastaví běh celého vlákna prohlížeče, dokud uživatel okno nezavře.

```html
<!DOCTYPE html>
<html lang="cs">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>1. Native HTML</title>

  <style>
    /* CSS definice pro vlastní tlačítko */
    .custom-button {
      background-color: #50c8ff;
      color: #000;
      padding: 10px 20px;
      border: 1px solid #000;
      border-radius: 4px;
      cursor: pointer;
      margin-top: 10px;
      font-weight: bold;
    }
    
    /* Definice stavu při kliknutí */
    .custom-button:active {
      background-color: #3aa0cc;
    }
  </style>
</head>
<body>

  <div style="padding: 20px; font-family: sans-serif;">
    <h2>1. Standardní HTML a CSS</h2>
    
    <p>Tlačítko bez CSS tříd:</p>
    <button>HTML Button</button>

    <hr style="margin: 20px 0;">

    <p>Tlačítko s definovaným CSS a onclick JS eventem:</p>
    
    <button class="custom-button" onclick="alert('Toto je nativní alert.')">
      Styled Button
    </button>
    
  </div>

</body>
</html>

```

---

## Příklad 2: Ionic Framework

V tomto příkladu je použita knihovna Ionic.

1. **Vzhled**: Nejsou definovány žádné vlastní CSS styly. Prvky `<ion-button>` a `<ion-alert>` přebírají vzhled z knihovny Ionic (soubor `ionic.bundle.css`).
2. **Struktura**: Aplikace je obalena do tagu `<ion-app>`, který zajišťuje správné vrstvení elementů.

```html
<!DOCTYPE html>
<html lang="cs">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>2. Ionic Framework</title>
  
  <script type="module" src="https://cdn.jsdelivr.net/npm/@ionic/core/dist/ionic/ionic.esm.js"></script>
  <script nomodule src="https://cdn.jsdelivr.net/npm/@ionic/core/dist/ionic/ionic.js"></script>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@ionic/core/css/ionic.bundle.css"/>
</head>
<body>

  <ion-app>
    <ion-content class="ion-padding">
      
      <h2>2. Ionic Komponenty</h2>
      <p>Ukázka použití vlastnosti trigger a předdefinovaných komponent.</p>
      
      <ion-button id="alert-trigger" expand="block">
        Otevřít Ionic Alert
      </ion-button>

      <ion-alert
        trigger="alert-trigger"
        header="Informační okno"
        sub-header="Komponenta ion-alert"
        message="Tady bude vaše zpráva."
      ></ion-alert>

    </ion-content>
  </ion-app>

  <script>
    // Konfigurace tlačítek v alertu
    // Vyhledáme element <ion-alert> v DOMu
    const alert = document.querySelector('ion-alert');
    
    // Přiřadíme pole řetězců do vlastnosti buttons
    alert.buttons = ['OK', 'Zavřít'];
  </script>

</body>
</html>

```

---

## Příklad 3: Vstupní pole (Input)

**HTML kód:**
```html
<div class="form-group">
  <label for="name">Jméno:</label>
  <input type="text" id="name" placeholder="Jan Novák">
</div>
```

**Ionic kód:**
```html
<ion-input
  label="Jméno"
  label-placement="floating"
  fill="outline"
  placeholder="Jan Novák"
></ion-input>
```

---

## Jak ladit Ionic aplikace (Debugging)

### 1. Ladění v prohlížeči (Browser Debugging)
Při vývoji mobilních aplikací v prohlížeči využíváme **Vývojářské nástroje (DevTools)**. Otevřete je klávesou `F12` nebo pravým kliknutím -> *Prozkoumat (Inspect)*.

*   **Režim zařízení (Device Mode)**: Mobilní aplikace by se měla testovat v rozměrech mobilního telefonu. V DevTools klikněte na ikonu tabletu/telefonu a vyberte simulované zařízení.

### 2. Ladění na reálném zařízení (USB Inspector)
Testování na reálném HW.

1.  **Povolit vývojářské možnosti na telefonu**:
    - *Nastavení > Informace o telefonu > Číslo sestavení* (7x poklepat).
    - V novém menu *Vývojářské možnosti* povolit **Ladění USB**.
2.  **Propojení**: Připojte telefon k PC přes USB kabel.
3.  **Chrome Inspector**:
    - Do adresního řádku v Chrome na PC napište: `chrome://inspect/#devices`.
    - Uvidíte připojené zařízení a seznam otevřených tabů.

### 3. Další nástroje pro vývoj a testování
Kromě standardních nástrojů existují i specializovaná řešení:

*   **Monaca**: Cloudové IDE a sada nástrojů, která umožňuje vyvíjet a ladit Ionic aplikace v cloudu bez konfigurace lokálního prostředí. Nabízí vlastní aplikaci pro live-preview na telefonu.
*   **WebNative**: Rozšíření pro VS Code (WebNative VS Code Extension), které integruje nástroje pro Ionic a Capacitor přímo do editoru. Pomáhá s generováním ikon, spouštěním na zařízeních a laděním.
