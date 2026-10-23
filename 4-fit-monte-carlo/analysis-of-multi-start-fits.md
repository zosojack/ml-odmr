# Benchmark del Fit Tradizionale Multi-Start Monte Carlo: Metodologia e Analisi Dati

**Paper di riferimento:**

[A Deep-Learning-Boosted Framework for Quantum Sensing with Nitrogen-Vacancy Centers in Diamond | **Yao et al.**](https://arxiv.org/abs/2603.14728v1)

---

## 1. Obiettivo dell'Analisi

Lo scopo di questo modulo di analisi è quantificare le prestazioni, i limiti di stabilità e il costo computazionale del **fit convenzionale ai minimi quadrati non lineari** (algoritmo di Levenberg-Marquardt con approccio multi-start Monte Carlo) operante su spettri ODMR sintetici realistici di un cristallo di diamante macroscopico (*bulk*).

I risultati aggregati da questo benchmark costituiscono la linea di base (*baseline*) quantitativa contro cui confrontare:

1. L'inferenza pura della rete neurale convoluzionale **1D-CNN**;
2. Il **metodo ibrido** (Levenberg-Marquardt inizializzato deterministicamente dalle predizioni della rete).

---

## 2. Definizione del Rapporto Segnale-Rumore (SNR)

### Formulazione Teorica

In spettroscopia ODMR basata su photon counting, il rumore dominante è lo **shot noise poissoniano** associato al flusso di fotoni di fotoluminescenza raccolti. Lontano dalle risonanze, il livello di emissione di base (fondo) su ciascuno dei $K = 167$ bin di frequenza è dato in media da:

$$N_0 = \frac{N_{\text{ph}}}{K}$$

dove $N_{\text{ph}}$ è il budget complessivo di fotoni integrati nell'intera finestra di sweep.

In corrispondenza del minimo di risonanza con contrasto ottico $C$, il segnale utile corrisponde alla profondità del calo di emissione:

$$\Delta S = C \cdot N_0 = C \frac{N_{\text{ph}}}{K}$$

Poiché per una distribuzione di Poisson la varianza coincide con la media ($\sigma^2 = N_0$), l'incertezza statistica su ogni singolo canale vale $\sigma = \sqrt{N_0}$. L'**SNR di canale** è dunque definito analiticamente come:

$$\text{SNR} = \frac{\Delta S}{\sigma} = \frac{C \cdot N_0}{\sqrt{N_0}} = C \sqrt{N_0} = C \sqrt{\frac{N_{\text{ph}}}{K}}$$

### Scelta Metodologica e Confronto con Yao et al.

* **Nel nostro approccio:** Non si ricorre a stime empiriche rumorose dell'SNR calcolate a posteriori sui singoli vettori sintetici. Poiché ciascuna campagna di fit risiede in una specifica combinazione controllata $(C, N_{\text{ph}})$, l'SNR viene calcolato in modo **analitico e deterministico**, fungendo da asse continuo e rigoroso per le curve di benchmark.
* **Analogia con Yao et al.:** Yao et al. parametrizzano l'accuratezza in funzione dell'SNR e del contrasto (si vedano le Fig. 2 e Fig. 4 del paper). La dipendenza da $C \sqrt{N_{\text{ph}}}$ rispecchia esattamente la relazione limite fondamentale di sensibilità shot-noise limitata nel sensing quantistico con centri NV:
$$\eta \propto \frac{1}{C \sqrt{R}}$$


dove $R$ è il tasso di conteggio di fotoni al secondo.

---

## 3. Metriche di Accuratezza e Definizione del Success Rate Fisico

I parametri estratti dal fit Levenberg-Marquardt nello spazio normalizzato ($c_{\text{fit}}, s_{\text{fit}} \in [0, 1]$) vengono confrontati direttamente con la ground truth del cristallo ($c_{\text{true}}, s_{\text{true}}$) memorizzata nel dataset.

### 1. Errori Assoluti e RMS

Gli scarti su scala normalizzata vengono convertiti nelle unità fisiche di laboratorio attraverso l'estensione effettiva dello sweep ($\Delta\nu_{\text{sweep}} = 200\text{ MHz}$):

$$\Delta c_{\text{MHz}} = \vert{}c_{\text{fit}} - c_{\text{true}}\vert{} \cdot \Delta\nu_{\text{sweep}}$$

$$\Delta s_{\text{MHz}} = \vert{}s_{\text{fit}} - s_{\text{true}}\vert{} \cdot \Delta\nu_{\text{sweep}}$$

Per aggregare le discrepanze sull'insieme degli spettri validi si calcolano sia il **MAE** (*Mean Absolute Error*) sia l'**RMSE** (*Root Mean Square Error*):

$$\text{RMSE}_s = \sqrt{\frac{1}{N_{\text{conv}}} \sum_{i=1}^{N_{\text{conv}}} (\Delta s_{\text{MHz}, i})^2}$$

L'RMSE risulta particolarmente sensibile alla presenza di fit che convergono in minimi locali errati, penalizzando severamente i fallimenti macroscopici dell'algoritmo.

### 2. Success Rate Fisico ($\vert{}\Delta s\vert{} < 1.0\text{ MHz}$)

* **Limite del success rate puramente numerico:** Le librerie numeriche (come `scipy.optimize.curve_fit`) restituiscono `converged = True` ogniqualvolta l'algoritmo si arresta soddisfacendo le tolleranze di gradiente (`gtol` o `ftol`). A basso SNR, tuttavia, l'algoritmo converge regolarmente in buche di shot noise casuali, producendo una convergenza matematica che corrisponde in realtà a un fallimento fisico totale.
* **Definizione della soglia fisica:** Viene introdotto il **Success Rate Fisico**, definito come la percentuale di spettri per cui il fit individua lo splitting reale con un errore assoluto inferiore a una soglia di tolleranza sperimentale:
$$\text{Success Rate} = \frac{1}{N_{\text{tot}}} \sum_{i=1}^{N_{\text{tot}}} \mathbb{I}\left( \Delta s_{\text{MHz}, i} < 1.0\text{ MHz} \right) \times 100$$


* **Confronto con Yao et al.:** Nel paper di Yao et al. il successo del fit viene valutato in funzione dell'accuratezza della termometria e della magnetometria; stime che deviano dai dip reali oltre la larghezza intrinseca di riga non consentono l'inversione fisica. La soglia di $1.0\text{ MHz}$ qui adottata corrisponde a una frazione minoritaria della larghezza di riga omogenea/disomogenea ($\Gamma \approx 4 - 8\text{ MHz}$), garantendo che solo i fit capaci di discriminare le due risonanze reali siano classificati come corretti.

---

## 4. Architettura dell'Analisi nel Codice

Il codice di analisi implementa una pipeline modulare a 4 stadi:

```
[ Cartella ../benchmarks/10k/ ]
              │
              ▼
 1. Scansione Ricorsiva dei File CSV
    ├── Parsing parametri da cartelle e nomi file: (C, M, N_ph)
    └── Calcolo analitico: SNR = C * sqrt(N_ph / 167)
              │
              ▼
 2. Estrazione e Normalizzazione Dati (Pandas)
    ├── Filtraggio divergenze e non-convergenze
    ├── Calcolo RMSE ed errore medio (in MHz fisici)
    └── Calcolo del tasso di successo fisico (|Δs| < 1.0 MHz)
              │
              ▼
 3. Aggregazione Statistica
    └── Raggruppamento in quantili/bin di SNR per ciascun valore di M
              │
              ▼
 4. Generazione Dashboard Diagnostica (4 Pannelli)

```

### Dettaglio dei Pannelli di Benchmark:

1. **RMSE Splitting $s$ vs SNR (Scala Log-Log):**
Dimostra l'andamento asintotico dell'errore al variare del rapporto segnale-rumore. Mostra la divergenza ad alto rumore (basso SNR) e il raggiungimento del limite statistico a elevati budget di fotoni.
2. **Tasso di Successo Fisico vs SNR:**
Mostra la transizione di fase tipica del fit non lineare: al di sotto di un SNR critico, il tasso di successo crolla rapidamente verso lo $0\%$. Aumentare i tentativi Monte Carlo $M$ sposta questa curva di transizione verso sinistra (verso regimi di rumore più severi).
3. **Impatto del Parametro Multi-Start ($M$) a Contrasto Fissato:**
Quantifica la legge dei rendimenti decrescenti (*diminishing returns*). Il passaggio da $M=1$ a $M=5$ e $M=10$ produce un abbattimento esponenziale dell'errore, mentre il guadagno tra $M=20$ ed $M=40$ si riduce a fronte di un raddoppio del tempo macchina.
4. **Costo Computazionale (Tempo Medio per Spettro in ms vs $M$):**
Quantifica la scalabilità lineare del tempo di calcolo all'aumentare di $M$ su architettura multi-core.

---

## 5. Sintesi Comparativa: Nostro Framework vs Yao et al.

| Aspetto Metodologico | Metodologia Yao et al. | Nostro Approccio |
| --- | --- | --- |
| **Fisica del Modello** | Fenomenologico a 2 lorentziane indipendenti | Cristallo bulk di diamante con simmetria 227 (EasySpin) |
| **Finestra di Sweep** | $[2.80, 2.92]\text{ GHz}$ ($120\text{ MHz}$, 101 punti) | $[2.75, 2.95]\text{ GHz}$ ($200\text{ MHz}$, 167 punti) |
| **Calcolo dell'SNR** | Derivato da contrasto e fotoni di picco per canale | Analitico esatto: $\text{SNR} = C \sqrt{N_{\text{ph}} / K}$ |
| **Inizializzazione Fit** | Multi-start Monte Carlo uniforme ($M$ elevato, fino a 200) | Multi-start Monte Carlo vincolato nello spazio normalizzato Z-score |
| **Criterio di Convergenza** | Soglia di consistenza su $\Delta T$ / residui | Success Rate Fisico su splitting reale ($\vert{}\Delta s\vert{} < 1.0\text{ MHz}$) |
| **Valutazione Prestazioni** | Runtime wall-clock su 5.000 spettri (CPU vs GPU) | Runtime per spettro e tempo aggregato su $10.000$ campioni |