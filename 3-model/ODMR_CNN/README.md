# CNN finale per spettri ODMR

Progetto Python per utilizzare e studiare la **CNN B finale** del progetto ODMR. Contiene l'architettura TensorFlow/Keras, la loss usata, il caricamento dei pesi appresi, la preparazione degli input e la conversione delle uscite nelle grandezze fisiche.

La rete stima **soltanto due parametri: `c_fit` e `s_fit`**. Il collega può importare la CNN nel proprio programma e utilizzare liberamente le sue previsioni come ingresso alla propria procedura. Qui il calcolo termina con le previsioni della CNN.

## 1. Prima di iniziare: sorgente e pesi sono due cose diverse

Il **sorgente** descrive come è fatta la rete. I **pesi** sono i numeri imparati durante l'addestramento. Per usare il modello già addestrato servono entrambi. Chiamare `build_cnn()` da solo costruisce una rete con pesi casuali; per fare previsioni usare `CNNPredictor`, che richiede i veri checkpoint.

Nel download del progetto ci sono codice e guida. I pesi originali si trovano sul PC di Antony, nella cartella completa `risultati_grid27_cfit_24h`. Lo script `export_weights.py` li raccoglie senza riaddestrare la rete. Se Antony ha già eseguito l'esportazione, il collega riceverà il progetto con la cartella `artifacts/cnn_50k_48M` già presente.

## 2. Quale CNN stiamo consegnando

| Impostazione | Valore finale |
|---|---|
| Famiglia del modello | CNN B residuale, 1D, TensorFlow/Keras |
| Dataset di riferimento | 50.000 spettri simulati EasySpin |
| Spettri effettivi di training | 40.000 |
| Validation / test | 5.000 / 5.000, disgiunti dal training |
| Griglia | 335 punti tra 2750 e 2950 MHz |
| Passo | Circa 0,5988 MHz |
| Fotoni | 48 milioni totali attesi per spettro |
| Contrasto sintetico | Uniforme tra 0,012 e 0,15 |
| Input | Fluorescenza min-max, forma `(N,335,1)` |
| Target | `c_fit`, `s_fit`, sulla finestra effettiva da 200 MHz |
| Numero di parametri | 806.946 per rete |
| Dropout effettivo | 0 |
| Loss | MSE pesata, peso 10 sul centro nello spazio q |
| Seed dei modelli | 42, 137, 2026 |
| Previsione finale di riferimento | Media delle previsioni dei tre modelli |

I tre seed corrispondono a **tre addestramenti della stessa architettura**, con inizializzazioni differenti. Ogni rete restituisce due numeri. La media delle tre previsioni è chiamata *ensemble*. Il modello consegnato è quello addestrato sugli spettri rumorosi a 48M; l'addestramento sul pulito era una diagnostica separata.

## 3. Antony: recupera i pesi già addestrati

Estrai questo progetto in:

```text
D:\Da disco locale\Desktop\ODMR\ODMR_CNN_GitHub
```

Dal terminale di VS Code aperto nella cartella `D:\Da disco locale\Desktop\ODMR`, dove esiste già `.venv`, esegui:

```powershell
.\.venv\Scripts\python.exe .\ODMR_CNN_GitHub\export_weights.py --reference ".\risultati_grid27_cfit_24h"
```

`--reference` deve indicare la cartella completa dei risultati, con `config.json`, `jobs` e `predictions`. Puoi omettere l'argomento per scegliere la cartella da una finestra.

Lo script recupera i tre checkpoint finali, anche quando sono nei backup `checkpoint_*.zip`, e crea:

- `ODMR_CNN_GitHub/artifacts/cnn_50k_48M`: pesi e trasformazioni dei target.
- Un nuovo `ODMR_CNN_con_pesi_<data_ora>.zip` nella cartella che contiene il progetto: codice, guida e pesi, pronti da passare al collega.

Ripetere il comando con gli stessi pesi già esportati è consentito: vengono verificati e riutilizzati. I risultati originali non vengono modificati. Se hai soltanto uno ZIP di analisi, devi recuperare la cartella originale con i checkpoint: gli archivi di analisi non contengono i pesi appresi.

## 4. Collega: installazione ed esecuzione

Apri in VS Code la cartella che contiene questo `README.md`. In **Terminal > New Terminal** crea un ambiente Python 3.12 e installa le dipendenze:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Se Python 3.12 non è installato, occorre prima installarlo. Su un PC dove l'ambiente del progetto funziona già puoi usare quell'interprete. Per questa inferenza la CPU è sufficiente; `GPU: []` non blocca il programma.

Per provare 20 spettri del MAT originale:

```powershell
.\.venv\Scripts\python.exe predict.py --mat "D:\Dati\odmr_easyspin_raw_50000_DOPPIA_DENSITÀ (1).mat" --samples 20
```

Sostituisci il percorso di esempio con quello reale. Antony deve passarti il MAT originale da 50k e 335 punti, con le label del fit pulito, se vuoi riprodurre questa dimostrazione. Il nome del file può cambiare; il contenuto deve coincidere con il riferimento.

Il programma usa per default i primi 20 esempi della **validation originale**, simula gli stessi conteggi a 48M e salva le sole previsioni CNN. Prima verifica 64 previsioni archiviate entro 1 kHz: è un controllo di compatibilità del trasferimento, non l'accuratezza della rete. Le cartelle di output hanno data e ora, quindi un nuovo run non sovrascrive quello precedente.

Per conteggi già osservati, prepara un file NPZ con due array:

```python
np.savez_compressed("conteggi.npz", counts=conteggi, freq_MHz=frequenze)
```

`conteggi` ha forma `(N,335)` e `frequenze` forma `(335,)`. Poi esegui:

```powershell
.\.venv\Scripts\python.exe predict.py --counts "conteggi.npz" --samples 100
```

Questo ramo **non aggiunge rumore** a conteggi già misurati. Il numero di campioni trattati è limitato da `--samples`, che per default è 20.

## 5. L'interfaccia da importare nel proprio progetto

Questo è il punto di collegamento per il codice del collega. Costruisci il predictor una volta e riutilizzalo per tutti gli spettri:

```python
import numpy as np
from cnn_model import CNNPredictor

cnn = CNNPredictor("artifacts/cnn_50k_48M")

# Conteggi osservati di fluorescenza: dip verso il basso.
# frequenze_MHz deve contenere la griglia 2750-2950 MHz, 335 punti.
dati = np.load("conteggi.npz")
pred = cnn.predict_counts(dati["counts"], dati["freq_MHz"])

c = pred["c"]                   # forma (N,), adimensionale
s = pred["s"]                   # forma (N,), separazione completa / 200 MHz
nu1 = pred["nu1_MHz"]           # forma (N,), frequenza inferiore prevista
nu2 = pred["nu2_MHz"]           # forma (N,), frequenza superiore prevista
centro = pred["center_MHz"]
splitting = pred["delta_MHz"]

# Da qui il collega utilizza c,s oppure nu1,nu2 nel PROPRIO codice.
```

`predict_counts` accetta anche un solo spettro di forma `(335,)`; le uscite conservano una dimensione campione, quindi il primo valore è `pred["c"][0]`.

Se il collega ha già applicato il min-max, può chiamare:

```python
pred = cnn.predict_normalized(X, frequenze_MHz)  # X: (N,335,1)
```

Per scegliere una sola rete invece della media:

```python
cnn = CNNPredictor("artifacts/cnn_50k_48M", seed=42)
```

In questo caso le prestazioni possono differire dall'ensemble. Il seed seleziona un checkpoint già addestrato; non introduce un nuovo addestramento.

## 6. Che cosa rappresentano c e s

Le label disponibili derivano da un fit sul segnale EasySpin pulito. Quel fit ha stimato due centri di componenti, con frequenze \(\nu_1<\nu_2\). Nelle formule seguenti tutte le frequenze sono in MHz:

$$c_{\mathrm{fit}}=\frac{(\nu_1+\nu_2)/2-2750}{200},\qquad
s_{\mathrm{fit}}=\frac{\nu_2-\nu_1}{200}.$$

Il centro è la posizione media delle due componenti. Lo splitting è la loro distanza completa. La normalizzazione divide per la finestra di 200 MHz, rendendo le label adimensionali.

Le conversioni inverse sono:

$$m=2750+200c,\qquad \delta=200s,$$

$$\nu_1=2750+200(c-s/2),\qquad \nu_2=2750+200(c+s/2).$$

**Attenzione al fattore due:** `s` è la distanza completa. Per esempio `c=0.55`, `s=0.105` significano centro 2860 MHz, splitting 21 MHz, frequenze 2849,5 e 2870,5 MHz.

La CNN approssima le label del fit pulito. Questi centri di componenti possono differire dai minimi del profilo totale quando le risonanze si sovrappongono. Non sono automaticamente D e B dell'Hamiltoniana. La qualità di tutte le label non è certificata: le metriche misurano l'accordo con il riferimento disponibile.

Il generatore mantiene lo splitting **nominale 20-24 MHz**. Lo splitting delle label fit nel dataset finale è circa **20-21,67 MHz**. Non allarghiamo artificialmente questo intervallo e non tagliamo il dataset. Le FWHM simulate sono già quelle corrette, **4,8-21,6 MHz**: nessuna larghezza viene raddoppiata nel codice.

## 7. Come prepariamo l'input

Lo spettro pulito EasySpin è un assorbimento positivo: le risonanze sono verso l'alto. L'input della CNN è la fluorescenza: le risonanze sono dip verso il basso.

Nel training e nel controllo sul MAT:

1. Il profilo di assorbimento è normalizzato al proprio massimo: \(y_i\).
2. Si campiona il contrasto \(C\) tra 0,012 e 0,15 e si costruisce \(I_i=1-Cy_i\).
3. Si calcola \(\lambda_i=N_{\mathrm{tot}}I_i/\sum_j I_j\), con \(N_{\mathrm{tot}}=48\,000\,000\).
4. Si campionano conteggi Poisson indipendenti per canale.
5. Per ogni spettro si applica \(X_i=(N_i-\min N)/(\max N-\min N)\).
6. Si aggiunge il canale finale: `(N,335,1)`.

48M è il numero totale **atteso** per tutto lo spettro, non per punto; la somma osservata fluttua. Il protocollo originale usa estrazioni annidate 6M + 18M + 24M: il controllo mantiene questi stessi numeri casuali. Le realizzazioni di rumore nel training cambiavano a ogni epoca.

Per conteggi osservati, il programma esegue direttamente i punti 5 e 6. Il min-max rimuove scala e offset affini per spettro, ma non elimina il rumore né rende tutte le condizioni sperimentali equivalenti. Non passare direttamente un assorbimento positivo a `predict_counts`: il verso delle risonanze sarebbe sbagliato.

## 8. Come è fatta la CNN

Una convoluzione 1D scorre lungo le frequenze e cerca motivi locali come pendenze, curvature e parti dei dip. I suoi filtri sono imparati durante l'addestramento.

| Stadio | Operazione | Forma escluso il batch |
|---|---|---|
| Input | Fluorescenza min-max | `(335,1)` |
| Ingresso | Conv1D, 32 filtri, kernel 13; LayerNorm; SiLU | `(335,32)` |
| Blocchi 1 e 2 | Residuali, 32 canali, kernel 9 | `(335,32)` |
| Riduzione | AveragePooling1D di fattore 2 | `(167,32)` |
| Blocchi 3 e 4 | Residuali, 64 canali, kernel 5 | `(167,64)` |
| Riduzione canali | Conv1D, 32 filtri, kernel 1; SiLU | `(167,32)` |
| Flatten | Conserva l'ordine delle posizioni | `(5344,)` |
| Dense | 128 unità SiLU; dropout effettivo 0 | `(128,)` |
| Dense | 64 unità SiLU | `(64,)` |
| Uscita | Dense lineare | `(2,)` |

Ogni blocco residuale contiene due convoluzioni, normalizzazioni e una somma con il suo ingresso. Quando serve, una convoluzione di kernel 1 adatta il numero dei canali della scorciatoia. LayerNormalization stabilizza le rappresentazioni interne. SiLU è l'attivazione liscia \(x\,\mathrm{sigmoid}(x)\). Flatten conserva l'informazione su dove si trovano le strutture nello spettro.

Nel sorgente il layer Dropout viene inizialmente costruito con 0,05 e subito impostato a 0, come nel codice storico. È un dettaglio mantenuto per rispettare la costruzione originale: **durante l'uso finale il dropout è zero**.

## 9. Le due uscite interne q e la loss del training

La rete emette due coordinate interne \(q_c,q_s\), ottenute tramite:

$$q=(\mathrm{target}-\mathrm{offset})/\mathrm{scale}.$$

I valori salvati sono:

```text
offset = [0.46012123978549435, 0.09999820744930771]
scale  = [0.18038035019926157, 0.008374244575435341]
```

Sono stati calcolati sul training minimo da 8k del confronto originale e mantenuti identici per tutti i suoi run. Non si ricalcolano sui nuovi spettri, sulla validation o sul test. L'interfaccia restituisce automaticamente `target = offset + scale*q`, poi applica le conversioni in MHz.

La loss usata è:

$$\mathcal L=\frac{10(q_c-\hat q_c)^2+(q_s-\hat q_s)^2}{11},$$

mediata sul batch. Il peso 10 agisce nello spazio q, con scale diverse per i due target: non significa semplicemente dieci volte lo stesso errore espresso in MHz.

Questa versione finale usa **MSE**, non NLL. Emette stime puntuali; non emette varianze, deviazioni standard o una distribuzione gaussiana. `cs_by_seed` permette di vedere le tre previsioni, ma la loro dispersione non è automaticamente un'incertezza calibrata.

La ricetta storica di addestramento comprende AdamW, learning rate iniziale 0,0003, weight decay 0,00001, clipnorm 1, batch 256, due osservazioni rumorose per profilo a ogni epoca, massimo 200 epoche, riduzioni del learning rate ed early stopping. Il checkpoint di ogni seed è scelto sulla validation. Questo progetto costruisce e carica la CNN già addestrata; non avvia il confronto storico di 27 addestramenti.

## 10. File prodotti e precisione di riferimento

`predict.py` salva:

- `cnn_predictions.csv`: una riga per spettro, con c, s, centro, splitting e due frequenze in MHz; nel controllo MAT anche label ed errori.
- `cnn_predictions.npz`: gli stessi risultati come array, conteggi osservati e frequenze, pronti da ricaricare in Python.
- `run_info.json`: versioni, seed selezionati, provenienza e controllo del riferimento.

Le chiavi del risultato Python sono:

| Chiave | Forma | Significato |
|---|---|---|
| `cs` | `(N,2)` | Colonna 0: c; colonna 1: s |
| `c`, `s` | `(N,)` | Coordinate adimensionali |
| `center_MHz` | `(N,)` | Centro previsto |
| `delta_MHz` | `(N,)` | Separazione completa prevista |
| `nu1_MHz`, `nu2_MHz` | `(N,)` | Centri delle due componenti previsti dalla CNN |
| `cs_by_seed` | `(S,N,2)` | Previsioni delle S reti caricate; S=3 per l'ensemble |
| `seeds` | `(S,)` | Identità dei checkpoint caricati |

Sul test completo del riferimento a 50k/48M, l'ensemble aveva RMSE circa **125,6 kHz sul centro** e **156,5 kHz sullo splitting** rispetto alle label fit. Sono risultati storici del protocollo, non una promessa su ogni spettro o sui dati reali. Un errore normalizzato di 0,001 equivale a 0,2 MHz, cioè 200 kHz. I nuovi risultati su soli 20 spettri possono differire dalla media su 5000.

## 11. Come mettere il progetto su GitHub

Carica i sorgenti, questo `README.md`, `requirements.txt`, le cartelle `docs` e `tests`, `.gitignore` e `.gitattributes`. Il file `README.md` è la pagina guida che GitHub visualizza automaticamente.

I MAT, gli ambienti virtuali, i risultati e i pesi sono esclusi dal versionamento tramite `.gitignore`. Per consentire al collega di usare subito la CNN, passa anche lo ZIP con i pesi creato da `export_weights.py`, oppure pubblica quell'archivio tra gli asset di una Release del tuo repository. Il MAT è necessario per riprodurre il controllo storico; per l'API su conteggi già osservati bastano codice e pesi.

Il pacchetto è pronto per essere caricato nel tuo repository; nessun account, token o percorso personale è richiesto per importare la CNN.

## 12. Limiti e verifiche

La griglia e il verso dei dip sono controllati dall'API. L'asse deve essere in **MHz**, crescente, con 335 punti tra 2750 e 2950. Non è prevista un'interpolazione automatica. Anche dati con la forma corretta possono essere fuori dal dominio di training per orientazione, contrasto, baseline, numero di risonanze o rumore.

Le uscite sono lineari e non vengono tagliate a [0,1]. Un risultato fuori intervallo resta visibile. Le frequenze previste sono trasformazioni delle due uscite CNN, senza ulteriori aggiustamenti.

I test controllano forma del modello, conteggio parametri, loss, conversioni, min-max e salvataggio/caricamento. La compatibilità con la struttura storica è inoltre verificata in fase di preparazione del pacchetto; la prova con i checkpoint reali viene completata sul PC che li conserva.

Per i dettagli sulla provenienza leggi [docs/REFERENCE.md](docs/REFERENCE.md). La documentazione delle API di riferimento è [Keras](https://keras.io/api/models/model_saving_apis/weights_saving_and_loading/) e [TensorFlow](https://www.tensorflow.org/install/pip).
