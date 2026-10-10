# Come approcciare il fit tradizionale?

### 1. Quante lorentziane usare: 2 o 8?

Meglio usare **due lorentziane**, esattamente come fanno Yao et al., per due ragioni fisiche e algoritmiche:

1. **Il regime fisico ($B \le 0.24\text{ mT}$):**
    Con un campo magnetico così debole, le 4 classi di orientazione NV non si separano in 8 picchi risolti. Lo splitting Zeeman massimo è di appena $3 - 4\text{ MHz}$, inferiore alla larghezza di riga intrinseca ($\Gamma_L \approx 4 - 8\text{ MHz}$). Visivamente e numericamente lo spettro presenta **due soli avvallamenti macroscopici**.

1. **Sovraparametrizzazione del fit:**
    Un modello a 8 lorentziane ha almeno $8 \times 3 = 24$ parametri liberi (ampiezze, centri, larghezze). Eseguire un fit non lineare a 24 parametri su uno spettro con forte shot noise che mostra due soli dip evidenti porta l'algoritmo (Levenberg-Marquardt o Trust-Region) a una degenerazione matematica immediata: la matrice Jacobiana diventa singolare e i parametri divergono.
    Il modello da fittare è quindi la somma di due lorentziane su un fondo normalizzato:

    $$y(\nu) = A_0 - \frac{A_1}{1 + \left(\frac{\nu - x_1}{w_1/2}\right)^2} - \frac{A_2}{1 + \left(\frac{\nu - x_2}{w_2/2}\right)^2}$$

    con parametri liberi da stimare: $\mathbf{p} = [A_0, A_1, A_2, x_1, x_2, w_1, w_2]$ (7 parametri).

---

### 2. Cosa significa "inizializzare i parametri con Monte Carlo"?

I metodi ai minimi quadrati non lineari (es. `scipy.optimize.curve_fit`, che usa l'algoritmo di Levenberg-Marquardt) sono ottimizzatori **locali**. Questo significa che per convergere al minimo corretto della funzione di costo:

$$\chi^2(\mathbf{p}) = \sum_{i=1}^{167} \left( y_i^{\text{exp}} - y(\nu_i; \mathbf{p}) \right)^2$$

hanno bisogno di un punto di partenza iniziale (guess) $\mathbf{p}_0$ sufficientemente vicino alla soluzione.

Se lo spettro ha poco rumore (alto SNR), basta trovare i minimi con una derivata o mettere $\mathbf{p}_0$ fisso a $2.87\text{ GHz}$.

**Ma cosa succede a basso SNR (basso numero di fotoni, rumore elevato)?**
Il rumore genera decine di falsi minimi locali. Se la guess iniziale $\mathbf{p}_0$ è statica o sfortunata, il fit converge in una buca di rumore lontana dai dip reali (*fitting failure*).

Per questo motivo, nella letteratura della metrologia quantistica e in Yao et al. il fit tradizionale non usa una sola guess fissa, ma un approccio **Multi-start Monte Carlo**:

1. Per ciascuno spettro, vengono campionati $M$ vettori di parametri iniziali casuali $\mathbf{p}_0^{(1)}, \dots, \mathbf{p}_0^{(M)}$ (in Yao et al. $M \approx 10 - 50$) all'interno dei limiti fisici noti ($x_1, x_2 \in [2.80, 2.92]\text{ GHz}$, ampiezze coerenti col contrasto).

1. Per ciascuno dei tentativi viene lanciato l'algoritmo ai minimi quadrati.

1. Tra le soluzioni convergenti, si seleziona quella che restituisce il valore minimo di $\chi^2$ (miglior fitting globale).

Questo è il "fit tradizionale" contro cui confrontare il modello: un approccio rigoroso, ma **estremamente lento** (ripetere Levenberg-Marquardt 20 volte per ogni spettro moltiplicato per migliaia di spettri richiede ore, mentre la rete calcola l'inferenza in millisecondi per batch).

---

### 3. Che tipo di analisi e benchmark dobbiamo fare?

Il confronto deve quantificare l'errore di stima di $[c, s]$ al variare del rapporto segnale-rumore (SNR) sul test set (i 1000 spettri congelati):

1. **Stratificazione per Budget di Fotoni ($N_{\text{tot}}$):**
    Suddividere il test set in bin di fotoni (ad esempio: basso SNR con $N_{\text{tot}} \sim 10^5$, medio SNR con $10^6$, alto SNR con $5 \times 10^6$).
1. **Metriche di Accuratezza:**
    * **MAE (Mean Absolute Error):** $\vert{}c_{\text{pred}} - c_{\text{true}}\vert{}$ e $\vert{}s_{\text{pred}} - s_{\text{true}}\vert{}$ sia per il fit sia per la CNN.
    * **RMSE (Root Mean Square Error):** per evidenziare quanto spesso il fit classico fallisce completamente (un singolo fit divergente fa esplodere l'RMSE del metodo classico, mentre la rete rimane vincolata).


1. **Tasso di Successo / Fallimento (Outliers):**
    Definire una soglia di tolleranza fisica (es. errore $> 5\text{ MHz}$). Quanti spettri a basso numero di fotoni mandano in tilt il fit rispetto alla rete neurale?
1. **Tempo di Inferenza:**
    Tempo medio per estrarre $[c, s]$ da un singolo spettro (CNN su GPU/MPS vs Fit Levenberg-Marquardt su CPU).

---

### 4. L'analisi in funzione del rumore (il benchmark vero)

Per quanto riguarda l'analisi sperimentale, la tua intuizione è esatta al 100%: l'obiettivo del confronto è dimostrare **dove e perché** il fit classico crolla rispetto alla rete neurale.

Il piano d'azione standard è proprio quello che hai descritto:

1. **Grafico dell'Errore (MAE o RMSE) vs Budget di Fotoni ($N_{\text{tot}}$):**
* Sull'asse $X$: il numero di fotoni $N_{\text{tot}}$ (da $10^5$ a $6 \times 10^6$ in scala logaritmica), che rappresenta direttamente l'SNR e il tempo di dwell.
* Sull'asse $Y$: l'errore assoluto su $c$ (in MHz o coordinate normalizzate) e su $s$ (splitting).
* **Cosa si osserva:**
* A $N_{\text{tot}} \sim 10^6$ (alto SNR), fit classico e 1D-CNN hanno prestazioni quasi identiche (entrambi ottimi).
* Sotto $N_{\text{tot}} \sim 5 \times 10^5$ (basso SNR), la curva dell'errore del fit classico si impenna verso l'alto (divergenza), mentre la CNN degrada in modo molto più dolce e controllato.




2. **Grafico del Success Rate (Tasso di Successo):**
* Si fissa una soglia di successo fisico: ad esempio un errore sullo splitting $\vert{}s_{\text{pred}} - s_{\text{true}}\vert{} < 1\text{ MHz}$ (o un valore in coordinate normalizzate).
* Sull'asse $X$: i bin di fotoni $N_{\text{tot}}$.
* Sull'asse $Y$: la percentuale di spettri per cui il metodo ha centrato la risonanza entro la soglia.
* È esattamente il grafico cardine del paper di Yao: mostra che a bassissimi conteggi di fotoni il fit convenzionale ha un success rate che cola a picco verso il $20-30\%$, mentre la rete si mantiene a livelli ampiamente utilizzabili ($> 85-90\%$).



Vuoi che strutturiamo prima la funzione Python per il fit Levenberg-Marquardt multi-start con `scipy.optimize.curve_fit` su un piccolo sottoinsieme di test per calibrarla, oppure preferisci prima consolidare l'addestramento della CNN?