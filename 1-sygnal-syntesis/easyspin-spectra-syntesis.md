# Generazione degli Spettri ODMR Sintetici con EasySpin

**Paper di riferimento:**

[A Deep-Learning-Boosted Framework for Quantum Sensing with Nitrogen-Vacancy
Centers in Diamond | **Yao et al.**](https://arxiv.org/abs/2603.14728v1)

## 1. Motivazione Fisica e Scelte Modellistiche
Per addestrare e validare l'architettura 1D-CNN proposta da Yao et al., la generazione dei dati sintetici viene effettuata risolvendo direttamente l'Hamiltoniana di spin dei centri NV (*Nitrogen-Vacancy*) in diamante tramite il toolbox EasySpin. Questo approccio sostituisce il modello puramente fenomenologico a due lorentziane con una simulazione quantistica coerente con i dati sperimentali di un cristallo di diamante macroscopico (*bulk*) ($D=2.87\ \text{GHz}$, $E=10\ \text{MHz}$, $\gamma=28\ \text{GHz T}^{-1}$).

Le scelte adottate nella simulazione derivano dai seguenti vincoli fisici e modellistici:

* **Inclusione della Simmetria Cristallina a 4 Assi:** 
    A differenza di modelli giocattolo monoassiali, la simulazione implementa il gruppo spaziale del diamante (`CrystalSymmetry = 227`) popolando le 4 classi di orientazione tetraedriche $\langle 111 \rangle$ regolate dalla terna di angoli di Eulero scelta semi-arbitrariamente in base alle osservazioni in laboratorio ($\alpha = 36.1^\circ, \beta = 42.39^\circ, \gamma = 0.61^\circ$). Nel regime di campo magnetico statico debole indagato ($B \le 0.237\ \text{mT}$), i dip delle diverse classi non si separano in 8 rami distinti ma si sovrappongono in due macro-strutture di risonanza allargate, riproducendo la risposta asimmetrica e convoluta del cristallo reale.

* **Finestra di Sweep Estesa e Risoluzione Costante:** 
    Per evitare il troncamento degli estremi dello spettro - con conseguente perdita dei dip dovuti all'iperfine del carbonio - e per mantenere al contempo la medesima densità di campionamento spettrale di Yao et al. ($\approx 1.20\ \text{MHz/punto}$), la finestra di sweep delle microonde è estesa a un intervallo di $200\ \text{MHz}$, nello specifico tra $2.75\ \text{GHz}$ e $2.95\ \text{GHz}$ (`Exp.mwRange = [2.75, 2.95]`), discretizzato su esattamente $167$ punti equispaziati in assorbimento diretto (`Exp.Harmonic = 0`). (Invece di $120\text{MHz}$ di finestra su $101\text{ punti}$).

* **Campionamento del Centro di Risonanza ($D$):**
    Il termine assiale di Zero-Field Splitting subisce variazioni indotte da fluttuazioni termiche ($dD/dT \approx -70\ \text{kHz/K}$) o gradienti di strain locale. Al fine di migliorare l'apprendimento della rete, il parametro $D$ viene campionato uniformemente nell'intervallo $D \in [2842, 2878]\ \text{MHz}$. Questo intervallo deriva dalla proiezione fisica dell'intervallo normalizzato del centro $c \in [0.35, 0.65]$ definito da Yao et al. sulla finestra di riferimento standard da $120\ \text{MHz}$ ($[2.80, 2.92]\ \text{GHz}$). 
    **Quindi:** usiamo la loro finestra per calcolare i nostri parametri, MA lo spettro lo generiamo su una finestra più ampia.

* **Campionamento dello Splitting e Calcolo del Campo Magnetico ($B$):** 
    Mantenendo costante l'anisotropia rombica a $E = 10\ \text{MHz}$, lo splitting minimo a campo nullo è pari a $\delta_{\min} = 2E = 20\ \text{MHz}$. Normalizzando tale valore sulla scala di riferimento da $120\ \text{MHz}$, lo splitting adimensionale $s$ viene campionato uniformemente tra $s_{\min} = 20/120 \approx 0.1667$ e $s_{\max} = 0.20$ (come prescritto da Yao et al.). Il campo magnetico statico $B$ viene ricavato per inversione analitica dello splitting $\delta = s \times 120\ \text{MHz}$:
    $$B = \frac{1}{\gamma} \sqrt{\left(\frac{\delta}{2}\right)^2 - E^2}$$
    risultando confinato nell'intervallo $B \in [0.0, 0.237]\ \text{mT}$.

* **Allargamento di Riga (Broadening):**
    La larghezza a metà altezza (FWHM) delle transizioni lorentziane viene campionata uniformemente come $\Gamma_L \in [2.4, 10.8]\ \text{MHz}$, derivata dal campionamento normalizzato $w \in [0.02, 0.09]$ di Yao et al. riscalato sui $120\ \text{MHz}$ di riferimento.

---

## 2. Dati Generati e Struttura di Output

Si simulano $N$ spettri (primi test: $N = 10\,000$).

Per ciascuna iterazione, EasySpin risolve numericamente l'Hamiltoniana di spin considerando tutti i siti equivalenti della cella unitaria del diamante:

$$\hat{H} = \sum_{k=1}^4 \left[ D \left( \hat{S}_{z,k}^2 - \frac{S(S+1)}{3} \right) + E \left( \hat{S}_{x,k}^2 - \hat{S}_{y,k}^2 \right) + g \mu_B \vec{B} \cdot \hat{\mathbf{S}}_k \right]$$

Ciascun profilo risultante viene normalizzato al proprio picco massimo.

I dati vengono esportati in formato MATLAB `.mat` compatibile con HDF5 (`-v7.3`) per la lettura diretta in Python tramite `h5py` o `scipy.io`:

* `spectra_data`: Matrice $[N \times 167]$ contenente i profili spettrali raw di assorbimento risonante calcolati sulla finestra $[2.75, 2.95]\ \text{GHz}$;
* `labels_B`: Vettore $[N \times 1]$ dei valori di campo magnetico statico $B$ applicato ($\text{mT}$);
* `labels_D`: Vettore $[N \times 1]$ dei valori del centro $D$ estratti ($\text{MHz}$);
* `labels_lw`: Vettore $[N \times 1]$ dei valori di broadening lorentziano intrinseco $\Gamma_L$ ($\text{MHz}$);
* `labels_c`: Vettore $[N \times 1]$ dei centri normalizzati adimensionali $c \in [0.35, 0.65]$;
* `labels_s`: Vettore $[N \times 1]$ degli splitting normalizzati adimensionali $s \in [0.1667, 0.20]$;
* `mw_freqs`: Vettore $[1 \times 167]$ con l'asse delle frequenze fisiche dello sweep espresse in $\text{GHz}$.##