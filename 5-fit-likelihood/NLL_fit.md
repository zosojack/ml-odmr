Usiamo il valore della doppia lorentziana come parametro di valore atteso ($\lambda$) dentro la distribuzione di Poisson.

**1. Il modello statistico del singolo punto (canale $i$)**
Uno spettro è discretizzato su $K$ frequenze $f_1, f_2, \dots, f_K$. Per una data scelta dei parametri fisici $\theta = (A_0, A_1, x_1, w_1, A_2, x_2, w_2)$, il modello teorico senza rumore predice per il canale $i$-esimo un numero **medio atteso** di fotoni conteggiati pari a:

$$\lambda_i(\theta) = A_0 - \mathcal{L}_1(f_i; A_1, x_1, w_1) - \mathcal{L}_2(f_i; A_2, x_2, w_2)$$

(dove $A_0, A_1, A_2$ sono espressi direttamente in numero di conteggi fotonici, non in Z-score).

Quando fai l'esperimento (o la simulazione rumorosa), al canale $i$ osservi un conteggio intero $y_i$. Quel singolo dato $y_i$ è una variabile casuale estratta da una distribuzione di Poisson la cui media è proprio $\lambda_i(\theta)$:

$$y_i \sim \text{Poisson}\big(\lambda_i(\theta)\big)$$

La probabilità di osservare esattamente $y_i$ fotoni nel canale $i$-esimo, fissati i parametri $\theta$, è quindi la PMF (*Probability Mass Function*) di Poisson:

$$P(y_i \mid \theta) = \frac{\big[\lambda_i(\theta)\big]^{y_i} \, e^{-\lambda_i(\theta)}}{y_i!}$$

**2. Costruzione della Likelihood totale ($\mathcal{L}$)**
Poiché lo shot noise fotonico in un canale di frequenza è statisticamente indipendente da quello negli altri canali, la probabilità congiunta di osservare l'intero spettro $\mathbf{y} = (y_1, \dots, y_K)$ è il **prodotto** delle probabilità dei singoli punti:

$$\mathcal{L}(\theta \mid \mathbf{y}) = \prod_{i=1}^{K} P(y_i \mid \theta) = \prod_{i=1}^{K} \frac{\big[\lambda_i(\theta)\big]^{y_i} \, e^{-\lambda_i(\theta)}}{y_i!}$$

**3. Passaggio alla Log-Likelihood (e alla statistica di Cash)**
Moltiplicare centinaia di probabilità piccole manda in *underflow* numerico qualsiasi calcolatore. Si passa quindi al logaritmo naturale, trasformando la produttoria in sommatoria:

$$\ln \mathcal{L}(\theta \mid \mathbf{y}) = \sum_{i=1}^{K} \Big[ y_i \ln\big(\lambda_i(\theta)\big) - \lambda_i(\theta) - \ln(y_i!) \Big]$$

Poiché il termine $\ln(y_i!)$ dipende solo dai dati misurati e non dai parametri $\theta$ che stai cercando, è una costante ininfluente durante l'ottimizzazione e si scarta. In fisica (specialmente in ottica quantistica e astrofisica dei fotoni, dove è nota come *Cash statistic*) si definisce la funzione costo da **minimizzare** come la **Negative Log-Likelihood (NLL)**:

$$\text{NLL}(\theta) = \sum_{i=1}^{K} \Big[ \lambda_i(\theta) - y_i \ln\big(\lambda_i(\theta)\big) \Big]$$

**4. Che differenza c'è rispetto a `curve_fit` (Minimi Quadrati)?**
Quando si usa `curve_fit`, si minimizza la somma dei residui quadrati $\sum_i (y_i - \lambda_i(\theta))^2$. Dal punto di vista della teoria della probabilità, fare i minimi quadrati non pesati equivale a fare una Maximum Likelihood assumendo che:

1. Il rumore sia **gaussiano** anziché poissoniano;
2. La varianza $\sigma_i^2$ sia **costante** su tutta la finestra di frequenza (omoschedasticità).

In realtà, per una variabile di Poisson la varianza è uguale al valore atteso ($\sigma_i^2 = \lambda_i(\theta)$). Dentro i due dip di risonanza ci sono meno fotoni rispetto al fondo, quindi **dentro i dip il rumore assoluto è più basso che sul fondo** (eteroschedasticità). Minimizzando la $\text{NLL}(\theta)$ poissoniana si tiene conto esattamente di questa asimmetria, ma si deve lavorare sui conteggi grezzi positivi ($\lambda_i(\theta) > 0$) senza applicare prima lo Z-score (che introduce valori negativi incompatibili con il logaritmo $\ln \lambda_i$).