### 1. Definizione dell'Architettura 1D-CNN e della Loss Probabilistica

---

**Paper di riferimento:**
 
[A Deep-Learning-Boosted Framework for Quantum Sensing with Nitrogen-Vacancy
Centers in Diamond | **Yao et al.**](https://arxiv.org/abs/2603.14728v1)

* **Architettura della rete:** Progettare la 1D-CNN in PyTorch (blocchi convoluzionali con BatchNorm/ReLU e layer lineari terminali) tarata sull'input a 101 punti.
* **Output probabilistico:** Far predire alla rete per ciascun parametro target (es. $c$ ed $s$) sia il valore atteso $\hat{\mu}$ sia la varianza $\hat{\sigma}^2$ (o $\log \sigma^2$ per stabilità numerica).
* **Negative Log-Likelihood (NLL) Loss:** Implementare la funzione di costo gaussiana:

$$\mathcal{L} = \frac{1}{2} \sum \left( \frac{(y - \hat{\mu})^2}{\hat{\sigma}^2} + \ln \hat{\sigma}^2 \right)$$



che consente alla rete di quantificare l'incertezza statistica (aleatoria) legata al livello di rumore del singolo spettro.

---

### 2. Training Loop e Monitoraggio

* Configurare l'ottimizzatore (es. AdamW) con learning rate decay/scheduler.
* Eseguire il ciclo di addestramento su GPU/MPS/CPU tracciando NLL loss, MAE e RMSE sia sul set di training che su quello di validazione per prevenire l'overfitting.
* Salvare i pesi del miglior modello validato su disco.

---

### 3. Analisi Statistica, Benchmark e Validazione (Cuore dell'elaborato)

* **Confronto con il fit non lineare classico (Least-Squares / Levenberg-Marquardt):** Applicare un fit a due lorenziane con `scipy.optimize.curve_fit` su un sottoinsieme di spettri di test. Mostrare come a basso SNR il fit classico diverga frequentemente in minimi locali o fallisca, mentre la CNN mantenga stabilità.
* **Scaling dell'errore (RMSE vs SNR):** Raggruppare gli spettri di test per livelli di conteggio fotonico ($N_{\text{tot}}$) e verificare se l'errore segue la scala teorica $\propto \text{SNR}^{-1}$ (limite di Cramér-Rao / shot-noise limited).
* **Calibrazione statistica dell'incertezza:** Calcolare i residui standardizzati (*pull distribution*):

$$z = \frac{y - \hat{\mu}}{\hat{\sigma}}$$



e verificare tramite istogramma, Q-Q plot o test di Kolmogorov-Smirnov che seguano una normale standard $\mathcal{N}(0, 1)$, confermando la corretta calibrazione delle barre d'errore.