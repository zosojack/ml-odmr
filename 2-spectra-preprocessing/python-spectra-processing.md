# Pipeline di Preprocessing e Preparazione Dati in Python

**Paper di riferimento:**
 
[A Deep-Learning-Boosted Framework for Quantum Sensing with Nitrogen-Vacancy
Centers in Diamond | **Yao et al.**](https://arxiv.org/abs/2603.14728v1)

## 1. Obiettivi della Pipeline

 Questa fase converte i profili di assorbimento ideali simulati tramite EasySpin per il diamante bulk in segnali ODMR realistici e pronti per l'apprendimento supervisionato. Vengono applicati **contrasto ottico**, **shot noise poissoniano** e **standardizzazione Z-score** coerentemente con la metodologia di Yao et al., e vengono definite le coordinate di regressione ($x_1, x_2, c, s$) normalizzate nello spazio adimensionale $[0, 1]$ associato alla finestra di sweep estesa a $167$ punti.

---

## 2. Definizione dei Target di Regressione (Ground Truth)

 Le grandezze fisiche estratte da EasySpin ($D$ in MHz, $B$ in mT, $E = 10\text{ MHz}$, $\gamma = 28\text{ GHz/T}$; si veda la fase di sintesi spettrale) individuano le frequenze di risonanza analitiche del centro NV:

 $$\nu_{\pm} = D \pm \sqrt{E^2 + (\gamma B)^2}$$

### Proiezione sulla Finestra Estesa di Sweep
 Mentre i bound di campionamento fisico sono stati regolati sui $120\text{ MHz}$ originali di Yao et al., l'acquisizione spettrale copre l'intervallo esteso $[\nu_{\min}, \nu_{\max}] = [2.75, 2.95]\text{ GHz}$ ($\Delta\nu_{\text{sweep}} = 200\text{ MHz}$) per preservare l'integrità dei dip senza troncamento ai bordi. Le coordinate adimensionali dei dip passate alla 1D-CNN sono definite rispetto all'effettiva finestra di sweep:

 $$x_1 = \frac{\nu_- - \nu_{\min}}{\Delta\nu_{\text{sweep}}}, \quad x_2 = \frac{\nu_+ - \nu_{\min}}{\Delta\nu_{\text{sweep}}}$$

 Per costruzione vale $0 < x_1 < x_2 < 1$. Da queste coordinate si calcolano il centro e lo splitting normalizzati:

 $$c = \frac{x_1 + x_2}{2}, \quad s = x_2 - x_1$$

 Le larghezze di riga lorentziane intrinseche $\Gamma_L$ vengono analogamente riscalate sulla finestra di sweep:

 $$w = \frac{\Gamma_L}{\Delta\nu_{\text{sweep}}}$$

 *(Nota: nel dataset salvato da MATLAB rimangono disponibili anche i riferimenti $c_{\text{ref}}$ ed $s_{\text{ref}}$ calcolati sulla scala nominale di $120\text{ MHz}$, permettendo un confronto diretto sia nello spazio dimensionale originale sia nelle coordinate normalizzate di rete).*

---

## 3. Trasformazione dello Spettro e Modello di Rumore

Ciascun profilo di assorbimento $y \in [0, 1]$ campionato su $167$ punti è sottoposto alle seguenti manipolazioni:

1. **Modulazione del Contrasto Ottico:**
   Viene campionato un contrasto casuale $C \sim \mathcal{U}(0.012, 0.15)$ che determina la profondità dei *dip* rispetto al livello di emissione di base:
   $$I_{\text{ideal}}(\nu) = 1 - C \cdot y(\nu)$$

2. **Shot Noise Poissoniano:**
   Viene estratto un budget totale di fotoni integrati,  $N_{\text{tot}} \sim \mathcal{U}(3 \times 10^5, 6 \times 10^6)$ per coprire un ampio intervallo di rapporti segnale-rumore (SNR). Gli estremi dell'intervallo sono ottenuti a partire da quelli usato da Yao et al.; l'intervallo è stato ri-scalato in seguito all'ingrandimento della finestra di sweep, con l'obiettivo di mantenere la medesima la densità di fotoni per punto (vedere [Quanti fotoni simulare?](/0-FAQ/quanti-fotoni-simulare.md)). 
   Il valor medio di fotoni (attesi) in ciascuno dei $167$ canali spettrali è dato dalla frazione d'intensità del bin stesso sul totale:

   $$\lambda_i = N_{\text{tot}} \cdot \frac{I_{\text{ideal}}(\nu_i)}{\sum_{j=1}^{167} I_{\text{ideal}}(\nu_j)}$$

   Il conteggio di fotoni osservato per ciascun bin viene quindi campionato indipendentemente da una Poissoniana con media $\lambda_i$:
   $$N_{\text{obs}}(\nu_i) \sim \text{Poisson}(\lambda_i)$$

3. **Standardizzazione Z-score:**
   Il vettore rumoroso $N_{\text{obs}}$ viene centrato e normalizzato rispetto alla propria media e deviazione standard:
   $$I_{\text{norm}}(\nu_i) = \frac{N_{\text{obs}}(\nu_i) - \mu}{\sigma}$$
   Questa operazione rende il segnale invariante rispetto alla potenza del laser di eccitazione, all'efficienza di raccolta ottica e agli offset sistematici.

---

## 4. Architettura del Dataset e Strategia di Implementazione

* **Dataset PyTorch (`torch.utils.data.Dataset`):** 
Mantiene in RAM la matrice grezza $[10\,000 \times 167]$ degli spettri prodotti da EasySpin e applica on-the-fly contrasto, shot noise e Z-score all'interno del metodo `__getitem__`. Questa strategia garantisce che la rete veda a ogni epoca una realizzazione di rumore statisticamente indipendente, fungendo da "regolarizzazione implicita" contro l'overfitting.

* **Forma dei Tensori di Input:** Ciascun campione di input ha dimensione tensoriale $[1, 167]$: singolo canale, 167 feature spaziali (= frequenze).

* **Partizionamento dei Dati:** Suddivisione deterministica (fissando il seed a 42) in:
  * **Training set (80%):** 8.000 spettri con generazione stocastica del rumore ad ogni epoca.
  * **Validation set (10%):** 1.000 spettri per il monitoraggio della loss e l'early stopping.
  * **Test set (10%):** 1.000 spettri riservati al benchmark finale e all'analisi dell'incertezza.
  
* **DataLoader:** Configurazione di batch size a 128 o 256 campioni con caricamento parallelizzato.