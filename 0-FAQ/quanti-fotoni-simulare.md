# Il riscalamento corretto per $167$ punti

Per garantire che la **densità di fotoni per punto** (e quindi l'effettivo SNR locale e il tempo di permanenza $t_{\text{dwell}}$ per canale) rimanga identica a quella studiata da Yao et al., il budget totale va moltiplicato per il rapporto dei punti:

$$\text{Fattore di scala} = \frac{167}{101} \approx 1.6535$$

Applichiamo il fattore agli estremi dell'intervallo:

* **Estremo inferiore ($N_{\min}$):**

$$1.8 \times 10^5 \times \frac{167}{101} \approx 2.976 \times 10^5 \approx \mathbf{3.0 \times 10^5}$$


* **Estremo superiore ($N_{\max}$):**

$$3.6 \times 10^6 \times \frac{167}{101} \approx 5.952 \times 10^6 \approx \mathbf{6.0 \times 10^6}$$



Nel metodo `__getitem__` della classe `ODMRDataset`, la riga diventa semplicemente:

```python
# Shot noise poissoniano coerente con il dwell-time di Yao et al. per 167 punti
n_tot = np.random.uniform(3.0e5, 6.0e6)

```

In questo modo ciascuno dei 167 bin riceve in media tra $\approx 1.8 \times 10^3$ e $\approx 3.6 \times 10^4$ fotoni, riproducendo l'identica fisica di rumore di Yao pur coprendo una finestra di sweep più ampia.