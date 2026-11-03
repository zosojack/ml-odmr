# Apparente contraddizione tra *success rate* e $\text{RMSE}$

Nell'immagine [centering.png](/4-fit-monte-carlo/graphs/centering.png) i due grafici in alto parrebbero contraddirsi:
il *success rate* sale vicino al $100\%$ ma l'$\text{RMSE}$ rimane stabilmente al di sopra dei $0.4\text{ MHz}$.

> **Perché $\small 0.4\text{ MHz}$ ?**\
> Per poter considerare riuscito (*successful*) un fit, la distanza assoluta tra $c$ vero e il valore fittato dev'essere inferiore a questa quantità. Il motivo è da ricercare nell'articolo di riferimento (*), in cui come limite è posta una distanza assoluta di $0.003$ all'interno dell'intervallo normalizzato $\left[0,1\right]$, originariamente largo $120\text{ MHz}$.

In realtà, va tenuto a mente che nell'$\text{RMSE}$ le deviazioni dal valore vero pesano quadraticamente. Una batch di $10$ mila fit può avere successo nel $99\%$ degli spettri, ma un piccolo gruppo di fallimenti può far esplodere l'errore quadratico medio. Insomma, pochissimi risultati 'catastrofici' possono compromettere l'intero errore di una batch. Ed è importante: perché a noi interessa che i fit funzionino tutti, quindi valutare **quanto** hanno funzionato. Ad esempio è chiaro che $20$ ripetizioni diano dei successi del $100\%$, ma con $40$ tentativi di fit si accede a $\text{RMSE}$ inferiori!

(*) [A Deep-Learning-Boosted Framework for Quantum Sensing with Nitrogen-Vacancy Centers in Diamond | **Yao et al.**](https://arxiv.org/abs/2603.14728v1)


---

### Riassumendo:

I due grafici non sono in contraddizione, ma descrivono due aspetti complementari:   
- *Success Rate*: misura la frequenza con cui il fit trova il dip corretto (ignora l'ampiezza dell'errore quando sbaglia).  
- $\text{RMSE}$: misura l'incertezza quadratica totale, la quale viene dominata pesantemente anche da un piccolo $5-10\%$ di fallimenti catastrofici rimasti intrappolati in minimi locali errati.   