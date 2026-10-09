# Ma come funziona ODMRDataset? E perché conviene?

**`ODMRDataset`** è classe erede dei **`Dataset`** di **`pytorch`**. Ci permette di usare **`random_split`** e di renderlo successivamente un **iterable** **`DataLoader`**: al modello non viene passata la matrice $N\times167$ completa, ma un oggetto che richiama uno spettro alla volta! 
Il metodo **`__getitem__`** di `ODMRDataset` **non viene eseguito nel costruttore, ma viene chiamato dall'iterabile ogni singola volta che si accede all'elemento $i$-esimo**.

Quando scrivi:

```python
train_loader = DataLoader(train_subset, batch_size=128, shuffle=True)

```

e poi nel training loop esegui:

```python
for batch_x, batch_y in train_loader:
    ...

```

accade precisamente questo sotto il cofano:

1. `DataLoader` pesca 128 indici casuali (es. `idx = 42, 107, 3051, ...`).


2. Invoca internamente la sintassi `dataset[idx]`, che in Python è la chiamata diretta a **`dataset.__getitem__(idx)`**.


3. Prende i 128 tensori restituiti da quelle 128 chiamate e li impila assieme (*collate*) formando un tensore unico di batch `[128, 1, 167]` e `[128, 2]`.



---

### Perché non farlo subito all'inizio?

Anche potrebbe sembrare, il peso sulla RAM non è il fattore determinante: $10\,000 \times 167$ float32 sono circa **$6.7\text{ MB}$**, una quantità irrisoria per qualsiasi macchina moderna. Se fosse solo una questione di memoria, fare tutto prima in un unico grande array NumPy andrebbe benissimo. Anche se si avessero centinaia di migliaia di spettri!

Il motivo per cui **non lo facciamo subito** è di natura prettamente **statistica e di machine learning**:

#### 1. Dataset virtualmente infinito (DATA AUGMENTATION fisica)

* **Se calcoli il rumore subito (statico):**
Generi una singola realizzazione di rumore e contrasto per ciascuno dei 10.000 spettri. La rete vedrà *esattamente gli stessi 8.000 spettri rumorosi* identici all'epoca 1, all'epoca 10, all'epoca 40. Poiché una CNN è un potente approssimatore non lineare, dopo 15–20 epoche comincerà a minimizzare la loss memorizzando le creste e le valli del rumore statico di quei campioni specifici (**overfitting**).


* **Calcolando dentro `__getitem__` (dinamico / on-the-fly):**
All'epoca 1 lo spettro `idx = 42` riceve un contrasto casuale del $13\%$ e $3.2 \times 10^6$ fotoni (rumore basso). All'epoca 2, lo **stesso** spettro `idx = 42` viene ripescato, ma `__getitem__` estrae un contrasto del $2\%$ e $4 \times 10^5$ fotoni (rumore molto marcato) con fluttuazioni poissoniane completamente nuove.


In 40 epoche la rete non vede 8.000 spettri, ma **$8.000 \times 40 = 320\,000$ realizzazioni stocastiche diverse** dello stesso fenomeno fisico sottostante. Questo costringe i filtri convoluzionali a imparare la forma del dip lorenziano e a scartare il rumore shot-noise, rendendo il modello robusto.