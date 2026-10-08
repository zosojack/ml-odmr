La struttura con la classe che eredita da `Dataset` e il metodo `__getitem__` è il design pattern standard di **PyTorch** per gestire i dati. Non serve per "complicare il codice", ma per risolvere due problemi specifici legati all'efficienza e alla statistica del rumore.

---

### 1. Perché si usa una classe (`torch.utils.data.Dataset`)?

In PyTorch, per passare i dati a una rete neurale non si passa quasi mai un enorme array NumPy direttamente al loop di calcolo. Si separano due ruoli ben distinti:

1. **La sorgente del singolo dato (`Dataset`):** sa come si estrae *un solo campione alla volta* (il campione all'indice $i$).


2. **Il distributore (`DataLoader`):** si occupa di prendere più campioni singoli generati dal `Dataset`, mescolarli casualmente (`shuffle=True`), raggrupparli in pacchetti (`batch_size=128`) e parallelizzare il caricamento su più core della CPU.



Affinché il `DataLoader` possa dialogare con i tuoi dati, PyTorch richiede che l'oggetto sorgente rispetti il protocollo di Python implementando due soli metodi speciali:

* `__len__(self)`: per sapere quanti elementi ci sono in totale (così sa quanti batch creare per epoca).


* `__getitem__(self, idx)`: la funzione che dice: *"Dato l'indice `idx` (da $0$ a $9999$), dammi la coppia $(X, Y)$ corrispondente a quel singolo spettro"*.


---

### 2. Perché abbiamo messo il rumore dentro `__getitem__` (on-the-fly)?

Questa è la scelta architetturale più importante della pipeline.

Ci sono due modi alternativi per creare il dataset rumoroso:

#### Approccio A: Pre-calcolare tutto prima (statico)

Prima del training prendi i 10.000 spettri ideali di EasySpin, applichi il contrasto e il rumore Poissoniano a ciascuno, e salvi una matrice rumorosa fissa in RAM.

* **Problema:** La rete vedrebbe *esattamente la stessa realizzazione di rumore* per tutti i 10.000 spettri a ogni singola epoca. Dopo 10–15 epoche, un modello neurale tende a memorizzare le singole fluttuazioni casuali dei pixel/bin anziché la forma fisica lorenziana sottostante (overfitting sul rumore).

#### Approccio B: Generare il rumore "al volo" dentro `__getitem__` (dinamico / on-the-fly)

In RAM tieni solo i profili fisici puri e puliti (`spectra_data`) generati da EasySpin.
Quando il training loop chiede un batch (ad esempio il batch contenente lo spettro $42$), il metodo `__getitem__` viene invocato per quell'indice ed esegue in quell'istante:

```python
def __getitem__(self, idx):
    y_ideal = self.raw_spectra[idx]  # 1. Prende lo spettro liscio puro di EasySpin
    contrast = np.random.uniform(...)  # 2. Estrae un contrasto a caso
    spectrum_ideal = 1.0 - contrast * y_ideal  # 3. Applica il dip
    n_tot = np.random.uniform(...)  # 4. Estrae un budget di fotoni a caso
    noisy_counts = np.random.poisson(
        ...
    )  # 5. Genera una realizzazione stocastica di rumore
    normalized_spectrum = (
        noisy_counts - mean
    ) / std  # 6. Standardizza il segnale
    return x_tensor, y_tensor

```

**Cosa comporta questo fisicamente?**

* All'Epoca 1, lo spettro $42$ viene visto con un contrasto dell'$8\%$ e $500.000$ fotoni (rumore moderato).


* All'Epoca 2, lo **stesso** spettro $42$ viene ripescato, ma `__getitem__` estrae casualmente un contrasto del $2\%$ e $200.000$ fotoni (rumore molto più elevato).


* **Risultato:** Dal punto di vista statistico, la rete neurale non vede mai due volte lo stesso input identico. Questo agisce come una regolarizzazione (data augmentation fisica), forzando i filtri convoluzionali a imparare la posizione invariante dei dip e ignorando il rumore ad alta frequenza.

---

### 3. Come interagiscono `Dataset` e `DataLoader`?

La sequenza di esecuzione a ogni epoca è la seguente:

1. Nel training loop scrivi:
```python
for batch_x, batch_y in train_loader:

```


2. Il `DataLoader` seleziona 128 indici casuali da `train_subset` (ad esempio: `[3, 891, 12, 5400, ...]`).


3. Chiama `ODMRDataset.__getitem__(3)`, `ODMRDataset.__getitem__(891)`, ecc.


4. Ciascuna chiamata genera il suo spettro rumoroso standardizzato e restituisce un tensore di forma `(1, 167)` e un target `(2,)`.


5. Il `DataLoader` impila i 128 risultati (*collate*) e ti restituisce:
* `batch_x`: tensore di forma `[128, 1, 167]` pronto per i layer convoluzionali.


* `batch_y`: tensore di forma `[128, 2]` contenente $[c, s]$ da confrontare con la predizione.





È tutto chiaro su questo meccanismo o ci sono passaggi specifici (sulla Poissoniana, sullo Z-score o sulla gestione degli indici) che vuoi approfondire prima di passare all'architettura?