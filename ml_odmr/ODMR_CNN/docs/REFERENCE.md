# Provenienza del modello finale

Questa estrazione consegna la CNN B del run `risultati_grid27_cfit_24h`, selezionando la condizione **dataset 50k / 48M fotoni** e i seed **42, 137, 2026**. La procedura di fitting del collega resta esterna a questo progetto.

## Corrispondenza con il codice storico

L'architettura di `build_cnn()` corrisponde alla funzione storica:

```python
build_model(length=335, q_scale=TARGET_SCALE, span_MHz=200,
            cfg=cfg, center_weight=10.0, dropout_rate=0.0)
```

Il nuovo nome e la riorganizzazione dei file servono a isolare il modello e la sua interfaccia. L'ordine delle operazioni, le dimensioni, le attivazioni, la loss e le trasformazioni non sono una nuova architettura. Il codice di caricamento mantiene tutti i parametri appresi: nessun `skip_mismatch`.

Il codice storico archiviato nel run è identificato da SHA256:

```text
1c03aece0377468e6a8912dd77cfbe182829656388be7d50e4f419b942c0542b
```

La firma della configurazione del run è:

```text
5a708b8af9addc728beba2ea4f5bd1c21bd10bdf0baaa2e5dc6cb93993c05444
```

I tre checkpoint sono quelli scelti per la migliore loss sulla validation:

| Seed | Epoca migliore, numerazione da 1 | SHA256 dei pesi |
|---|---|---|
| 42 | 160 | `c52e94a6aefab3d49369e701ad1316c380e73d2483527ce2321c7135adffa4e2` |
| 137 | 172 | `6b3f88e10b39f4758642051a5a65fa196b676d7ce9d9805528f12ed7f4379e22` |
| 2026 | 172 | `e6386309eaf31aa02119fc360c65ad0354aa8d0c86fef1ef863eebe8809efe2f` |

Le firme permettono di identificare file uguali anche se rinominati. Non misurano la qualità statistica della rete.

## Dataset e dominio fisico

Il dataset finale contiene 50.000 profili di assorbimento EasySpin, 335 punti su 2,75-2,95 GHz. Le label `labels_c` e `labels_s` sono già state sostituite con quelle del fit pulito, normalizzate sulla finestra effettiva di 200 MHz. Questo progetto le legge, non le ricalcola.

La simulazione del progetto usa quattro orientazioni NV, geometria del cristallo fissata, E=10 MHz, D campionato in 2842-2878 MHz, campo nominale circa 0-0,237 mT e FWHM corrette 4,8-21,6 MHz. Le label operative hanno splitting circa 19,9996-21,6747 MHz, pur partendo dal campionamento nominale 20-24 MHz.

Il MAT del riferimento è identificato da:

```text
55a0abdc85da2fb783bd71e0edc5851ced8a6bb0de8c2ad0908f9b6497e074d3
```

Il confronto storico ha training annidati da 8k, 16k e 40k, con validation e test comuni da 5k ciascuno. La condizione consegnata usa i 40k del training maggiore. La scala q è stata fissata usando il training minimo da 8k e mantenuta comune; ricalcolarla sui 40k renderebbe errata la decodifica di questi pesi.

## Ambiente e riproducibilità

Il riferimento registra TensorFlow 2.20.0, Keras 3.15.1, Python 3.12.10 e NumPy 2.5.4 su Windows. Le dipendenze del progetto fissano TensorFlow/Keras e consentono versioni compatibili delle altre librerie. Cambi di piattaforma o libreria possono introdurre differenze numeriche: il controllo sul MAT confronta 64 previsioni archiviate con tolleranza di 1 kHz.

Non si promette identità bit a bit tra sistemi. Il codice non rimpiazza mai i pesi mancanti con un modello casuale. L'archivio distribuito con i sorgenti non include i pesi del PC di Antony; lo script di esportazione li recupera localmente.

## Ricetta di training già eseguita

- Batch 256, due osservazioni rumorose per spettro a ogni epoca.
- AdamW: learning rate iniziale 3e-4, weight decay 1e-5, clipnorm 1.
- MSE pesata con centro 10 e splitting 1 nello spazio q, divisa per 11.
- Massimo 200 epoche; riduzioni del learning rate e early stopping sul valore di validation.
- Monitoraggio separato degli errori di centro e splitting in MHz.
- Miglior checkpoint scelto separatamente per ciascun seed, poi media delle previsioni.

Il pacchetto contiene l'implementazione del modello e della loss; non include un nuovo ciclo di addestramento o una procedura di ripresa del vecchio confronto. Servono codice e pesi per eseguire la CNN già addestrata.
