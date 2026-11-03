# Verifiche del progetto estratto

Verifiche tecniche eseguite il 1 novembre 2026 con TensorFlow CPU 2.20.0 e Keras 3.15.1.

## Test inclusi

Il comando `python -m unittest discover -s tests -v` esegue nove test:

1. Struttura CNN, 806.946 parametri, dropout effettivo 0 e salvataggio/caricamento.
2. Rifiuto delle griglie in GHz, invertite o con lunghezza errata.
3. Conversione di c/s in centro, separazione completa e due frequenze.
4. Lettura di MAT classici e HDF5 con orientamento corretto della matrice.
5. Min-max per spettro con il verso dei dip corretto.
6. Errore esplicito se mancano i pesi appresi.
7. Metrica RMSE con conversione corretta delle unità.
8. Ripetibilità del protocollo Poisson di riferimento.
9. Valore della loss pesata confrontato con la formula analitica.

## Confronto con il sorgente storico

Durante la preparazione sono stati eseguiti anche controlli privati sul codice archiviato nel run:

- Con lo stesso seed, i parametri iniziali della nuova funzione `build_cnn()` coincidono con quelli del costruttore storico.
- Un file di pesi salvato dal costruttore storico è caricato nella CNN estratta e produce le stesse previsioni sugli stessi input artificiali.
- I conteggi, contrasti e input normalizzati del nuovo preprocessing coincidono con il protocollo originale su 300 profili artificiali, includendo più di un blocco RNG.
- Esportazione da file finale, cartella `work` e backup ZIP; nessuna modifica ai file originali.
- Ripetizione dell'esportazione già completata, API ensemble e singola rete, esecuzione da MAT e da NPZ di conteggi.
- Rifiuto di un checkpoint modificato dopo l'esportazione.

Le prove di integrazione usano pesi casuali e dati artificiali temporanei, esclusi dal pacchetto. Non sono risultati di accuratezza scientifica e non sostituiscono i checkpoint del progetto.

## Verifica da completare sul PC di Antony

I checkpoint reali non erano inclusi negli archivi di analisi disponibili. `export_weights.py` verifica le loro identità nella cartella originale; `predict.py --mat ...` confronta inoltre 64 predizioni di validation con quelle archiviate.

La tolleranza di 1 kHz in questo controllo serve a verificare il trasferimento del modello. La precisione storica del modello è invece quantificata dalle metriche sul test riportate nel README.
