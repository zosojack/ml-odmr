import numpy as np
import scipy.io as sio

# 1. Carichiamo il dizionario dal file .mat
mat_contents = sio.loadmat('data/10k/odmr_dataset_10k.mat')

# 2. Estraiamo gli array che ci interessano
# (scipy.io estrae anche metadati, noi prendiamo solo i nostri dati)
X_spectra = mat_contents['spectra_data']  # Array di shape (n, 1024)
y_B_field = mat_contents['B_labels'].flatten()  # Array di shape (n,)
x_axis = mat_contents['mw_freqs'].flatten() # Le frequenze (se ti servono per plottare)

separate = True

# 3. Salviamo in formato binario compresso nativo di Numpy
if separate:
    np.save('data/X_spectra_10k.npy', X_spectra)
    np.save('data/Y_B_field_10k.npy', y_B_field)

    print(f"Dataset salvato! Shape degli spettri: {X_spectra.shape}")
    print(f"Dataset salvato! Shape delle label B: {y_B_field.shape}")
# 3.1 Salvo in npz
else: 
    np.savez_compressed('data/odmr_dataset_10k.npz', X_spectra=X_spectra, y_B_field=y_B_field)

    print(f"Dataset salvato! Shape degli spettri: {X_spectra.shape}")
    print(f"Dataset salvato! Shape delle label B: {y_B_field.shape}")
