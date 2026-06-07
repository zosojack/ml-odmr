import numpy as np
from scipy.spatial.transform import Rotation as R

# 1. Definiamo le matrici dei 4 assi NV nel sistema del cristallo
U = np.array([
    [ 1,  1,  1],
    [ 1, -1, -1],
    [-1,  1, -1],
    [-1, -1,  1]
]) / np.sqrt(3)

# 2. Inserisci i 4 angoli corretti (ESEMPIO geometricamente valido)
# Nota: devono rispettare la condizione sum(cos^2) = 4/3
angoli_deg = np.array([35.26, 35.26, 90.0, 90.0]) 
cos_theta = np.cos(np.radians(angoli_deg))

# 3. Risolviamo il sistema lineare per trovare le coordinate di B nel cristallo
# Usiamo i minimi quadrati (lstsq) perché il sistema è sovradeterminato (4 equazioni, 3 incognite)
B_c, _, _, _ = np.linalg.lstsq(U, cos_theta, rcond=None)

# Normalizziamo il vettore B per sicurezza
B_c = B_c / np.linalg.norm(B_c)
print(f"Direzione del campo B nel cristallo: {B_c}")

# 4. Ricaviamo gli angoli di Eulero
# Per definire gli angoli di Eulero serve una rotazione completa (matrice 3x3).
# Assumendo che il tuo sistema di riferimento di laboratorio abbia l'asse Z lungo il campo B:
Z_lab = np.array([0, 0, 1])

# Troviamo la rotazione che porta Z_lab a coincidere con B_c
v = np.cross(Z_lab, B_c)
c = np.dot(Z_lab, B_c)
s = np.linalg.norm(v)
kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
rot_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

# Convertiamo la matrice di rotazione in angoli di Eulero (es. sequenza 'zyx')
rot = R.from_matrix(rot_matrix)
euler_angles = rot.as_euler('zyx', degrees=True)

print(f"Angoli di Eulero (Z-Y-X) del cristallo rispetto a B: {euler_angles}")