clear all; clc;

% =========================================================================
% CONFIGURAZIONE DATASET
% =========================================================================
N_spectra = 10000;         % Numero di spettri da generare
B_min = 0.0;               % Limite inferiore campo magnetico (mT)
B_max = 8.;                % Limite superiore campo magnetico (mT)

% Angoli ottimali trovati
alpha_deg = 36.1;        
beta_deg = 42.39;         
gamma_deg = 0.61;        

% =========================================================================
% ----- Setup del Sistema di Spin -----
Sys.S = 1;
Sys.g = 2.0;
Sys.D = [2870 10];
Sys.Nucs = 'C';
Sys.A = [127, 127, 127];
Sys.lwpp = [16 4];

% ----- Parametri Sperimentali -----
Exp.mwRange = [2.5 3.3];     % Range in GHz
Exp.Harmonic = 0;
Exp.CrystalSymmetry = 227;
Opt.Sites = [];
Exp.nPoints = 800;           % Risoluzione asse X

% Configurazione geometrica corretta per i 4 siti NV
Exp.MolFrame = [0, acos(1/sqrt(3)), 0];
Exp.SampleFrame = [alpha_deg, beta_deg, gamma_deg] * pi/180;

% =========================================================================
% INIZIALIZZAZIONE E PRE-ALLOCAZIONE
% =========================================================================
% Pre-allocazione essenziale per non far crashare la RAM con N alti
B_labels = zeros(N_spectra, 1);
spectra_data = zeros(N_spectra, Exp.nPoints);
mw_freqs = zeros(1, Exp.nPoints); 

fprintf('Inizio simulazione di %d spettri...\n', N_spectra);

% =========================================================================
% LOOP DI GENERAZIONE
% =========================================================================
for i = 1:N_spectra
    % Estrazione randomica uniforme del campo magnetico
    B_rand = B_min + rand() * (B_max - B_min);
    Exp.Field = B_rand;
    
    % Simulazione EasySpin
    [x, y] = pepper(Sys, Exp, Opt);
    
    % Inversione del segno come da tuo codice originale
    ips = -y;
    
    % Salvataggio nei vettori
    B_labels(i) = B_rand;
    spectra_data(i, :) = ips;
    
    % Salviamo l'asse x (frequenze in GHz) solo alla prima iterazione
    if i == 1
        mw_freqs = x;
    end
    
    % Progress bar a terminale ogni 100 spettri
    if mod(i, 100) == 0
        fprintf('Generati %d/%d spettri...\n', i, N_spectra);
    end
end

% =========================================================================
% SALVATAGGIO FILE .MAT
% =========================================================================
% Genera il nome del file dinamicamente in base a N_spectra
nome_file = sprintf('~/ml-odmr/matlab-code/odmr_dataset_%d.mat', N_spectra);

save(nome_file, 'spectra_data', 'B_labels', 'mw_freqs');

fprintf('✅ Generazione completata e salvata in "%s"\n', nome_file);