clear all; clc;

% =========================================================================
% CONFIGURAZIONE DISTRIBUZIONI FISICHE (SCALA ORIGINARIA DI YAO ET AL.)
% =========================================================================
N_spectra = 10000;

% Finestra di riferimento di Yao (usata esclusivamente per la mappa dei parametri fisici)
ref_mw_min_GHz = 2.80;
ref_mw_max_GHz = 2.92;
ref_mw_range_MHz = (ref_mw_max_GHz - ref_mw_min_GHz) * 1e3; % 120 MHz

% Costanti fisiche di laboratorio
E_MHz = 10.0;             % Anisotropia rombica
gamma_MHz_mT = 28.0;      % Costante giromagnetica

% Bound normalizzati da Yao et al.
c_min = 0.35;
c_max = 0.65;

% Splitting normalizzato s sulla scala di 120 MHz (s_min = 20 / 120 = 0.1667)
s_min = (2 * E_MHz) / ref_mw_range_MHz; 
s_max = 0.20;

% Larghezza lorenziana normalizzata w in [0.02, 0.09]
w_min = 0.02;
w_max = 0.09;

% =========================================================================
% CONFIGURAZIONE DI ACQUISIZIONE EASYSPIN (SWEEP ALLARGATO)
% =========================================================================
% Finestra estesa a 200 MHz per accogliere i 4 assi del cristallo
mw_min_GHz = 2.75;
mw_max_GHz = 2.95;
Exp.mwRange = [mw_min_GHz, mw_max_GHz]; 
Exp.nPoints = 167;                     % 167 punti -> ~1.20 MHz/punto
Exp.Harmonic = 0;

% Angoli di Eulero del campione determinati in laboratorio (gradi -> radianti)
alpha_deg = 36.1;        
beta_deg  = 42.39;         
gamma_deg = 0.61;        

Exp.CrystalSymmetry = 227;             % Gruppo spaziale reticolo diamante
Exp.MolFrame = [0, acos(1/sqrt(3)), 0]; % Orientazione assi <111>
Exp.SampleFrame = [alpha_deg, beta_deg, gamma_deg] * pi/180;
Opt.Sites = [];                         % Popola tutti i 4 siti equivalenti
Opt.Verbosity = 0;

% Setup di spin
Sys.S = 1;
Sys.g = 2.0028;

% =========================================================================
% PRE-ALLOCAZIONE
% =========================================================================
spectra_data = zeros(N_spectra, Exp.nPoints);
labels_B = zeros(N_spectra, 1);
labels_D = zeros(N_spectra, 1);
labels_lw = zeros(N_spectra, 1);
labels_c = zeros(N_spectra, 1);
labels_s = zeros(N_spectra, 1);
mw_freqs = zeros(1, Exp.nPoints);

fprintf('Generazione di %d spettri bulk con EasySpin (167 punti, 4 siti NV)...\n', N_spectra);

% =========================================================================
% LOOP DI GENERAZIONE
% =========================================================================
for i = 1:N_spectra
    % 1. Campionamento adimensionale sui bound di Yao
    c_rand = c_min + rand() * (c_max - c_min);
    s_rand = s_min + rand() * (s_max - s_min);
    w_rand = w_min + rand() * (w_max - w_min);
    
    % 2. Mappatura nelle grandezze fisiche (ancorata ai 120 MHz di riferimento)
    D_MHz = (ref_mw_min_GHz * 1e3) + c_rand * ref_mw_range_MHz;
    delta_MHz = s_rand * ref_mw_range_MHz;
    
    % Campo magnetico ricavato analiticamente
    B_mT = (1 / gamma_MHz_mT) * sqrt((delta_MHz / 2)^2 - E_MHz^2);
    lw_MHz = w_rand * ref_mw_range_MHz;
    
    % 3. Assegnazione a EasySpin
    Sys.D = [D_MHz, E_MHz];
    Sys.lw = [0, lw_MHz];
    Exp.Field = B_mT;
    
    % 4. Simulazione dello spettro con simmetria a 4 siti
    [freqs, y] = pepper(Sys, Exp, Opt);
    
    if max(y) > 0
        y = y / max(y);
    end
    
    % 5. Salvataggio
    spectra_data(i, :) = y;
    labels_B(i) = B_mT;
    labels_D(i) = D_MHz;
    labels_lw(i) = lw_MHz;
    labels_c(i) = c_rand;
    labels_s(i) = s_rand;
    
    if i == 1
        mw_freqs = freqs;
    end
    
    if mod(i, 500) == 0
        fprintf('Completati %d/%d spettri\n', i, N_spectra);
    end
end

% =========================================================================
% SALVATAGGIO MAT-FILE
% =========================================================================
nome_file = sprintf('odmr_easyspin_raw_%d.mat', N_spectra);
save(nome_file, 'spectra_data', 'labels_B', 'labels_D', 'labels_lw', 'labels_c', 'labels_s', 'mw_freqs', '-v7.3');
fprintf('✅ File salvato con successo: %s\n', nome_file);