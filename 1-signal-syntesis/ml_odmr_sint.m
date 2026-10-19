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
N_points = 334+1
Exp.nPoints = N_points; % 167 punti -> ~1.20 MHz/punto
% 334 raddoppio + 1 per simmetria 
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

fprintf('Generazione di %d spettri bulk con EasySpin (%d punti, 4 siti NV)...\n', N_points, N_spectra);

% =========================================================================
% LOOP DI GENERAZIONE
% =========================================================================
for i = 1:N_spectra
    % 1. Campionamento adimensionale sui bound di Yao
    c_rand = c_min + rand() * (c_max - c_min);
    s_rand = s_min + rand() * (s_max - s_min);
    w_rand = w_min + rand() * (w_max - w_min);
    
    % 2. Mappatura nelle grandezze fisiche (scala di riferimento 120 MHz)
    D_MHz = (ref_mw_min_GHz * 1e3) + c_rand * ref_mw_range_MHz;
    delta_MHz = s_rand * ref_mw_range_MHz;
    B_mT = (1 / gamma_MHz_mT) * sqrt((delta_MHz / 2)^2 - E_MHz^2);
    
    % Correzione HWHM -> FWHM (fattore 2 per EasySpin)
    fwhm_MHz = 2 * w_rand * ref_mw_range_MHz; 
    
    % 3. Assegnazione a EasySpin
    Sys.D = [D_MHz, E_MHz];
    Sys.lw = [0, fwhm_MHz];
    Exp.Field = B_mT;
    
    % 4. Simulazione dello spettro
    [freqs, y] = pepper(Sys, Exp, Opt);
    
    if max(y) > 0
        y = y / max(y);
    end
    
    % 5. Riproiezione coerente di c ed s sulla finestra effettiva di sweep (2.75 - 2.95 GHz)
    % mw_range_eff = 200 MHz
    mw_range_eff_MHz = (mw_max_GHz - mw_min_GHz) * 1e3;
    nu_minus = D_MHz - delta_MHz / 2;
    nu_plus  = D_MHz + delta_MHz / 2;
    
    x1_eff = (nu_minus - mw_min_GHz * 1e3) / mw_range_eff_MHz;
    x2_eff = (nu_plus  - mw_min_GHz * 1e3) / mw_range_eff_MHz;
    
    c_eff = (x1_eff + x2_eff) / 2;
    s_eff = x2_eff - x1_eff;
    
    % Salvataggio
    spectra_data(i, :) = y;
    labels_B(i)  = B_mT;
    labels_D(i)  = D_MHz;
    labels_lw(i) = fwhm_MHz;
    labels_c(i)  = c_eff;       % Normalizzato in [0, 1] sui 200 MHz effettivi
    labels_s(i)  = s_eff;       % Normalizzato in [0, 1] sui 200 MHz effettivi
    
    if i == 1
        mw_freqs = freqs;
    end
end

% =========================================================================
% SALVATAGGIO MAT-FILE
% =========================================================================
nome_file = sprintf('odmr_easyspin_raw_%d.mat', N_spectra);
save(nome_file, 'spectra_data', 'labels_B', 'labels_D', 'labels_lw', 'labels_c', 'labels_s', 'mw_freqs', '-v7.3');
fprintf('✅ File salvato con successo: %s\n', nome_file);