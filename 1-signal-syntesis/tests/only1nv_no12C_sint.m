clear all; clc;

% =========================================================================
% CONFIGURAZIONE SECONDO YAO ET AL.
% Praticamente come fanno loro:
% - 1 solo asse, quindi solo un centro NV
% - Finestra di 120 MHz, quindi niente iperfina del carbonio
% =========================================================================
N_spectra = 10000;

% Finestra sperimentale di sweep
mw_min_GHz = 2.80;
mw_max_GHz = 2.92;
mw_range_MHz = (mw_max_GHz - mw_min_GHz) * 1e3; % 120 MHz

% Costanti fisiche di laboratorio
E_MHz = 10.0;             % Anisotropia rombica
gamma_MHz_mT = 28.0;      % Costante giromagnetica

% Bound normalizzati da Yao et al.
c_min = 0.35;
c_max = 0.65;

% Splitting normalizzato s:
% Poiché E = 10 MHz, lo splitting minimo fisico a B=0 è 2*E = 20 MHz.
% Normalizzato: s_min = 20 / 120 = 0.1667
% s_max = 0.20 come definito in Yao et al.
s_min = (2 * E_MHz) / mw_range_MHz; 
s_max = 0.20;

% Larghezza lorenziana normalizzata w in [0.02, 0.09]
w_min = 0.02;
w_max = 0.09;

% =========================================================================
% CONFIGURAZIONE EASYSPIN
% =========================================================================
Sys.S = 1;
Sys.g = 2.0028;

% Sweep continuo in frequenza: impostare SOLO mwRange
Exp.mwRange = [mw_min_GHz, mw_max_GHz]; 
Exp.nPoints = 101;
Exp.Harmonic = 0;
Opt.Verbosity = 0;

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

fprintf('Generazione di %d spettri con EasySpin...\n', N_spectra);

% =========================================================================
% LOOP DI GENERAZIONE
% =========================================================================
for i = 1:N_spectra
    % 1. Campionamento adimensionale (Yao et al.)
    c_rand = c_min + rand() * (c_max - c_min);
    s_rand = s_min + rand() * (s_max - s_min);
    w_rand = w_min + rand() * (w_max - w_min);
    
    % 2. Mappatura nelle grandezze fisiche
    D_MHz = (mw_min_GHz * 1e3) + c_rand * mw_range_MHz;
    delta_MHz = s_rand * mw_range_MHz;
    
    % Formula inversa per ricavare B dallo splitting
    B_mT = (1 / gamma_MHz_mT) * sqrt((delta_MHz / 2)^2 - E_MHz^2);
    lw_MHz = w_rand * mw_range_MHz;
    
    % 3. Assegnazione a EasySpin
    Sys.D = [D_MHz, E_MHz];
    Sys.lw = [0, lw_MHz];
    Exp.Field = B_mT;
    
    % 4. Simulazione dello spettro di assorbimento
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
    
    if mod(i, 1000) == 0
        fprintf('Completati %d/%d spettri\n', i, N_spectra);
    end
end

% =========================================================================
% SALVATAGGIO MAT-FILE
% =========================================================================
nome_file = sprintf('odmr_easyspin_raw_%d.mat', N_spectra);
save(nome_file, 'spectra_data', 'labels_B', 'labels_D', 'labels_lw', 'labels_c', 'labels_s', 'mw_freqs', '-v7.3');
fprintf('✅ File salvato con successo: %s\n', nome_file);