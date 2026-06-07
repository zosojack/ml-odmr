clear all; clc; close all;

% =========================================================================
% PARAMETRI DA MODIFICARE AL VOLO
% =========================================================================
B_test = 4.3;             % Campo magnetico in mT
alpha_deg = 36.1;        % Angolo di Eulero Alpha in gradi
beta_deg = 42.39;         % Angolo di Eulero Beta in gradi
gamma_deg = 0.61;        % Angolo di Eulero Gamma in gradi
% =========================================================================

% ----- Setup del Sistema di Spin (Parametri fissi) -----
Sys.S = 1;
Sys.g = 2.0;
Sys.D = [2870 10];         % Zero-field splitting (D e E) in MHz
Sys.Nucs = 'C';
Sys.A = [127, 127, 127];   
Sys.lwpp = [16 4];              % Larghezza riga (tiene i picchi uniti senza iperfine)

% ----- Parametri Sperimentali -----
Exp.mwRange = [2.5 3.3];  % Range in GHz (2500 - 3250 MHz)
Exp.Harmonic = 0;
Exp.CrystalSymmetry = 227; % Simmetria del diamante
Opt.Sites = [];            % Calcola tutti e 4 i siti NV simultaneamente
Exp.nPoints = 750;         % Numero di bin identico al tuo dataset Python

% Applicazione dei parametri di test
Exp.Field = B_test;
Exp.MolFrame =  [0, acos(1/sqrt(3)), 0]; %[45 54.73561 0]*pi/180;
Exp.SampleFrame = [alpha_deg, beta_deg, gamma_deg] * pi/180;

% ----- Simulazione EasySpin -----
fprintf('Simulazione in corso... B = %.1f mT, Angoli = [%.1f, %.1f, %.1f]°\n', ...
    B_test, alpha_deg, beta_deg, gamma_deg);

[x_freq, y_raw] = pepper(Sys, Exp, Opt);

% ----- Grafico Rapido del Dato RAW -----
figure('Name', 'EasySpin ODMR Quick Check', 'NumberTitle', 'off', 'Position', [100, 100, 800, 500]);
plot(x_freq * 1000, y_raw, 'LineWidth', 2, 'Color', [0 0.4470 0.7410]); % Asse x convertito in MHz

% Estetica e Leggenda
title(sprintf('Spettro ODMR RAW (B = %.1f mT)', B_test), 'FontSize', 12);
subtitle(sprintf('Angoli di Eulero: \\alpha=%.2f°, \\beta=%.2f°, \\gamma=%.2f°', ...
    alpha_deg, beta_deg, gamma_deg), 'FontSize', 10);
xlabel('Frequenza (MHz)', 'FontSize', 11);
ylabel('Intensità di Assorbimento (U.A.)', 'FontSize', 11);
grid on;

% Linea verticale di riferimento allo Zero-Field Splitting standard (2870 MHz)
xline(2870, '--r', 'D = 2870 MHz', 'LabelVerticalAlignment', 'bottom', 'LineWidth', 1.2);

fprintf('Fatto! max(y) = %e, min(y) = %e\n', max(y_raw), min(y_raw));




%alpha_deg = 48.62;        % Angolo di Eulero Alpha in gradi
%beta_deg = 42.39;         % Angolo di Eulero Beta in gradi
%gamma_deg = 0.61;        % Angolo di Eulero Gamma in gradi