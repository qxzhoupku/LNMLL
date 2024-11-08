clear;

% Key Parameters
the_FSR = 25e9;
the_M = 1.0;
the_P_pump = 20e-3;


% Constants
c0 = 299792458;
h = 6.62607015e-34;  % Planck constant
hbar = h / (2*pi);   % Reduced Planck constant
k_B = 1.380649e-23;  % Boltzmann constant

% Erbium Lasing Parameters
N = 3.959e26; % Er ion density (m^-3)
sigma_sa = 4.03e-25; % Signal absorption cross section (m^2)
sigma_se = 6.76e-25; % Signal emission cross section (m^2)
sigma_pa = 4.48e-25; % Pump absorption cross section (m^2)
sigma_pe = 1.07e-25; % Pump emission cross section (m^2)
tau_g = 10e-3; % Upper level lifetime (s)
lambda_s = 1550e-9; % Signal wavelength (m)
lambda_p = 1480e-9; % Pump wavelength (m)
lambda_g = 30e-9; % Gain bandwidth (m)
nu_s = c0 / lambda_s;
nu_p = c0 / lambda_p;
omega_p = 2 * pi * nu_p;
omega_s = 2 * pi * nu_s;
A_s = 0.9e-12; % Signal effective mode area (m^2)
A_p = 0.9e-12; % Pump effective mode area (m^2)
T = 300; % Temperature (K)
Gamma_s = 0.9; % Signal mode overlap with Er ions
Gamma_p = 0.9; % Pump mode overlap with Er ions
beta = exp(-1.0 / k_B / T * h * c0 * (1.0/lambda_p - 1.0/lambda_s));

% Micro-cavity Parameters
beta2 = -58e-27; % Dispersion (s^2/m)
n2 = 1.8e-19; % Kerr coefficient of LiNbO3 (m^2/W)
FSR = the_FSR;
L_d = 5e-3 * 25e9 / FSR; % Cavity length (m)
T_R = 1 / FSR; % Roundtrip time (s)
omega_m = 2 * pi * FSR;
Omega_g = 2 * pi * c0 / lambda_s^2 * lambda_g; % Gain FWHM (rad)
Q_s_in = 2e6; % Intrinsic Q of cavity
Q_s_ex = 2e6; % Coupling Q of cavity
Q_s = 1 / (1 / Q_s_in + 1 / Q_s_ex); % Total Q of cavity
Q_p_in = 2e6;
Q_p_ex = 2e6;
Q_p = 1.0 / (1.0 / Q_p_in + 1.0 / Q_p_ex);
D = -0.5 * beta2 * L_d;
delta_kerr = n2 * omega_s * L_d / (c0 * A_s);
phi_opt = 0.0; % Pump detuning
gamma = 0; % Loss at the waveguide-resonator coupling
k = 2 * pi * omega_p / omega_m / Q_p_ex;
k_s = 2 * pi * omega_s / omega_m / Q_s_ex;
total_loss = 2 * pi * omega_p / omega_m / Q_p;
l_p_in = pi * omega_p / omega_m / Q_p_in;
l_s = pi * omega_s / omega_m / Q_s;

mode_number = 2^10;

t = linspace(-T_R/2, T_R/2 - T_R/mode_number, mode_number);
delta_t = T_R / mode_number;
q = linspace(-mode_number/2, mode_number/2 - 1, mode_number);
xs = linspace(-pi, pi - 2*pi/mode_number, mode_number);

phi_disp = ifftshift(2e-5 * q.^2); % Dispersion
q_ishift = ifftshift(q);

M = the_M;
P_pump = the_P_pump;
prompt = sprintf('P_pump=%gmW, M=%g, FSR=%gGHz', P_pump*1000, M, FSR/1e9);
disp(['prompt: ', prompt]);

% Time normalization to T_R
scale = 1; % Number of roundtrip times for each save
steps = 1;
eta = 1 / steps;
save_round = 100000;
plot_round = min(round(save_round / 6), 50000);
plot_round = save_round;
begin_to_save = 0;

% Display simulation parameters
disp(['steps = ', num2str(steps)]);
disp(['save_round = ', num2str(save_round)]);

% Functions (example for roundtrip_evolution_for_EO_comb)
function new_field = roundtrip_evolution_for_EO_comb(E_np, loss, M, t, omega_m, k, P_pump, phi_opt, phi_disp)
    spectrum = fft(E_np);
    phi_micro = 0.0;
    new_spectrum = spectrum .* exp(-1.0j * (phi_opt + phi_micro + phi_disp));
    field = ifft(new_spectrum);
    new_field = exp(-loss) .* (sqrt(1 - k) * field + 1.0j * sqrt(k * P_pump) * exp(-1.0j * phi_opt)) .* exp(1.0j * M * cos(omega_m * t));
end

function A_spectrum = roundtrip_evolution_for_signal(A, loss, gain, delta_kerr, steps, M, D, xs, q_ishift, omega_m, Omega_g)
    eta = 1 / steps;
    for k = 1:steps
        A = A .* exp((-loss + 1.0j * delta_kerr * abs(A).^2 + 1.0j * M * cos(xs)) * eta);
        A_spectrum = fft(A);
        r = -1.0j * D * (q_ishift * omega_m).^2 + gain ./ (1 + (omega_m / Omega_g * q_ishift).^2);
        A_spectrum = A_spectrum .* exp(eta * r);
    end
end

function g_next = next_g(g, g_0, signal_power, p_sat, tau_prime, T_R)
    g_limit = g_0 / (1 + signal_power / p_sat);
    delta_g = (g_0 - (1 + signal_power / p_sat) * g) * T_R / tau_prime;
    if delta_g == 0
        g_next = g;
    elseif delta_g > 0 && g + delta_g > g_limit
        g_next = g_limit;
    elseif delta_g < 0 && g + delta_g < g_limit
        g_next = g_limit;
    else
        g_next = g + delta_g;
    end
end

function spectrum_modified = ASE(A_spectrum, g, l, N, Gamma_s, sigma_se, nu_s, FSR, q_ishift, beta, sigma_sa, h, L_d)
    N_2 = (N + (2 * g) / (Gamma_s * sigma_se * nu_s)) / (1 + beta + sigma_se / sigma_sa);
    alpha = max(h * FSR * sigma_se * Gamma_s * N_2 * L_d * (exp(2 * g) - 1) / (g - l), 0);
    ase_spectrum = sqrt(alpha * (nu_s + FSR * q_ishift));
    ase_spectrum_modified = ase_spectrum .* exp(1.0j * rand(size(A_spectrum)) * 2 * pi);
    spectrum_modified = A_spectrum + ase_spectrum_modified;
end

function [pump_power, signal_power, rsignal_power, tau_prime, p_sat, g_0, l_p] = parameter_calculation(E_p, A, g, delta_t, T_R, k_B, T, h, nu_p, A_p, sigma_sa, sigma_se, tau_g, Gamma_p, sigma_pa, sigma_pe, Gamma_s, lambda_p, lambda_s, FSR, N, L_d, c0, beta, Q_p_in)
    pump_power = sum(abs(E_p).^2) * delta_t / T_R;
    signal_power = sum(abs(A).^2) * delta_t / T_R;
    rsignal_power = signal_power;
    tau_prime = (1 + beta) / (1 / tau_g + (1 + beta + beta * sigma_pe / sigma_pa) * pump_power * sigma_pa * Gamma_p / h / nu_p / A_p);
    p_sat = h * nu_p * A_p / (Gamma_s * tau_prime * (sigma_sa + sigma_se / (1 + beta)));
    g_0 = 0.5 * Gamma_s * L_d * sigma_se * N * tau_prime / (1 + beta) * ((1 - sigma_sa / sigma_se * beta * sigma_pe / sigma_pa) * sigma_pa * Gamma_p / h / nu_p / A_p * pump_power - sigma_sa / sigma_se / tau_g);
    g_p = 0.5 * Gamma_p * L_d * N * (beta * sigma_pe * sigma_sa - sigma_pa * sigma_se) / (sigma_se + sigma_sa * (1 + beta)) + (sigma_pe * beta + sigma_pa * (1 + beta)) / (sigma_se * beta + sigma_sa * (1 + beta)) * g;
    l_p = pi * nu_p / FSR / Q_p_in - g_p;
end


% Initialize fields
E_0p = zeros(1, mode_number) + 1.0j * sqrt(k * P_pump) / (exp(total_loss/2) - sqrt(1-k));
A_0 = (rand(1, mode_number) + 1.0j*rand(1, mode_number)) * 1e-3;

% Main loop
A_save = zeros(plot_round, mode_number);
g_save = zeros(1, plot_round);
E_p_save = zeros(plot_round, mode_number);
[pump_power, signal_power, rsignal_power, tau_prime, p_sat, g_0, l_p] = parameter_calculation(E_0p, A_0, 0, delta_t, T_R, k_B, T, h, nu_p, A_p, sigma_sa, sigma_se, tau_g, Gamma_p, sigma_pa, sigma_pe, Gamma_s, lambda_p, lambda_s, FSR, N, L_d, c0, beta, Q_p_in);
E_p = E_0p;
A = A_0;
g = l_s; % Initialize gain

disp("start simulation")

for round_index = 1:save_round
    for jjj = 1:scale
        [pump_power, signal_power, rsignal_power, tau_prime, p_sat, g_0, l_p] = parameter_calculation(E_p, A, g, delta_t, T_R, k_B, T, h, nu_p, A_p, sigma_sa, sigma_se, tau_g, Gamma_p, sigma_pa, sigma_pe, Gamma_s, lambda_p, lambda_s, FSR, N, L_d, c0, beta, Q_p_in);
        g = next_g(g, g_0, signal_power, p_sat, tau_prime, T_R);
        A_spectrum = roundtrip_evolution_for_signal(A, l_s, g, delta_kerr, steps, M, D, xs, q_ishift, omega_m, Omega_g);
        % A_spectrum = ASE(A_spectrum, g, l_s, N, Gamma_s, sigma_se, nu_s, FSR, q_ishift, beta, sigma_sa, h, L_d);
        A = ifft(A_spectrum);
        E_p = roundtrip_evolution_for_EO_comb(E_p, l_p, M, t, omega_m, k, P_pump, phi_opt, phi_disp);
    end
    % fprintf('Process: %.2f%%, g = %.6f, pump_power = %.6f, signal_power = %.6f, p_sat = %.6f, l_p = %.6f, l_s = %.6f\r', round_index / save_round * 100, g, pump_power, signal_power, p_sat, l_p, l_s);
    if round_index > save_round - plot_round
        index = round_index - save_round + plot_round;
        A_save(index, :) = A;
        g_save(index) = g;
        E_p_save(index, :) = E_p;
    end
end

disp("end simulation")

% Calculate the average and peak power for the EO comb and signal pulse
EO_comb_average_power = sum(abs(E_p_save(end,:)).^2) / mode_number * k;
EO_comb_peak_power = max(abs(E_p_save(end,:)).^2) * k;
signal_average_power = sum(abs(A_save(end,:)).^2) / mode_number * k_s;
signal_peak_power = max(abs(A_save(end,:)).^2) * k_s;

% Display the results
fprintf('EO comb average power = %.3f mW\n', EO_comb_average_power * 1e3);
fprintf('EO comb peak power = %.3f mW\n', EO_comb_peak_power * 1e3);
fprintf('Signal pulse average power = %.3f mW\n', signal_average_power * 1e3);
fprintf('Signal pulse peak power = %.3f mW\n', signal_peak_power * 1e3);
fprintf('Conversion efficiency = %.3f %%\n', signal_average_power / P_pump * 100);
fprintf('Pump power coupling loss (k) = %.3f\n', k);
fprintf('Pump power intrinsic loss (alpha) = %.3f\n', 2 * pi * omega_p / Q_p_in / omega_m);
fprintf('Pump power total loss (l_s) = %.3f\n', total_loss);
fprintf('Signal power coupling loss (k'') = %.3f\n', k_s);
fprintf('Signal power intrinsic loss (alpha'') = %.3f\n', 2 * pi * omega_s / Q_s_in / omega_m);
fprintf('Signal power total loss (l_s'') = %.3f\n', 2 * pi * omega_s / Q_s / omega_m);



A_magnitude = abs(A_save).^2;
g_data = real(g_save);
pump_power_data = abs(E_p_save).^2;


figure;

subplot(3,1,1);
plot(1:plot_round, g_data, 'LineWidth', 1.5);
title('Gain Evolution');
xlim([1, plot_round]);
xlabel('Roundtrip Number');
ylabel('Gain');
% grid on;

subplot(3,1,2);
imagesc(1:plot_round, 1:mode_number, A_magnitude');
title('Signal Field Intensity');
xlabel('Roundtrip Number');
ylabel('Mode Number');
% colorbar;
% colormap jet;

subplot(3,1,3);
imagesc(1:plot_round, 1:mode_number, pump_power_data');
title('Pump Power Evolution');
xlabel('Roundtrip Number');
ylabel('Pump Power (W)');
% grid on;
