% NGA MATLAB_SAR polar format (pfa_mem, unmodified) on the Panama phase history; writes the image and its grid.
%   octave nga_pfa_run.m <work dir> <MATLAB_SAR dir> <pulses> <samples>
args = argv(); W = args{1}; addpath(genpath(args{2})); if numel(args) > 4, addpath(args{5}); end
P = str2double(args{3}); K = str2double(args{4});
f = fopen([W '/pfa_ph.c64'], 'r'); x = fread(f, [2*K, P], 'single=>single'); fclose(f);
ph = complex(x(1:2:end, :), x(2:2:end, :)); clear x
f = fopen([W '/pfa_nb.f64'], 'r'); nbm = fread(f, [P, 11], 'double'); fclose(f);
nb.TxPos = nbm(:, 1:3); nb.RcvPos = nbm(:, 4:6); nb.SRPPos = nbm(:, 7:9); nb.SC0 = nbm(:, 10); nb.SCSS = nbm(:, 11);
t = tic; img = pfa_mem(ph, nb, 1.5); secs = toc(t);
printf('pfa_mem: %.1f s, image %d x %d, class %s\n', secs, size(img, 1), size(img, 2), class(img));
% the grid pfa_mem used (its own first lines, repeated)
scp = nb.SRPPos(1, :);
[bi_pos, bi_fs] = pfa_bistatic_pos(nb.TxPos, nb.RcvPos, nb.SRPPos);
fpn = wgs_84_norm(scp).'; ref_pulse = ceil(P/2);
arp_coa_vel = diff(bi_pos(ref_pulse + [1 -1], :)); arp_coa = bi_pos(ref_pulse, :);
srv = arp_coa.'; look = fpn*cross(srv, arp_coa_vel).'; ipn = look*cross(srv, arp_coa_vel); ipn = ipn/norm(ipn);
[k_a, k_sf] = pfa_polar_coords(bi_pos, [0 0 0], arp_coa, ipn, fpn);
rf = (2/SPEED_OF_LIGHT) .* k_sf .* bi_fs; k_r0 = nb.SC0 .* rf; k_r_ss = nb.SCSS .* rf;
[kv, ku] = pfa_inscribed_rectangle_coords(k_a, k_r0, k_r_ss * (K - 1));
f = fopen([W '/pfa_grid.txt'], 'w'); fprintf(f, '%.17g ', [secs, size(img), fpn(:).', ipn(:).', arp_coa(:).', kv(:).', ku(:).', K, P]); fclose(f);
f = fopen([W '/pfa_img.c64'], 'w'); y = zeros(2*size(img, 1), size(img, 2), 'single'); y(1:2:end, :) = real(img); y(2:2:end, :) = imag(img); fwrite(f, y, 'single'); fclose(f);
