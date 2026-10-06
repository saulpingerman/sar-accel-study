% NGA MATLAB_SAR bpBasic (Gorham & Moore) on raw-binary inputs.   octave nga_run.m <work dir> <case> <nfft or 0>
args = argv(); W = args{1}; name = args{2}; nfft = str2double(args{3});
addpath(genpath([W '/MATLAB_SAR/Processing/IFP/BP']));
meta = jsondecode(fileread([W '/' name '_meta.json']));
P = meta.P; K = meta.K; N = meta.N;
f = fopen([W '/' name '_ph.f32']); A = fread(f, [2 * K, P], 'single=>single'); fclose(f);
data.phdata = double(complex(A(1:2:end, :), A(2:2:end, :))); clear A;      % fast time in rows, slow time in columns
f = fopen([W '/' name '_ant.f64']); ant = fread(f, [3, P], 'double'); fclose(f);
f = fopen([W '/' name '_r0.f64']); data.R0 = fread(f, [P, 1], 'double'); fclose(f);
f = fopen([W '/' name '_pix.f64']); pix = fread(f, [3, N], 'double'); fclose(f);
data.Tx.X = ant(1, :); data.Tx.Y = ant(2, :); data.Tx.Z = ant(3, :);
data.minF = repmat(meta.minF, 1, P); data.deltaF = repmat(meta.deltaF, 1, P);
data.x_mat = pix(1, :); data.y_mat = pix(2, :); data.z_mat = pix(3, :);
data.hide_waitbar = true;
if nfft > 0, data.Nfft = nfft; end
tic; img = bpBasic(data); secs = toc;
out = zeros(2, N, 'single'); out(1, :) = real(img); out(2, :) = imag(img);
f = fopen(sprintf('%s/%s_nga_%d.f32', W, name, nfft), 'w'); fwrite(f, out, 'single'); fclose(f);
f = fopen(sprintf('%s/%s_nga_%d.txt', W, name, nfft), 'w'); fprintf(f, '%f\n', secs); fclose(f);
printf('%s nfft %d: %.1f s for %d pixels and %d pulses\n', name, nfft, secs, N, P);
