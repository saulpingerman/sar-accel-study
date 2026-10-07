function c = cross(a, b, dim)
  % Octave compatibility for MATLAB_SAR. Two 3-element vectors: the cross product, a row vector when the inputs
  % differ in orientation (as the MATLAB_SAR code expects) and the orientation of the inputs otherwise. Arrays: the
  % cross product along dimension dim (default: the first of size 3).
  if numel(a) == 3 && numel(b) == 3 && isvector(a) && isvector(b)
    x = a(:).'; y = b(:).';
    c = [x(2)*y(3) - x(3)*y(2), x(3)*y(1) - x(1)*y(3), x(1)*y(2) - x(2)*y(1)];
    if all(size(a) == size(b)) && size(a, 1) == 3
      c = c.';
    end
    return
  end
  if nargin < 3
    dim = find(size(a) == 3, 1);
  end
  p = [dim, setdiff(1:ndims(a), dim)];
  A = permute(a, p); B = permute(b, p); s = size(A);
  A = reshape(A, 3, []); B = reshape(B, 3, []);
  C = [A(2,:).*B(3,:) - A(3,:).*B(2,:); A(3,:).*B(1,:) - A(1,:).*B(3,:); A(1,:).*B(2,:) - A(2,:).*B(1,:)];
  c = ipermute(reshape(C, s), p);
end
