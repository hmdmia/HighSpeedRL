function [ecefVec] = enuVec2ecefVec(enuVec,ecefPos)
%enuVec2ecefVec Transforms vector in ENU coordinate system to ECEF
%coordinate system
%   Takes in the current position in ECEF and the vector in ENU. enuVec is
%   a n x 3 matrix. The n x 3 vector in ECEF is output.
%
%   ENU is East-North-Up. The columns of enuVec must correspond to these
%   coordinates, in that order. ENU in this case is based in the spherical
%   coordinates, and assumes a spherical Earth, which is valid for the
%   hypersonic equations of motion.
%
%   ECEF is Earth-Centered, Earth-Fixed. The columns of ecefPos must
%   correspond to x, y, z coordinates, in that order.
%
%   The rows of ecefVec will have magnitudes equal to their corresponding
%   rows in enuVec

% TODO: Add option to give position in ENU instead of ECEF

ecefVec = zeros(3,length(enuVec(:,1)));

for i = 1:length(enuVec(:,1))
    % Get spherical angles
    [theta,phi] = cart2sph(ecefPos(i,1),ecefPos(i,2),ecefPos(i,3));

    % Transformation matrix
    trans = [ -sin(theta), -sin(phi)*cos(theta), cos(phi)*cos(theta);...
        cos(theta), -sin(phi)*sin(theta), cos(phi)*sin(theta);...
        0, cos(phi), sin(phi)];

    ecefVec(:,i) = trans*enuVec(i,:).';
end

ecefVec = ecefVec.';

end

