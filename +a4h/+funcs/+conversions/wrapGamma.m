function [gamma] = wrapGamma(gamma)

gamma = wrapToPi(gamma);

gamma(gamma>pi/2) = pi - gamma(gamma>pi/2);
gamma(gamma<-pi/2) = -pi - gamma(gamma<-pi/2);

end