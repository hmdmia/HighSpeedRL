# https://nebula.wsimg.com/ab321c1edd4fa69eaa94b5e8e769b113?AccessKeyId=AF1D67CEBF3A194F66A3&disposition=0&alloworigin=1

from math import exp
from bisect import bisect

gas_constant = 287.053
g0 = 9.80665
r = 6.356766e6


class StandardAtmosphere:
    """
    Class that computes standard atmosphere values
    """
    def __init__(self):
        self.alt_layers = [0, 11e3, 20e3, 32e3, 47e3, 51e3, 71e3, 84.8520e3]
        self.lam_layers = [-6.5e-3, 0, 1e-3, 2.8e-3, 0, -2.8e-3, -2e-3, 0]

        self.temp_layers = [288.16]
        self.pres_layers = [101325]
        self.dens_layers = [1.22500]

        for alt_0, alt_1, lam in zip(self.alt_layers[:-1], self.alt_layers[1:], self.lam_layers):
            if lam == 0:
                temp, pres, dens = self._compute_iso(
                    alt_1, alt_0, self.temp_layers[-1], self.pres_layers[-1], self.dens_layers[-1])

                self.temp_layers.append(temp)
                self.pres_layers.append(pres)
                self.dens_layers.append(dens)

            else:
                temp, pres, dens = self._compute_grad(
                        alt_1, alt_0, lam, self.temp_layers[-1], self.pres_layers[-1], self.dens_layers[-1])

                self.temp_layers.append(temp)
                self.pres_layers.append(pres)
                self.dens_layers.append(dens)

    @staticmethod
    def hg_to_h(hg):
        return r/(r + hg)*hg

    @staticmethod
    def _compute_iso(h, h0, temp0, p0, rho0):
        k = exp(-g0/(gas_constant * temp0) * (h - h0))
        p = p0 * k
        rho = rho0 * k

        return temp0, p, rho

    @staticmethod
    def _compute_grad(h, h0, lam, temp0, p0, rho0):
        temp = temp0 + lam * (h - h0)
        p = p0 * (temp / temp0) ** (-g0/(lam * gas_constant))
        rho = rho0 * (temp / temp0) ** (-1 - g0/(lam * gas_constant))

        return temp, p, rho

    def calc_values(self, hg):

        h = self.hg_to_h(hg)

        layer = bisect(self.alt_layers[1:], h)

        if layer >= len(self.lam_layers):
            layer = -1

        lam = self.lam_layers[layer]
        h0 = self.alt_layers[layer]
        temp0 = self.temp_layers[layer]
        p0 = self.pres_layers[layer]
        rho0 = self.dens_layers[layer]

        if lam == 0:
            return self._compute_iso(h, h0, temp0, p0, rho0)

        else:
            return self._compute_grad(h, h0, lam, temp0, p0, rho0)


gamma = 1.4


def calc_speed_of_sound(temp):
    return (gamma * gas_constant * temp)**0.5


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    atmos = StandardAtmosphere()

    test_h = np.linspace(0, 80e3, 500)
    test_temp = []
    test_p = []
    test_rho = []

    for _h in test_h:
        _temp, _p, _rho = atmos.calc_values(_h)
        test_temp.append(_temp)
        test_p.append(_p)
        test_rho.append(_rho)

    plt.figure()

    plt.subplot(1, 3, 1)
    plt.plot(test_temp, test_h)
    plt.xlabel('Temperature [K]')
    plt.ylabel('Altitude [m]')

    plt.subplot(1, 3, 2)
    plt.plot(test_p, test_h)
    plt.xlabel('Pressure [N/m^2]')
    plt.ylabel('Altitude [m]')

    plt.subplot(1, 3, 3)
    plt.plot(test_rho, test_h)
    plt.xlabel('Density [kg/m^3]')
    plt.ylabel('Altitude [m]')

    plt.show()

