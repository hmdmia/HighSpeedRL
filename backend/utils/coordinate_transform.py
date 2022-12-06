import numpy as np

RADIUS_EARTH = 6378000


def lla2cart(lat, long, alt, radius_earth=RADIUS_EARTH):
    r = alt + radius_earth

    x = r * np.cos(lat) * np.cos(long)

    y = r * np.cos(lat) * np.sin(long)

    z = r * np.sin(lat)

    return x, y, z


def cart2lla(x, y, z, radius_earth=RADIUS_EARTH):
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    alt = r - radius_earth

    lat = np.arcsin(z / r)

    long = np.arctan2(x, y)

    return lat, long, alt


def lla_dist(lat1, long1, alt1, lat2, long2, alt2, radius_earth=RADIUS_EARTH):
    x1, y1, z1 = lla2cart(lat1, long1, alt1, radius_earth=radius_earth)

    x2, y2, z2 = lla2cart(lat2, long2, alt2, radius_earth=radius_earth)

    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
