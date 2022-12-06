from math import sin, cos, asin, sqrt, atan2, pi


def wrap_ang(ang):
    """
    Wraps angles to fall betweem -pi and pi
    :param ang:
    :return:
    """
    return (ang + pi) % (2 * pi) - pi


def hav(theta):
    """
    Haversine function
    :param theta:
    :return: haversine(theta)
    """
    return (1 - cos(theta))/2


def ahav(x):
    """
    Inverse haversine function
    :param x:
    :return:
    """
    return 2*asin(sqrt(x))


def circle_ang_dist(lat1, long1, lat2, long2):
    """
    Calculate angular distance between two lat-lon coordinates
    :param lat1: latitude for first point
    :param long1: longitude for first point
    :param lat2: latitude for second point
    :param long2: longitude for second point
    :return: angular distance
    """
    hav_theta = hav(lat2 - lat1) + cos(lat1)*cos(lat2)*hav(long2 - long1)
    return ahav(hav_theta)


def calc_bearing(lat1, long1, lat2, long2):
    """
    Calculate bearing angle from first coordinates to second coordinates
    measured eastward from north.
    :param lat1: latitude for first point
    :param long1: longitude for first point
    :param lat2: latitude for second point
    :param long2: longitude for second point
    :return: bearing angle
    """
    dlong = long2 - long1
    x = cos(lat2) * sin(dlong)
    y = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlong)
    return atan2(x, y)

