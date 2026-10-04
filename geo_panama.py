"""Pixel-to-ground mapping for the Panama image from its SICD header, and the reverse.

Image pixel (i, j) (azimuth, range) of the images formed here lies at (i - nx/2) spx e1 + (j - ny/2) spy e2 from
the scene reference point in the slant plane; a ground point is the intersection of the line through that slant-plane
point along the plane's normal with the WGS84 ellipsoid raised by the scene height (a first-order layover model).
"""
import re
import xml.etree.ElementTree as ET
import numpy as np

A, F = 6378137.0, 1 / 298.257223563
E2 = F * (2 - F)


def ecf_to_lla(p):
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    lon = np.arctan2(y, x)
    r = np.hypot(x, y)
    lat = np.arctan2(z, r * (1 - E2))
    for _ in range(6):
        n = A / np.sqrt(1 - E2 * np.sin(lat) ** 2)
        h = r / np.cos(lat) - n
        lat = np.arctan2(z, r * (1 - E2 * n / (n + h)))
    n = A / np.sqrt(1 - E2 * np.sin(lat) ** 2)
    h = r / np.cos(lat) - n
    return np.degrees(lat), np.degrees(lon), h


def lla_to_ecf(lat, lon, h):
    lat, lon = np.radians(lat), np.radians(lon)
    n = A / np.sqrt(1 - E2 * np.sin(lat) ** 2)
    return np.stack([(n + h) * np.cos(lat) * np.cos(lon), (n + h) * np.cos(lat) * np.sin(lon), (n * (1 - E2) + h) * np.sin(lat)], -1)


class Geo:
    def __init__(self, xml_path, nx, ny):
        r = ET.fromstring(re.sub(r'xmlns="[^"]+"', '', open(xml_path).read(), count=1))
        g = lambda p: float(r.find(p).text)
        self.scp = np.array([g('GeoData/SCP/ECF/X'), g('GeoData/SCP/ECF/Y'), g('GeoData/SCP/ECF/Z')])
        self.hae = g('GeoData/SCP/LLH/HAE')
        self.urow = np.array([g('Grid/Row/UVectECF/X'), g('Grid/Row/UVectECF/Y'), g('Grid/Row/UVectECF/Z')])
        self.ucol = np.array([g('Grid/Col/UVectECF/X'), g('Grid/Col/UVectECF/Y'), g('Grid/Col/UVectECF/Z')])
        self.row_ss, self.col_ss = g('Grid/Row/SS'), g('Grid/Col/SS')
        self.n = np.cross(self.urow, self.ucol)
        self.n /= np.linalg.norm(self.n)
        self.nx, self.ny = nx, ny

    def slant(self, i, j):
        i, j = np.asarray(i, np.float64), np.asarray(j, np.float64)
        return self.scp + (i - self.nx / 2.0)[..., None] * self.col_ss * self.ucol + (j - self.ny / 2.0)[..., None] * self.row_ss * self.urow

    def to_ground(self, i, j, hae=None):
        """-> lat, lon of pixel (i, j): the slant-plane point moved along the plane normal to height hae."""
        hae = self.hae if hae is None else hae
        p = self.slant(i, j)
        t = np.zeros(p.shape[:-1])
        for _ in range(8):
            q = p + t[..., None] * self.n
            lat, lon, h = ecf_to_lla(q)
            up = lla_to_ecf(lat, lon, h) - lla_to_ecf(lat, lon, h - 1.0)       # local up, per metre
            t = t - (h - hae) / np.sum(up * self.n, -1)
        lat, lon, _ = ecf_to_lla(p + t[..., None] * self.n)
        return lat, lon

    def to_pixel(self, lat, lon, hae=None):
        """-> (i, j) fractional pixel of a ground point at height hae (projected along the plane normal)."""
        hae = self.hae if hae is None else hae
        g = lla_to_ecf(np.asarray(lat, np.float64), np.asarray(lon, np.float64), hae) - self.scp
        return (g @ self.ucol) / self.col_ss + self.nx / 2.0, (g @ self.urow) / self.row_ss + self.ny / 2.0


if __name__ == '__main__':
    import sys
    geo = Geo(sys.argv[1], 12207, 8808)
    for name, (i0, j0) in {'locks': (3400, 6700), 'port': (9500, 1150), 'ships': (6400, 2650), 'corner': (11400, 8000), 'center': (12207 / 2 - 256, 8808 / 2 - 256)}.items():
        lat, lon = geo.to_ground(i0 + 256, j0 + 256)
        print(f'{name:7s} {lat:.5f} {lon:.5f}')
    for (i, j) in ((0, 0), (0, 8807), (12206, 8807), (12206, 0)):
        lat, lon = geo.to_ground(i, j)
        print('corner', i, j, f'{lat:.5f} {lon:.5f}')
