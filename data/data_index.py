# Pixel indices for each parking slot / region.
class Region:
    def __init__(self, id):
        self.id = id
        self.pixels = []
        self.p0 = (0,0)
        self.p1 = (0,0)
        self.p2 = (0,0)
        self.p3 = (0,0)

    def get_pixels(self):
        return self.pixels

    def set_pixels(self, pixels):
        self.pixels = pixels

    def get_corners(self):
        return [self.p0, self.p1, self.p2, self.p3]

    def set_corners(self, pts):
        self.p0 = pts[0]
        self.p1 = pts[1]
        self.p2 = pts[2]
        self.p3 = pts[3]


data_index = {
    '53': Region(53),
    '54': Region(54),
    '55': Region(55),
    '56': Region(56),
    '57': Region(57),
    '58': Region(58),
    '-1': Region(-1),
    '-2': Region(-2),
    '-3': Region(-3),
    '-4': Region(-4),
}