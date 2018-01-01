import numpy as np
import math

"""
I originally designed these classes to help me with `PIL.Image.transform`.
But then I realized its easier to use `PIL.Image.rotate` for rotating
and `PIL.Image.transform` for scaling and translating only.

@author possatti
"""

class IdentityMatrix(object):
    def get(self):
        return np.identity(3)
    def getsix(self):
        return self.get().reshape(9)[:6]

class RotationMatrix(IdentityMatrix):
    """Rotate counter-clockwise."""
    def __init__(self, deg=0, rad=0):
        if deg != 0:
            self.rad = math.radians(deg)
        else:
            self.rad = rad
    def get(self):
        return np.array([
            [math.cos(self.rad), -math.sin(self.rad), 0],
            [math.sin(self.rad),  math.cos(self.rad), 0],
            [0, 0, 1],
        ])

class TranslationMatrix(IdentityMatrix):
    def __init__(self, tx, ty):
        self.tx = tx
        self.ty = ty
    def get(self):
        return np.array([
            [1,0,self.tx],
            [0,1,self.ty],
            [0,0,1]])

class ScalingMatrix(IdentityMatrix):
    def __init__(self, sx, sy):
        self.sx = sx
        self.sy = sy
    def get(self):
        return np.array([
            [self.sx, 0, 0],
            [0, self.sy, 0],
            [0, 0, 1]])

class ComposeMatrix(IdentityMatrix):
    def __init__(self):
        self.transforms = [IdentityMatrix()]
    def add(self, matrix):
        self.transforms.append(matrix)
    def get(self):
        result_m = self.transforms[0].get()
        for m in self.transforms[1:]:
            result_m = np.dot(result_m, m.get())
        assert result_m[-1,0] == 0, "Houston... We have a problem."
        assert result_m[-1,1] == 0, "Houston... We have a problem."
        return result_m / result_m[-1,-1]

if __name__ == '__main__':
    cm = ComposeMatrix()
    cm.add(TranslationMatrix(32,32))
    cm.add(RotationMatrix(45))
    cm.add(TranslationMatrix(-32,-32))
    print(cm.getsix())
