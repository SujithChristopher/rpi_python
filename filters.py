import numpy as np

class ExponentialMovingAverageFilter3D:
    def __init__(self, alpha):
        self.alpha = alpha
        self.ema_x = None
        self.ema_y = None
        self.ema_z = None

    def update(self, ema):
        if self.ema_x is None:
            self.ema_x = ema[0]
            self.ema_y = ema[1]
            self.ema_z = ema[2]
        else:
            self.ema_x = self.alpha * ema[0] + (1 - self.alpha) * self.ema_x
            self.ema_y = self.alpha * ema[1] + (1 - self.alpha) * self.ema_y
            self.ema_z = self.alpha * ema[2] + (1 - self.alpha) * self.ema_z
        return np.array([self.ema_x, self.ema_y, self.ema_z])
