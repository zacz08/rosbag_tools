import cv2
import numpy as np
import re

# import matplotlib
# from matplotlib import pyplot as plt
# matplotlib.use("GTK3Agg")


class ReRes:
    def __init__(self, resolution=(None, None), scale_factor=None):
        self.res = None
        self.scale = None
        self.use_scale = None

        self.set_res(resolution, scale_factor)

    def draw(self, img_in):
        img_out = img_in.copy()

        if self.use_scale:
            img_out = cv2.resize(img_out, None, fx=self.scale, fy=self.scale)
        else:
            img_out = cv2.resize(img_out, self.res)

        return img_out

    def set_res(self, resolution, scale_factor):
        self.scale = scale_factor
        self.res = resolution
        self.use_scale = self.scale is not None


class BlurDistort:
    def __init__(self, size=None):
        if not size:
            size = 0
        self.kernel_size = (size, size)

        self.img_out = None

    def draw(self, img_in):
        self.img_out = img_in.copy()

        try:
            self.img_out = cv2.GaussianBlur(self.img_out, self.kernel_size, cv2.BORDER_CONSTANT)
        except cv2.error as e:
            # if error is thrown because the kernel size is even, +1 to kernel size
            if re.findall(r"% 2 == 1", str(e)):
                self.kernel_size = (self.kernel_size[0]+1, self.kernel_size[1]+1)
                self.img_out = cv2.GaussianBlur(self.img_out, self.kernel_size, cv2.BORDER_CONSTANT)
            else:
                raise e

        return self.img_out

    def set_kernel(self, size=None):
        if not size:
            size = 0
        self.kernel_size = (size, size)


class GlareDistort:
    def __init__(self, threshold=None, scale=None):
        if not threshold:
            threshold = 200
        self.threshold = threshold

        if not scale:
            scale = 2
        self.scale = scale

        self.img_out = None

    def draw(self, img_in):
        self.img_out = img_in.copy()

        if (self.img_out >= self.threshold).any():
            self.img_out[np.where(self.img_out < self.threshold)] = \
                self.img_out[np.where(self.img_out < self.threshold)] / self.scale

        return self.img_out


class RectangleOcculsion:
    # only configured for grayscale images at the moment
    def __init__(self, dimensions=None, border=False, centre_point=None, x_move=0, y_move=0):
        if dimensions is not None:
            self.dimensions = dimensions
        else:
            self.dimensions = np.array([150, 100])

        if border:
            self.bordersize = np.array([10, 10])
        else:
            self.bordersize = np.array([0, 0])
        
        if centre_point is not None:
            self.tl_corner = np.round(centre_point - self.dimensions / 2).astype(int)
        else:
            self.tl_corner = np.array([0, 0])

        self.br_corner = self.tl_corner + self.dimensions
        self.xmove = x_move
        self.ymove = y_move
        self.img_out = None

        self.COLOUR_BLACK = 0
        self.COLOUR_WHITE = 255
        self.RECT_FILLED = -1

    def draw(self, img_in):
        self.img_out = img_in.copy()
        self.move_rect()
        cv2.rectangle(self.img_out, self.tl_corner - self.bordersize, self.br_corner + self.bordersize, self.COLOUR_WHITE, self.RECT_FILLED)
        cv2.rectangle(self.img_out, self.tl_corner, self.br_corner, self.COLOUR_BLACK, self.RECT_FILLED)
        return self.img_out

    def move_rect(self):
        if self.img_out is not None:
            newx = [self.tl_corner[0] + self.xmove, self.br_corner[0] + self.xmove]
            if any(x > self.img_out.shape[1] for x in newx) or any(x < 0 for x in newx):
                self.xmove = -self.xmove
                newx = [self.tl_corner[0] + self.xmove, self.br_corner[0] + self.xmove]
            self.tl_corner[0] = newx[0]
            self.br_corner[0] = newx[1]

            newy = [self.tl_corner[1] + self.ymove, self.br_corner[1] + self.ymove]
            if any(y > self.img_out.shape[0] for y in newy) or any(y < 0 for y in newy):
                self.ymove = -self.ymove
                newy = [self.tl_corner[1] + self.ymove, self.br_corner[1] + self.ymove]
            self.tl_corner[1] = newy[0]
            self.br_corner[1] = newy[1]
        else:
            raise ValueError("No image set yet for bounce boundaries")
