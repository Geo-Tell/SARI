from PIL import Image
import random


class RandomCrop(object):
    def __init__(self, size, *args, **kwargs):
        self.size = size

    def __call__(self, im_lb):
        im = im_lb['im']
        lb = im_lb['lb']
        assert im.size == lb.size
        W, H = self.size
        w, h = im.size

        if (W, H) == (w, h): return dict(im = im, lb = lb)
        if w < W or h < H:
            image_padding = Image.new('RGB', (W, H), (255, 255, 255))
            label_padding = Image.new('L', (W, H), 255)
            col_init = random.randint(0,W-w)
            row_init = random.randint(0,H-h)
            image_padding.paste(im, (col_init, row_init, w+col_init, h+row_init))
            label_padding.paste(lb, (col_init, row_init, w+col_init, h+row_init))
            im = image_padding
            lb = label_padding
            w = W
            h = H
        sw, sh = random.randint(0, w - W), random.randint(0, h - H)
        
        crop = int(sw), int(sh), int(sw) + W, int(sh) + H
        return dict(
                im = im.crop(crop),
                lb = lb.crop(crop)
                    )
class RandomRotate(object):
    def __init__(self, angle, *args, **kwargs):
        self.angle = angle

    def __call__(self, im_lb):
        im = im_lb['im']
        lb = im_lb['lb']
        angle = random.choice(self.angle)
        return dict(im = im.rotate(angle),
                    lb = lb.rotate(angle),
                )

class HorizontalFlip(object):
    def __init__(self, p = 0.5, *args, **kwargs):
        self.p = p

    def __call__(self, im_lb):
        if random.random() > self.p:
            return im_lb
        else:
            im = im_lb['im']
            lb = im_lb['lb']
            return dict(im = im.transpose(Image.FLIP_LEFT_RIGHT),
                        lb = lb.transpose(Image.FLIP_LEFT_RIGHT),
                    )
        
class VerticalFlip(object):
    def __init__(self, p = 0.5, *args, **kwargs):
        self.p = p

    def __call__(self, im_lb):
        if random.random() > self.p:
            return im_lb
        else:
            im = im_lb['im']
            lb = im_lb['lb']
            return dict(im = im.transpose(Image.FLIP_TOP_BOTTOM),
                        lb = lb.transpose(Image.FLIP_TOP_BOTTOM),
                    )

class RandomScale(object):
    def __init__(self, scales = (1, ), *args, **kwargs):
        self.scales = scales

    def __call__(self, im_lb):
        im = im_lb['im']
        lb = im_lb['lb']
        W, H = im.size
        scale = random.choice(self.scales)
        w, h = int(W * scale + 0.5), int(H * scale + 0.5)
        return dict(im = im.resize((w, h), Image.BILINEAR),
                    lb = lb.resize((w, h), Image.NEAREST),
                )

class Compose(object):
    def __init__(self, do_list):
        self.do_list = do_list

    def __call__(self, im_lb):
        for comp in self.do_list:
            im_lb = comp(im_lb)
        return im_lb