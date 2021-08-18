import timeit

s = """
import cv2
import jpeg4py as jpeg
import torchvision
import PIL
import numpy

try:
    import tensorflow as tf
except:
    pass


def warming_up():
    cv2.imread('/tmp/benchmark.jpg')
    cv2.imread('/tmp/benchmark.png')
    cv2.imread('/tmp/benchmark.bmp')
    cv2.imread('/tmp/benchmark.tif')
    cv2.imread('/tmp/benchmark.webp')

def cv2_imread_jpg():
    img = cv2.imread('/tmp/benchmark.jpg', cv2.IMREAD_COLOR)
    return img

def cv2_imread_png():
    img = cv2.imread('/tmp/benchmark.png', cv2.IMREAD_COLOR)
    return img

def cv2_imread_bmp():
    img = cv2.imread('/tmp/benchmark.bmp', cv2.IMREAD_COLOR)
    return img

def cv2_imread_tif():
    img = cv2.imread('/tmp/benchmark.tif', cv2.IMREAD_COLOR)
    return img

def cv2_imread_webp():
    img = cv2.imread('/tmp/benchmark.webp', cv2.IMREAD_COLOR)
    return img

def jpeg4py_jpg():
    img = jpeg.JPEG('/tmp/benchmark.jpg')
    img = img.decode(pixfmt=jpeg.TJPF_BGR)
    return img

def torch_jpg():
    img = torchvision.io.read_image('/tmp/benchmark.jpg')
    return img

def torch_png():
    img = torchvision.io.read_image('/tmp/benchmark.png')
    return img

def tf_jpg():
    img = tf.io.read_file('/tmp/benchmark.jpg')
    img = tf.image.decode_jpeg(img, channels=3)
    return img

def tf_png():
    img = tf.io.read_file('/tmp/benchmark.png')
    img = tf.image.decode_png(img, channels=3)
    return img

def pil_jpg():
    img = PIL.Image.open('/tmp/benchmark.jpg')
    img = numpy.asarray(img)
    return img

def pil_png():
    img = PIL.Image.open('/tmp/benchmark.png')
    img = numpy.asarray(img)
    return img

def skip():
    return

for i in range(10):
    warming_up()

"""

f1 = "cv2_imread_jpg()"
f2 = "cv2_imread_png()"
f3 = "cv2_imread_bmp()"
f4 = "cv2_imread_tif()"
f5 = "cv2_imread_webp()"
f6 = "jpeg4py_jpg()"
f7 = "torch_jpg()"
f8 = "torch_png()"
f9 = "tf_jpg()"
f10 = "tf_png()"
f11 = "pil_jpg()"
f12 = "pil_png()"


import numpy as np
import cv2
import sys
import os
import os.path

print('Generating sample images 512x512x3 ...')
img = (np.random.standard_normal([512, 512, 3]) * 255).astype(np.uint8)

for f in ['/tmp/benchmark.jpg','/tmp/benchmark.png','/tmp/benchmark.bmp','/tmp/benchmark.tif']:
    cv2.imwrite(f, img)

try:
    cv2.imwrite('/tmp/benchmark.wepb', img)
except:
    f5 = "skip()"
    print('OpenCV has not Webp support.', sys.exc_info())

try:
    import tensorflow as tf
except:
    f9 = "skip()"
    f10 = "skip()"

print('Start bencmark...')
for f in [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12]:
    print(f, timeit.timeit(setup = s, stmt = f, number = 1000))

print('Cleanup')
for f in ['/tmp/benchmark.jpg','/tmp/benchmark.png','/tmp/benchmark.bmp','/tmp/benchmark.tif','/tmp/benchmark.wepb']:
    if os.path.exists(f):
        os.remove(f)
