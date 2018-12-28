# -*- coding: utf-8 -*-
"""
Created on Wed Nov 07 16:46:09 2018

@author: nune558
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import filters
import matplotlib.cm as cm
import matplotlib.colors as colors

resolution = 607  # pixel/mm

# Register cividis with matplotlib
rgb_cividis = np.loadtxt('cividis.txt').T
cmap = colors.ListedColormap(rgb_cividis.T, name='cividis')
cm.register_cmap(name='cividis', cmap=cmap)

def show(img, name=None, vmax=None, vmin=None, cmap='cividis', axis='off', origin='upper', norm=None):
    plt.imshow(img, cmap=cmap, vmax=vmax, vmin=vmin, interpolation='none', origin=origin, norm=norm)
    plt.axis(axis)
    if name is not None:
        plt.savefig(name, dpi=500, bbox_inches='tight')
        plt.pause(0.005)
        plt.ion()
        plt.show()
        plt.close()

def show_wet_dry(img, val, name=None):    
    temp_img = np.copy(img)
    ind = np.where(np.logical_and(img >= 0, img < val))
    temp_img[ind] = 1
    ind = np.where(img >= val)
    temp_img[ind] = 2
    show(temp_img, name=name)

# Line: (rho, theta)
# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
def get_points(line, w):
    rho, theta = line
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + w * (-b))
    y1 = int(y0 + w * (a))
    x2 = int(x0 - w * (-b))
    y2 = int(y0 - w * (a))
    return x1, y1, x2, y2


def exaggerate_bounding_box(img, iterations=2):
    img[img < filters.threshold_otsu(img)] = 0
    img[img > 0] = 50
    kernel = np.ones((5, 5), np.uint8)
    img = cv2.erode(img, kernel, iterations=2)
#    kernel = np.ones((200, 200), np.uint8)
    img = cv2.dilate(img, kernel, iterations=iterations)
#    img = cv2.erode(img, kernel, iterations=iterations)
#    kernel = np.ones((10, 10), np.uint8)
#    img = cv2.erode(img, kernel)
    return img

def find_lines_of_box(img, use_edges=False, hd=False):
    edges = cv2.Canny(img, 0, 1)
    w = img.shape[1]
    num_points = w / 4
    if use_edges:
        edges[:, 300:-300] = 0
        num_points = 300
    if hd:
        res = 10
    else:
        res = 7
    lines = cv2.HoughLines(edges, res, np.pi / 180, num_points)
    if lines is not None:
        lines = [x for x in lines if x[0][1] > 1 and x[0][1] < 2]
#        print(len(lines))
        temp1 = [x for x in lines[1:] if abs(x[0][0] - lines[0][0][0]) > 450]
        if len(temp1) == 0:
            temp1 = [x for x in lines[1:] if abs(x[0][0] - lines[0][0][0]) > 400]
#        print(len(temp1))
        temp2 = [x for x in temp1 if abs(x[0][1] - lines[0][0][1]) < 0.01]
#        print(len(temp2))
        if len(temp2) == 0:
            temp2 = [x for x in temp1 if abs(x[0][1] - lines[0][0][1]) < 0.02]
#            print(len(temp2))
        lines = [lines[0]]
        lines.extend(temp2)
    if lines is None or len(lines) < 2:
        return None
    return lines

def get_equation(line, w):
    rho, theta = line
    points = get_points(line, w)
    x1, y1, x2, y2 = [float(x) for x in points]
    m = (y2 - y1) / (x2 - x1)
    b = y1 - x1 * m
    return m, b


#def draw_lines(img, lines):
#    img = np.copy(img)
#    for line in lines:
#        for rho,theta in line:
#            x1, y1, x2, y2 = get_points([rho, theta], img.shape[1])
#            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 20)
#    return img

def draw_lines(img, lines):
    img = np.copy(img)
    w = img.shape[1]
    for line in lines:
        m, b = line
        x1 = 0; y1 = int(b); x2 = w; y2 = int(m * w + b)
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 20)
    return img


def stats(img, val):
    img = img[~np.isnan(img)]
    dried_pixels = np.sum(img < val)
    wet_pixels = np.sum(img >= val)
    dried_area = float(dried_pixels) * (1. / (resolution ** 2))
    wet_area = float(wet_pixels) * (1. / (resolution ** 2))
    perc = dried_area / (dried_area + wet_area) * 100
    return dried_area, wet_area, perc


def print_stats(img, val):
    dried_area, wet_area, perc = stats(img, val)
    print('%.2f mm2, %.2f mm2, %i perc' % (dried_area, wet_area, perc))

def remove_pixels(img, line, above=True):
    m, b = line
    x = np.arange(img.shape[1])
    y = m * x + b
    for i in range(len(x)):
        if above:
            img[:y[i], x[i]] = np.nan
        else:
            img[y[i]:, x[i]] = np.nan
    return img


def thresh_img(img):
    img = np.copy(img)
    
    # Remove NaN values (if any)
    n = img.shape[0] * img.shape[1]
    pixels = np.copy(img).reshape(n, 1)
    ind = np.array(1 - np.isnan(pixels), dtype=bool)
    pixels = pixels[ind]
    
    # Thresh
    return filters.threshold_otsu(pixels)


# Extract the actual channel from the image
def get_channel(og_img, lines):
    img = cv2.cvtColor(np.copy(og_img), cv2.COLOR_BGR2GRAY)
    img = np.array(img, dtype=np.float)
    img = remove_pixels(img, lines[0])
    img = remove_pixels(img, lines[1], above=False)
    return img
    

def plot_all_results(save_path, fname, og_img, lines, img, val):
    if save_path is not None and fname is not None:
        fname1 = '%s%s_boundingregion' % (save_path, fname)
        fname2 = '%s%s_channel' % (save_path, fname)
        fname3 = '%s%s_channel_dryregions' % (save_path, fname)
    else:
        fname1 = None
        fname2 = None
        fname3 = None
    show(draw_lines(og_img, lines), name=fname1)
    if fname1 is None:
        plt.show()
    show(img, name=fname2)
    if fname2 is None:
        plt.show()
    show_wet_dry(img, val, name=fname3)
    if fname3 is None:
        plt.show()