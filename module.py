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


#%% Image analysis and processing functions

def find_thresh(img):
    return filters.threshold_otsu(img)


def threshold_img(img, val):
    img[img < val] = 0
    img[img > 0] = 1
    return img


def _get_points(edges, x):
    top_y = []
    bot_y = []
    for col in x:
        points = np.where(edges[:, col] > 0)
        if len(points[0]) > 0:
            top_y.append(points[0][0])
            bot_y.append(points[0][-1])
        else:
            top_y.append(np.nan)
            bot_y.append(np.nan)
    return np.array(top_y), np.array(bot_y)


def get_points(img, left=False, right=False):
    edges = cv2.Canny(np.copy(img), 0, 1)
    w = img.shape[1]
    x = [50, 75, 100, 125, 150, 175, 200]
    for temp in list(reversed(x)):
        x.append(w - temp)
    if left:
        x = x[:len(x) / 2]
    elif right:
        x = x[len(x) / 2:]
    top_y, bot_y = _get_points(edges, x)
    return np.array(x), np.array(top_y), np.array(bot_y)


def _find_lines(img, left=False, right=False):
    x, top_y, bot_y = get_points(img, left=left, right=right)
    ind = np.array(np.isfinite(top_y), ndmin=1)
    (m1, b1), r1, _, _, _ = np.polyfit(x[ind], top_y[ind], 1, full=True)
    (m2, b2), r2, _, _, _ = np.polyfit(x[ind], bot_y[ind], 1, full=True)
    return [[m1, b1], [m2, b2]], r1 + r2


def find_lines(img):
    img = np.copy(img)
    lines, r = _find_lines(img)
    if r < 200:
        return lines

    else:
        lines, r = _find_lines(img, right=True)
        if r < 200:
            return lines

        else:
            lines, r = _find_lines(img, right=True)
            if r < 200:
                return lines

    return None


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


# Extract the actual channel from the image
def get_channel(og_img, lines):
    img = cv2.cvtColor(np.copy(og_img), cv2.COLOR_BGR2GRAY)
    img = np.array(img, dtype=np.float)
    img = remove_pixels(img, lines[0])
    img = remove_pixels(img, lines[1], above=False)
    return img


def draw_lines(img, lines):
    img = np.copy(img)
    w = img.shape[1]
    for line in lines:
        m, b = line
        x1 = 0; y1 = int(b); x2 = w; y2 = int(m * w + b)
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 20)
    return img


#%% Reporting functions

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


#%% Plotting functions

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
    img[np.isnan(img)] = -1  # Avoids runtime warnings
    ind = np.where(np.logical_and(img >= 0, img < val))
    temp_img[ind] = 1
    ind = np.where(img >= val)
    temp_img[ind] = 2
    show(temp_img, name=name)


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
        