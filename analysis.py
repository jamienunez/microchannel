# -*- coding: utf-8 -*-
"""
@author: nune558
"""

# Imports
import argparse
import cv2
import numpy as np
import os

import module as mod


def main(path, fmat, delim, save=True, handdrawn=False, use_edges=False):
    # Create output folder if needed
    save_path = os.path.join(path, 'output/')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    f = open(os.path.join(save_path, '_results.txt'), 'w')
    f.write('Dried Area%sWet Area%sPerc. Dry\n' % (delim, delim))
    
    files = [x for x in os.listdir(path) if fmat in x]
    
    for fname in files:
        res = process_image(path, fname, save_path, fmat, save=save,
                            handdrawn=handdrawn, use_edges=use_edges)
        f.write('%s%s' % (res, '\n'))
    f.close()

def get_points(img):
    edges = cv2.Canny(np.copy(img), 0, 1)
    x = [50, 100, 150, 200, 250, -250, -200, -150, -100, -50]
    top_y = []
    bot_y = []
    for col in x:
        points = np.where(edges[:, col] > 0)
        top_y.append(points[0][0])
        bot_y.append(points[0][-1])
    return x, top_y, bot_y

def draw_line(img, m, b):
    x1 = 0
    x2 = img.shape[1]
    y1 = f(x1, m, b)
    y2 = f(x2, m, b)
    print(x1, x2, y1, y2)
    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 20)
    return
    
from scipy.optimize import curve_fit

def f(x, m, b):
    return m*x + b

def find_line(img):
    img = np.copy(img)
    x, top_y, bot_y = get_points(img)
    m1, b1 = curve_fit(f, x, top_y)[0]
    m2, b2 = curve_fit(f, x, bot_y)[0]
    draw_line(img, m1, b1)
    draw_line(img, m2, b2)
    return

#def process_image(path, fname, save_path, fmat, save=True, handdrawn=False, use_edges=False):

path = '../InputImages/'
fmat = '.tif'
fname = 'completely dried' + fmat
handdrawn = False
save=False
use_edges = True
# Open image
og_img = cv2.imread(os.path.join(path, fname))
img = cv2.cvtColor(np.copy(og_img), cv2.COLOR_BGR2GRAY)

# Find lines for bounding box
if handdrawn:  # Grabs lines drawn in handdrawn folder and applies to image
    hd_path = os.path.join(path, 'handdrawn', fname)
    if os.path.exists(hd_path):
        hd_img = cv2.imread(hd_path)
        hd_img = cv2.cvtColor(np.copy(hd_img), cv2.COLOR_BGR2GRAY)
        hd_img[hd_img < 255] = 0
        hd_img[hd_img > 0] = 50
        lines = mod.find_lines_of_box(hd_img, hd=True)
    else:
        lines = None  # No handdrawn image found. Ignore this image.

else:  # Handdrawn not being used. Find box with IA tools
    img = mod.exaggerate_bounding_box(img)
    find_line(img)
    mod.show(img)
#    lines = mod.find_lines_of_box(img, use_edges)
#
##    if lines is None:  # First try failed
##        lines = mod.find_lines_of_box(img, not use_edges)
#
#if lines is not None:
#
#    # Reduce and reformat line output from HoughLines
#    lines = lines[:2]
#    linesl = [list(x[0]) for x in lines]
#    linesl.sort()
#
#    # Remove empty space outside bounding box
#    img = mod.get_channel(og_img, linesl)
#
#    # Split wet and dry regions
#    val = mod.thresh_img(img)
#    res = '\t'.join([str(x) for x in mod.stats(img, val)])  # Calc here before split
#
#    # Show images
#    if not save:  # Just show
#        mod.plot_all_results(None, None, og_img, lines, img, val)
#    else:
#        mod.plot_all_results(save_path, fname.replace(fmat, ''), og_img, lines, img, val)

    # Report stats
#    return res


#if __name__ == '__main__':
#    
#    # Parse input
#    parser = argparse.ArgumentParser(description='Property calculation using cxcalc')
#    parser.add_argument('path', type=str, help='path to folder root')
#    parser.add_argument('-f','--format', help='Format of images',
#                        required=False, default='.tif')
#    parser.add_argument('-d', '--delim', required=False,
#                        help='Delimiter for results file', default='\t')
#    parser.add_argument('-s', action='store_false', default=True,
#                    dest='save', help='Don\'t save images')
#    parser.add_argument('-a', action='store_true', default=False,
#                    dest='handdrawn', help='Use handdrawn lines')
#    parser.add_argument('-u', action='store_true', default=False,
#                    dest='use_edges', help='Use edges of channel only')
#    args = parser.parse_args()
#
#    # Run
#    main(args.path, args.format, args.delim, save=args.save,
#         handdrawn=args.handdrawn, use_edges=args.use_edges)
