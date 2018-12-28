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

#def draw_line(img, m, b):
#    x1 = 0
#    x2 = img.shape[1]
#    y1 = int(f(x1, m, b))
#    y2 = int(f(x2, m, b))
#    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 20)
#    return

def _find_line(img, left=False, right=False):
    x, top_y, bot_y = get_points(img, left=left, right=right)
    ind = np.array(np.isfinite(top_y), ndmin=1)
#    print(top_y[ind])
#    print(bot_y[ind])
    (m1, b1), r1, _, _, _ = np.polyfit(x[ind], top_y[ind], 1, full=True)
    (m2, b2), r2, _, _, _ = np.polyfit(x[ind], bot_y[ind], 1, full=True)
#    print(r1, r2)
    return [[m1, b1], [m2, b2]], r1 + r2

def find_line(img):
    img = np.copy(img)
    lines, r = _find_line(img)
    if r < 200:
        return lines

    else:
        lines, r = _find_line(img, right=True)
        if r < 200:
            return lines

        else:
            lines, r = _find_line(img, right=True)
            if r < 200:
                return lines

    return None

#def process_image(path, fname, save_path, fmat, save=True, handdrawn=False, use_edges=False):

path = '../InputImages/'
fmat = '.tif'
fname = 'A10_2_S_ch2_t70h' + fmat
handdrawn = False
save = False
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
#    img = mod.exaggerate_bounding_box(img)

    # Threshold
    val = mod.filters.threshold_otsu(img)
    img[img < val] = 0
    img[img > 0] = 50
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.erode(img, kernel, iterations=2)
    lines = find_line(img)

    if lines is None:  # First try failed
        print('Fname:', fname)
    #        lines = mod.find_lines_of_box(img, not use_edges)
    
    if lines is not None:
        
        # Remove empty space outside bounding box
        img = mod.get_channel(og_img, lines)
    
        # Split wet and dry regions
#        val = 49  #mod.thresh_img(img)
        res = '\t'.join([str(x) for x in mod.stats(img, val)])  # Calc here before split
    
        # Show images
        if not save:  # Just show
            mod.plot_all_results(None, None, og_img, lines, img, val)
        else:
            mod.plot_all_results(save_path, fname.replace(fmat, ''), og_img, lines, img, val)
            
#                # Report stats
#                return res
#        return ''


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
