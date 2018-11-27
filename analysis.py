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


def process_image(path, fname, save_path, fmat, save=True, handdrawn=False, use_edges=False):

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
        lines = mod.find_lines_of_box(img, use_edges)

        if lines is None:  # First try failed
            lines = mod.find_lines_of_box(img, not use_edges)

    if lines is not None:

        # Reduce and reformat line output from HoughLines
        lines = lines[:2]
        linesl = [list(x[0]) for x in lines]
        linesl.sort()

        # Remove empty space outside bounding box
        img = mod.get_channel(og_img, linesl)

        # Split wet and dry regions
        val = mod.thresh_img(img)
        res = '\t'.join([str(x) for x in mod.stats(img, val)])  # Calc here before split

        # Show images
        if not save:  # Just show
            mod.plot_all_results(None, None, og_img, lines, img, val)
        else:
            mod.plot_all_results(save_path, fname.replace(fmat, ''), og_img, lines, img, val)

    # Report stats
    return res


if __name__ == '__main__':
    
    # Parse input
    parser = argparse.ArgumentParser(description='Property calculation using cxcalc')
    parser.add_argument('path', type=str, help='path to folder root')
    parser.add_argument('-f','--format', help='Format of images',
                        required=False, default='.tif')
    parser.add_argument('-d', '--delim', required=False,
                        help='Delimiter for results file', default='\t')
    parser.add_argument('-s', action='store_false', default=True,
                    dest='save', help='Don\'t save images')
    parser.add_argument('-a', action='store_true', default=False,
                    dest='handdrawn', help='Use handdrawn lines')
    parser.add_argument('-u', action='store_true', default=False,
                    dest='use_edges', help='Use edges of channel only')
    args = parser.parse_args()

    # Run
    main(args.path, args.format, args.delim, save=args.save,
         handdrawn=args.handdrawn, use_edges=args.use_edges)
