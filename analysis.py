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


def main(path, fmat, delim, save=True, handdrawn=False):
    
    # Create output folder if needed
    save_path = os.path.join(path, 'output/')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    f = open(os.path.join(save_path, '_results.txt'), 'w')
    f.write('Dried Area%sWet Area%sPerc. Dry\n' % (delim, delim))
    
    files = [x for x in os.listdir(path) if fmat in x]
    
    for fname in files:
        res = process_image(path, fname, save_path, fmat, save=save,
                            handdrawn=handdrawn)
        f.write('%s%s' % (res, '\n'))
    f.close()


def process_image(path, fname, save_path, fmat, save=True, handdrawn=False):

    # Open image
    og_img = cv2.imread(os.path.join(path, fname))
    img = cv2.cvtColor(np.copy(og_img), cv2.COLOR_BGR2GRAY)
    
    # Find lines for bounding box
    if handdrawn:
        # Grabs lines drawn in handdrawn folder and applies to image
        hd_path = os.path.join(path, 'handdrawn', fname)
        if os.path.exists(hd_path):
            hd_img = cv2.imread(hd_path)
            hd_img = cv2.cvtColor(np.copy(hd_img), cv2.COLOR_BGR2GRAY)
            
            val = 255
            hd_img = mod.threshold_img(hd_img, val)
            lines = mod.find_lines_of_box(hd_img, hd=True)
        else:
            # No handdrawn image found. Ignore this image.
            lines = None
    
    else:  # Handdrawn not being used. Find box with IA tools

        # Threshold
        val = mod.find_thresh(img)
        img = mod.threshold_img(img, val)
        
        # Remove noise
        kernel = np.ones((3, 3), np.uint8)
        img = cv2.erode(img, kernel, iterations=2)
        
        # Find lines
        lines = mod.find_lines(img)
    
    # Exit if lines not found
    if lines is None:
        print('Fname:', fname)
        return ''
        
    # Remove empty space outside bounding box
    img = mod.get_channel(og_img, lines)

    # Split wet and dry regions
    res = fname + '\t'
    res = res + '\t'.join([str(x) for x in mod.stats(img, val)])

    # Show images and save
    mod.plot_all_results(save_path, fname.replace(fmat, ''), og_img,
                         lines, img, val)
        
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
    args = parser.parse_args()

    # Run
    main(args.path, args.format, args.delim, save=args.save,
         handdrawn=args.handdrawn)
