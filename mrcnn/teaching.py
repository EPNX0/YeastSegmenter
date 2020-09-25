
"""
Created on Fri Sep 27 13:51:52 2019

@author: erich
"""

"""
Created on Mon May 27 12:30:07 2019

@author: erich
"""
import matplotlib
matplotlib.use('Agg') # prevents patplotlib from showing plots but still saves them
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.patches import Polygon

import os
import cv2 #new
import random

import colorsys
import math

import numpy as np
import numpy.linalg as la

import skimage
from skimage.measure import find_contours


import imageio


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

def display_instances(image_id, image, boxes, masks, class_ids, class_names, SAVE_DIR, 
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=False, show_bbox=False,
                      colors=None, captions=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    SAVE_DIR: Directory to save the images
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    
    # Center
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    #auto_show = False
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
        #auto_show = True

    # Generate random colors
    colors = colors or random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)
   
   # Counter for class_names; needed for mask pull
   # For other/more classes; change these variables and the Pull Masks sector
    cl1 = 0
    cl2 = 0
    cl3 = 0
    cl4 = 0
    cl5 = 0

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        IM_SAVE_DIR = os.path.join(SAVE_DIR, 'New'+str(i)+'/image')
        if not os.path.exists(IM_SAVE_DIR):
            os.makedirs(IM_SAVE_DIR)
        MASK_SAVE_DIR = os.path.join(SAVE_DIR, 'New'+str(i)+'/masks')
        if not os.path.exists(MASK_SAVE_DIR):
            os.makedirs(MASK_SAVE_DIR)
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                alpha=0.7, linestyle="dashed",
                                edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Label
        if not captions:
            box=boxes[i] #debug prupose
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            caption = "{} {} {:.3f} ".format(label, box, score) if score else label, box.tolist()
        else:
            box = boxes[i] #debug purpose
            caption = captions[i]
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            caption = "{} {} {:.3f} {}".format(caption, label, score, box) if score else label
        ax.text(x1 - 175 , y1 - 8, caption,
                color=color, size=8, backgroundcolor="none")

        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        # Pull Masks
        if class_id == 1:
            cl1 += 1
            m = masks[:, :, np.where(class_ids == class_id)[0]]
            m = np.sum(m * np.arange(1, m.shape[-1] + 1), -1)
            ma=np.where(m==cl1, 1, 0)
            if np.sum(ma) !=0:
                if SAVE_DIR is not None:
                    imageio.imwrite(os.path.join(MASK_SAVE_DIR, 'ImgID'+'_'+str(image_id)+'.result.'
                                                 +str(class_names[class_id])+'._.'+'Sub_ID'+'_'+class_id
                                                 +class_id+'.tif'), np.where(ma==1, 255, 0).astype(np.uint8))
        if class_id == 2:
            cl2 +=1
            m = masks[:, :, np.where(class_ids == class_id)[0]]
            m = np.sum(m * np.arange(1, m.shape[-1] + 1), -1)
            ma=np.where(m==cl2, 1, 0)
            if np.sum(ma) !=0:
                if SAVE_DIR is not None:
                    imageio.imwrite(os.path.join(MASK_SAVE_DIR, 'ImgID'+'_'+str(image_id)+'.result.'
                                                 +str(class_names[class_id])+'._.'+'Sub_ID'+'_'+class_id
                                                 +'.tif'), np.where(ma==1, 255, 0).astype(np.uint8))
        if class_id == 3:
            cl3 += 1
            m = masks[:, :, np.where(class_ids == class_id)[0]]
            m = np.sum(m * np.arange(1, m.shape[-1] + 1), -1)
            ma=np.where(m==cl3, 1, 0)
            if np.sum(ma) !=0:
                if SAVE_DIR is not None:
                    imageio.imwrite(os.path.join(MASK_SAVE_DIR, 'ImgID'+'_'+str(image_id)+'.result.'
                                                 +str(class_names[class_id])+'._.'+'Sub_ID'+'_'+class_id
                                                 +'.tif'), np.where(ma==1, 255, 0).astype(np.uint8))
        if class_id == 4:
            cl4 += 1
            m = masks[:, :, np.where(class_ids == class_id)[0]]
            m = np.sum(m * np.arange(1, m.shape[-1] + 1), -1)
            ma=np.where(m==cl4, 1, 0)
            if np.sum(ma) !=0:
                if SAVE_DIR is not None:
                    imageio.imwrite(os.path.join(MASK_SAVE_DIR, 'ImgID'+'_'+str(image_id)+'.result.'
                                                 +str(class_names[class_id])+'._.'+'Sub_ID'+'_'+class_id
                                                 +'.tif'), np.where(ma==1, 255, 0).astype(np.uint8))
                    
        if class_id == 5:
            cl5 += 1
            m = masks[:, :, np.where(class_ids == class_id)[0]]
            m = np.sum(m * np.arange(1, m.shape[-1] + 1), -1)
            ma=np.where(m==cl5, 1, 0)
            if np.sum(ma) !=0:
                if SAVE_DIR is not None:
                    imageio.imwrite(os.path.join(MASK_SAVE_DIR, 'ImgID'+'_'+str(image_id)+'.result.'
                                                 +str(class_names[class_id])+'._.'+'Sub_ID'+'_'+class_id
                                                 +'.tif'), np.where(ma==1, 255, 0).astype(np.uint8))
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8))
    plt.savefig(os.path.join(IM_SAVE_DIR, 'orig.'+str(image_id)+'.tif'))
    plt.close('all')



