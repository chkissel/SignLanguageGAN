# -*- coding: utf-8 -*-
#credit: https://jdhao.github.io/2017/11/06/resize-image-to-square-with-padding/
from PIL import Image, ImageOps

def padding(img, desired_size, mode):
    """Applies zero padding to image"""
    
    old_size = img.size  # old_size[0] is in (width, height) format

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    if mode == "L" or mode == "P":
        img = img.resize(new_size)
    else:
        img = img.resize(new_size, Image.ANTIALIAS)
    

    # create a new image and paste the resized on it
    new_img = Image.new(mode, (desired_size, desired_size))
    new_img.paste(img, ((desired_size-new_size[0])//2,
                        (desired_size-new_size[1])//2))

    return new_img
