#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Helper functions to process image data.

@author: amagrabi

'''

import numpy as np
import os
import urllib
import glob
import validators
import magic
from PIL import Image
from mimetypes import guess_extension
import time
import shutil
import hashlib
import signal

 
def valid_img_url(url):
    '''Check if url is valid and refers to an image.
    
    Args:
        url: url to an image.
        
    Returns:
        True if image url is valid.
    '''
    
    if validators.url(url):
        try:
            req = urllib.request.urlopen(url)
        except:
            return False
        if req.getcode() in range(200, 209) and req.headers.get_content_maintype() == 'image':
            return True
        else:
            return False
    else:
        return False

        
def open_img_from_url(url):
    return Image.open(urllib.request.urlopen(url))
    
    
def download_imgs(url_list, subfolder_list, path_root, NAME_PROJECT, 
                  convert_jpg=True, min_img_size=150, num_length=5, 
                  min_cat_nr=20, wait=10, delay=0, delete_low_n=True):
    '''Downloads all images from a list of urls into subfolders.
    
    Args:
        url_list: List of image urls.
        subfolder_list: List of folder names to store images (order corresponding to url_list).
    '''

    url_list_len = len(url_list)
                         
    if len(url_list) != len(subfolder_list):
        raise Exception('Url and subfolder list need to have the same length.')
    
    for i, url in enumerate(url_list):
        try:
            img_name = NAME_PROJECT + '_' + subfolder_list[i] + '_' + str(i).zfill(num_length)
            source = urllib.request.urlopen(url)
            extension = guess_extension(source.info()['Content-Type'])
            if extension:
                if not convert_jpg:
                    img_name += extension
                    path = os.path.join(path_root, subfolder_list[i], img_name)
                    urllib.request.urlretrieve(url, path)
                else:
                    path = os.path.join(path_root, subfolder_list[i], img_name)
                    img = open_img_from_url(url)
                    
                    if all(x >= min_img_size for x in img.size): 
#                        if img.mode != "RGB":
#                            img = img.convert("RGB")
                        img.thumbnail((640,640), Image.NEAREST)
                        img_whitebg = Image.new("RGB", img.size, (255,255,255))
                        img_whitebg.paste(img)
                        img_whitebg.save(path + '.jpg', "JPEG")
                        print('Iteration {}/{}: Saved image {} ({}).'.format(i, url_list_len, img_name, url))
                    else:
                        print('Iteration {}/{}: Image {} was too small ({},{} < {}).'.format(i, url_list_len, url, img.size[0], img.size[1], min_img_size))
                
            else:
                print('Iteration {}/{}: Image {} could not be saved, no extension identified.'.format(i, url_list_len, url))
                pass
        except:
                print('Iteration {}/{}: Image {} could not be saved, attempt failed.'.format(i, url_list_len, url))
                pass
        time.sleep(delay) if delay else None
    
    if delete_low_n:
        for path, subdirs, files in os.walk(path_root):
            for subdir in subdirs:
                subdir_path = os.path.join(path, subdir)
                subdir_len = len(os.listdir(subdir_path))
                if subdir_len < min_cat_nr:
                    shutil.rmtree(subdir_path)
                    print('Images for category {} are too few ({} < {}) - deleted.'.format(subdir, subdir_len, min_cat_nr))
                
            
def del_images_below_size(img_folder, min_cat_nr=150):
     for path, subdirs, files in os.walk(img_folder):
        for file in files:
            try:
                filepath = os.path.join(path, file)
                im = Image.open(filepath)
                if all(x < min_cat_nr for x in im.size): 
                    os.remove(filepath)
                    print('Image deleted: {} ({},{} < {}).'.format(file, im.size[0], im.size[1], min_size))
            except:
                pass
            

def convert_to_jpg(path):
    for path, subdirs, files in os.walk(path):
        for file in files:
            filepath = os.path.join(path, file)
            try:
                im = Image.open(filepath)
                if im.mode != "RGB":
                    im = im.convert("RGB")
                im.thumbnail((640,640), Image.NEAREST)
                im.save(filepath + '.jpg', "JPEG")
                print('Printed file: {}'.format(filepath))
            except:
                pass

                
def delete_jpgs(path):
    for path, subdirs, files in os.walk(path):
        for file in files:
            filepath = os.path.join(path, file)
            os.remove(filepath) if filepath.endswith('.jpg') else None
            
            
def find_duplicate_image(target_url, template_url):
    img_target = open_img_from_url(target_url)
    img_template = open_img_from_url(template_url)
    width, height = img_template.size
    pixels = list(img_template.getdata())
    for col in range(width):
        print(pixels[col:col+width])
    return img_template.shape == img_target.shape and not(np.bitwise_xor(img_template,img_target).any())
    

def convert_png_to_jpg(path):
    for path, subdirs, files in os.walk(path):
        for file in files:
            filepath = os.path.join(path, file)
            if filepath.endswith('.png'):
                try:
                    im = Image.open(filepath)
                    if im.mode != "RGB":
                        im = im.convert("RGB")
                    im.save(filepath[0:-4] + '.jpeg', "JPEG")
                    print('Converted to jpg: {}'.format(filepath))
                except:
                    pass
    
    