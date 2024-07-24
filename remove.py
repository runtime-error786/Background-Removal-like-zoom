import cv2
import os
import numpy as np
from cvzone.SelfiSegmentationModule import SelfiSegmentation

segmentor = SelfiSegmentation()

img_folder = r'C:\Users\musta\OneDrive\Desktop\Background Removal Like Zoom\img'
background_images = [os.path.join(img_folder, img) for img in os.listdir(img_folder) if img.endswith(('png', 'jpg', 'jpeg'))]

if not background_images:
    raise Exception("No images found in the 'img' folder")

bg_index = 0
bg_image = cv2.imread(background_images[bg_index])

cap = cv2.VideoCapture(0)

def change_background(image, bg_image):
    img_out = segmentor.removeBG(image, bg_image)
    
    img_out = apply_post_processing(img_out, bg_image)
    
    return img_out

