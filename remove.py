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

def apply_post_processing(foreground, background):
    foreground = foreground.astype(float)
    background = background.astype(float)
    
    alpha = foreground[:, :, 3] / 255.0 if foreground.shape[2] == 4 else np.ones(foreground.shape[:2], dtype=float)
    
    blended = alpha[..., np.newaxis] * foreground + (1 - alpha[..., np.newaxis]) * background
    
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    
    return blended

while True:
    success, frame = cap.read()
    if not success:
        break

    height, width, _ = frame.shape

    bg_image_resized = cv2.resize(bg_image, (width, height))

    output_frame = change_background(frame, bg_image_resized)

    cv2.imshow('Background Replacement', output_frame)

    key = cv2.waitKey(1)
    if key == 27:  
        break
    elif key == ord('a'):  
        bg_index = (bg_index - 1) % len(background_images)
        bg_image = cv2.imread(background_images[bg_index])
    elif key == ord('d'):  
        bg_index = (bg_index + 1) % len(background_images)
        bg_image = cv2.imread(background_images[bg_index])

cap.release()
cv2.destroyAllWindows()
