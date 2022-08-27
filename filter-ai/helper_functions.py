from PIL import Image
from io import BytesIO
import glob
import os
import cv2
import base64
import numpy as np
import tensorflow as tf



def jpg_to_png(source_path, target_path):
    all_jpg = glob.glob(source_path)
    print("Found " + str(len(all_jpg)) + " .jpg files")

    i = 0
    for image_path in all_jpg:
        temp_img = Image.open(image_path)
        temp_img.save(target_path + "\\VIRS_new_img" + str(i) + ".png")
        i = i + 1

def image_corruption_check(path):
    all_image = os.listdir(path)
    isCorrupted = False
    for image in all_image:
        try:
            Image.open(path + "\\" + image)
        except Exception as e:
            print(e)
            print(path + "\\" + image)
            isCorrupted = True
    if not isCorrupted:
        print("No corrupted file found")

def gif_to_png(source_path, target_path):
    all_gif = glob.glob(source_path)
    print("Found " + str(len(all_gif)) + " .gif files")

    i = 0
    for gif_path in all_gif:
        temp_gif = Image.open(gif_path)
        print("Processing gif: " + gif_path + ",  with nframe: " + str(temp_gif.n_frames))
        try:
            temp_gif.seek(int(temp_gif.n_frames / 16))
            temp_gif.save(target_path + "\\img_from_gif" + str(i) + ".png")
            i = i + 1
            temp_gif.seek(int(temp_gif.n_frames / 8))
            temp_gif.save(target_path + "\\img_from_gif" + str(i) + ".png")
            i = i + 1
            temp_gif.seek(int(temp_gif.n_frames / 4))
            temp_gif.save(target_path + "\\img_from_gif" + str(i) + ".png")
            i = i + 1
            temp_gif.seek(int(temp_gif.n_frames / 2))
            temp_gif.save(target_path + "\\img_from_gif" + str(i) + ".png")
            i = i + 1
        except:
            print("error processing gif, skipping")

def video_to_png(source_path, target_path):
    all_mp4 = glob.glob(source_path)
    print("Found " + str(len(all_mp4)) + " .mp4 files")

    img_num = 0
    for mp4_path in all_mp4:
        try:
            print("Extracting from mp4: " + mp4_path)
            cam = cv2.VideoCapture(mp4_path)
        except Exception as e:
            print(e)
            print("error opening, skipping...")
            continue
        
        current_frame = 0
        while (True):
            ret,frame = cam.read()
            if ret:
                if current_frame % 120 == 0:
                    cv2.imwrite(target_path + "\\img_from_mp4" + str(img_num) + ".png", frame)
                    print("create image num: #" + str(img_num))
                    img_num += 1
                current_frame += 1
            else:
                break
        cam.release()
    cv2.destroyAllWindows()


source = r'C:\Users\Mango\Documents\Datasets\original_reddit_violent_images\violent_image_reddit_scraper\*.jpg'
target = r'C:\Users\Mango\Documents\Datasets\Data_converted'

test_image = r'C:\Users\Mango\Documents\Datasets\test_image\normal\322868_1100-800x825.jpeg'


def image_base64_string(path):
    with open(path, 'rb') as image:
        encoded_string = base64.b64encode(image.read())
    return encoded_string

def base64_to_array(base64_string):
    im = Image.open(BytesIO(base64.b64decode(base64_string)))
    img_bgr = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img_bgr, (200, 200))
    img = [img]
    img = np.array(img) / 255
    return img