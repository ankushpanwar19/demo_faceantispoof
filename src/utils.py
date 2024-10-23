import math
import tensorflow as tf
import numpy as np


def load_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    return img

def calculate_angle(point1, point2):
    angle = math.atan2(point2[1]-point1[1], point2[0]-point1[0])
    deg_angle = (math.degrees(angle))
    return angle

def calcAngle(line1_pt1,line1_pt2,line2_pt1,line2_pt2):

    #calculate angle between pairs of lines
    angle1 = calculate_angle(line1_pt1,line1_pt2)
    angle2 = calculate_angle(line2_pt1,line2_pt2)
    angleDegrees = (angle1-angle2) * 360 / (2*math.pi)
    return angleDegrees

def calculate_snr(image):
    # Compute the mean of the image (signal)
    mean_signal = np.mean(image)
    
    # Compute the standard deviation (noise)
    noise = np.std(image)
    
    # Return the SNR
    return mean_signal / noise

def signal_to_noise(image):
    snr_values = {}
    
    # Split the image into its channels
    for i, color in enumerate(['Red', 'Green', 'Blue']):
        channel = image[:, :, i]
        snr_values[color] = calculate_snr(channel)
    print(snr_values)
    print(f"Average: {np.mean(list(snr_values.values()))}")
    return snr_values
    


if __name__ == '__main__':

    im_path = "data/faces/selfie.jpg"
    # im_path = "data/faces/face_selfie.png"
    # im_path = "data/faces/1.png"
    # im_path = "data/faces/0.png"
    img = load_img(im_path)
    print(f"****** {im_path} ******")
    signal_to_noise(img)