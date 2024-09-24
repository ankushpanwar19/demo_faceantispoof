import math
import tensorflow as tf

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