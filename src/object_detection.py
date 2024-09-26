import numpy as np
import tensorflow as tf
import json

from src.utils import load_img

def get_classes(json_path="models/object_detection/classes.json"):

    with open(json_path, 'r') as file:
        obj_classes = json.load(file)

    return obj_classes

def load_objdetection_model(model_path):
    interpreter = tf.lite.Interpreter(model_path= model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the model on random input data.
    input_shape = input_details[0]['shape']
    image_path = "data/images/phone.png"
    img = load_img(image_path)
    img_resized = tf.image.resize(img,[320,320],preserve_aspect_ratio=False,antialias=False,name=None)
    # img_resized = np.array(np.random.random_sample(img_resized), dtype=np.uint8)
    img_resized = tf.cast(img_resized, tf.uint8)
    input_data = np.expand_dims(img_resized.numpy(), axis = 0)
    # input_data = input_data.numpy()
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    detected_bbox = interpreter.get_tensor(output_details[0]['index'])
    detected_labels = interpreter.get_tensor(output_details[1]['index'])
    detected_score = interpreter.get_tensor(output_details[2]['index'])
    # print(f"TFLITE output : {output_data}")
    obj_classes = get_classes()
    return interpreter, obj_classes

detector,obj_classes = load_objdetection_model("models/object_detection/lite-model_efficientdet_lite0_detection_default_1.tflite")

def run_detector(img):
        
    h,w,_ = img.shape
    # model takes [1,320,320,3] image with data type as uint8
    img_resized = tf.image.resize(img,[320,320],preserve_aspect_ratio=False,antialias=False,name=None)
    img_resized = tf.cast(img_resized, tf.uint8)
    input_data = np.expand_dims(img_resized.numpy(), axis = 0)

    result = {}

    input_details = detector.get_input_details()
    output_details = detector.get_output_details()
    detector.set_tensor(input_details[0]['index'], input_data)

    detector.invoke()

    result['detection_boxes'] = detector.get_tensor(output_details[0]['index'])
    result['detection_class_labels'] = detector.get_tensor(output_details[1]['index'])
    result['detection_scores'] = detector.get_tensor(output_details[2]['index'])

    THRESHOLD = 0.2
    detected_parts = result['detection_class_labels'][result['detection_scores']>THRESHOLD]
    detected_bbox = result['detection_boxes'][result['detection_scores']>THRESHOLD] *  np.array([h,w,h,w])
    detected_bbox = detected_bbox.astype(np.int32)

    obj_detected =[]
    obj_bbox = []
    # check if object is ['laptop','tv','cellphone','book']
    for i,obj in enumerate(detected_parts):
        obj = obj_classes[str(int(detected_parts[i]+1))]
        # if obj in ['laptop','tv','cellphone','book']:
        obj_detected.append(obj)
        ymin,xmin,ymax,xmax = detected_bbox[i]
        ymin = max(0,ymin)
        xmin = max(0,xmin)
        ymax = min(h,ymax)
        xmax = min(w,xmax)
        obj_bbox.append([ymin,xmin,ymax,xmax])


    return obj_detected, obj_bbox


if __name__ == "__main__":
    
    import cv2

    cap = cv2.VideoCapture(0)
    i=0
    while True:
        ret, frame = cap.read()
        rgb_frame_org = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        obj_detected, obj_bbox = run_detector(rgb_frame_org)
        
        cv2.putText(frame, "Obj Detected: "+str(obj_detected), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Face Detection and Keypoints', frame)
        # Exit loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
