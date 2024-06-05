import os
import cv2
import numpy as np
import sys
import glob
import importlib.util
import logging

from az_models import SSDDetectorResult
from utils import save_detection_result
from az_models.enums import DetectorType


# python TFLite_detection_image.py --modeldir=ssd_model_lite --imagedir images --save_results --noshow_results
class SSDDetector:
    def __init__(self, model_name, tf_lite_file_name = 'detect.tflite', labels_file_name='labelmap.txt', use_TPU = False):

        try:
            self.min_conf_threshold = float(0.5)
            self.labels = []

            # Import TensorFlow libraries
            # pkg = importlib.util.find_spec('tflite_runtime')
            # if pkg:
            #     from tflite_runtime.interpreter import Interpreter
            #     if use_TPU:
            #         from tflite_runtime.interpreter import load_delegate
            # else:
            #     from tensorflow.lite.python.interpreter import Interpreter
            #     if use_TPU:
            #         from tensorflow.lite.python.interpreter import load_delegate
            from tensorflow.lite.python.interpreter import Interpreter
        
            working_dir_path = os.getcwd()
            
            # Path to .tflite file and labels
            path_to_tf_model = os.path.join(working_dir_path, model_name, tf_lite_file_name)
            path_to_labels = os.path.join(working_dir_path, model_name, labels_file_name)

            with open(path_to_labels, 'r') as f:
                self.labels = [line.strip() for line in f.readlines()]
            
            if self.labels and self.labels[0] == '???':
                del self.labels[0]
            
            # Load the TensorFlow Lite model
            self.interpreter = Interpreter(model_path=path_to_tf_model)
            self.interpreter.allocate_tensors()

            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.height = self.input_details[0]['shape'][1]
            self.width = self.input_details[0]['shape'][2]
            logging.info("Height value: %s", self.height)
            logging.info("Width value: %s", self.width)

            self.floating_model = (self.input_details[0]['dtype'] == np.float32)
            self.input_mean = 127.5
            self.input_std = 127.5
            
            # Check output layer name to determine model type
            outname = self.output_details[0]['name']
            
            if 'StatefulPartitionedCall' in outname: # TF2 model
                self.boxes_idx, self.classes_idx, self.scores_idx = 1, 3, 0
            else: # TF1 model
                self.boxes_idx, self.classes_idx, self.scores_idx = 0, 1, 2
        except Exception as e:
            logging.error("Error occurred during SSDDetector __init__: %s", e)
            pass  # Ignore errors and continue code execution
            
    def process_image_and_get_predictions(self, img_path, result_img_dir_path) -> SSDDetectorResult:
        
        try:
            
             # Load image and resize to expected shape [1xHxWx3]
            image = cv2.imread(img_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            imH, imW, _ = image.shape 
            image_resized = cv2.resize(image_rgb, (self.width, self.height))
            input_data = np.expand_dims(image_resized, axis=0)
            
             # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
            if self.floating_model:
                input_data = (np.float32(input_data) - self.input_mean) / self.input_std

            # Perform the actual detection by running the model with the image as input
            self.interpreter.set_tensor(self.input_details[0]['index'],input_data)
            self.interpreter.invoke()
            
            # Retrieve detection results
            boxes = self.interpreter.get_tensor(self.output_details[self.boxes_idx]['index'])[0] # Bounding box coordinates of detected objects
            classes = self.interpreter.get_tensor(self.output_details[self.classes_idx]['index'])[0] # Class index of detected objects
            scores = self.interpreter.get_tensor(self.output_details[self.scores_idx]['index'])[0] # Confidence of detected objects

            detections = []
            
            # Loop over all detections and draw detection box if confidence is above minimum threshold
            for i in range(len(scores)):
                if ((scores[i] > self.min_conf_threshold) and (scores[i] <= 1.0)):

                    # Get bounding box coordinates and draw box
                    # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                    ymin = int(max(1,(boxes[i][0] * imH)))
                    xmin = int(max(1,(boxes[i][1] * imW)))
                    ymax = int(min(imH,(boxes[i][2] * imH)))
                    xmax = int(min(imW,(boxes[i][3] * imW)))
                    
                    cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

                    # Draw label
                    object_name = self.labels[int(classes[i])] # Look up object name from "labels" array using class index
                    label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                    label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                    cv2.rectangle(image, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                    cv2.putText(image, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

                    detections.append([object_name, scores[i], xmin, ymin, xmax, ymax])
                    
                     # Get filenames and paths
                    image_fn = os.path.basename(img_path)

                    # Save image
                    det_img_filename, det_img_path = save_detection_result(image, result_img_dir_path, image_fn, DetectorType.SSD)
                    
                    highest_confidence_detection_label = ""  # Default label
                    highest_confidence_score = 0.0  # Default score
                    
                    for detection in detections:
                        confidence_score = float(detection[1])  # Convert confidence score to float
                        if confidence_score > highest_confidence_score:
                            highest_confidence_detection_label = detection[0]
                            highest_confidence_score = confidence_score
                    
                    # Log the detection result
                    result = SSDDetectorResult(label=highest_confidence_detection_label, value=highest_confidence_score, det_img_filename=det_img_filename, det_img_path=det_img_path)
                    logging.info(f"Detection result: {result}")

                    # Return the detection result
                    return result    
        
        except Exception as e:
            logging.error("Error occurred during process_image_and_get_predictions: %s", e)
            return SSDDetectorResult(error_message=str(e))