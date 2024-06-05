from ssd_detector import SSDDetector
from utils import get_child_directory_path, configure_logging

# Call this function in your main file to configure logging
configure_logging('sys-logs')

result_img_dir = 'images_results'

try:
    ssd_detector = SSDDetector('ssd_model_lite')
    ssd_detector.process_image_and_get_predictions('/home/midtempo/SSD/images/1233.jpg', get_child_directory_path(result_img_dir))
except Exception as e:
    print("Errorka!!")
 