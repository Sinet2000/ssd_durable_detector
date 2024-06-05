from dataclasses import dataclass
from decimal import Decimal

@dataclass
class SSDDetectorResult:
    label: str = "UNKNOWN"
    value: Decimal = Decimal(0.0)
    error_message: str = ""
    det_img_filename: str = ""
    det_img_path: str = ""
    
    def __str__(self):
        return f"Label: {self.label}, Value: {self.value}, Error Message: {self.error_message}, Detected Image Filename: {self.det_img_filename}, Detected Image Path: {self.det_img_path}"