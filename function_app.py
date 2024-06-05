import json
import logging
import os

import azure.durable_functions as df
import azure.functions as func

from managers import AzureBlobManager, AzureTableStorageManager
from az_models import BlobToProcessQueueMessage, VisioDetectorHttpRequest, ImagePredictionResult
from az_models.enums import BlobProcessStatus, DetectorType
from ssd_detector import SSDDetector
from utils import get_child_directory_path, configure_logging

# Assuming you have configured logging already
configure_logging('sys-logs')

blob_container_name = os.environ.get("BlobContainerName")
blob_connection_string = os.environ.get("BlobConnectionString")
azure_blob_manager = AzureBlobManager(blob_connection_string, blob_container_name)

predictions_blob_container_name = os.environ.get("ProcessedBlobsContainerName")
predictions_azure_blob_manager = AzureBlobManager(blob_connection_string, predictions_blob_container_name)

table_storage_name = os.environ.get("TableStorageName")
table_connection_string = os.environ.get("TableConnectionString")
azure_table_storage_manager = AzureTableStorageManager(table_connection_string, table_storage_name)

source_img_dir = 'images'
result_img_dir = 'images_results'
        
# We can provide a key, and use function level: https://learn.microsoft.com/en-us/azure/azure-functions/functions-bindings-http-webhook-trigger?tabs=python-v2%2Cisolated-process%2Cnodejs-v4%2Cfunctionsv2&pivots=programming-language-python
app = df.DFApp(http_auth_level=func.AuthLevel.ANONYMOUS)

try:
    ssd_detector = SSDDetector('ssd_model_lite')
except Exception as ex:
    logging.error("SSD: Func App initialisation - error occurred while initialising SSD Detector: %s", ex)
    

@app.function_name(name="ObjectDetectionHttpTrigger")
@app.route(route="ssd/detect", methods=("POST",))
@app.durable_client_input(client_name="client")
async def http_start(req: func.HttpRequest, client: df.DurableOrchestrationClient):
    req_body = req.get_body().decode('utf-8')
    logging.info(f"Started ObjectDetectionHttpTrigger, received: {req_body}")

    instance_id = await client.start_new("image_detection_orchestrator", client_input=req_body)
    
    await client.wait_for_completion_or_create_check_status_response(req, instance_id)     

    # Get orchestration execution status
    status = await client.get_status(instance_id)     

    # Retrieves orchestration execution results and displays them on the screen
    runtime = status.runtime_status
    output = status.output
    logging.info(f"runtime: {runtime}\n\n output:{output}")

    return output

@app.orchestration_trigger(context_name="context")
def image_detection_orchestrator(context: df.DurableOrchestrationContext):
    visio_detector_req_str = context.get_input()
    visio_detector_json = json.loads(visio_detector_req_str)
    logging.info(f"image_detection_orchestrator.visio_detector_json: {visio_detector_json}")

    try:
        visio_detector_req = VisioDetectorHttpRequest.from_json(visio_detector_json)
        logging.info(f"image_detection_orchestrator.detector_type: {visio_detector_req.detector_type}")

        if visio_detector_req.detector_type == DetectorType.SSD:
            result = yield context.call_activity("run_ssd_detection_activity", visio_detector_req_str)
        else:
            result = ImagePredictionResult(
                image_name=visio_detector_req.file_name,
                detector_type=visio_detector_req.detector_type,
                errors= "The detector type is incorrect, must be SSD",
                has_errors=True
                ).to_json()

        logging.info(f"image_detection_orchestrator.result: {result}")
        return result
    except Exception as ex:
        logging.error(f"An unexpected error occurred: {ex}")
        return ImagePredictionResult(
            image_name=visio_detector_json['fileName'],
            detector_type=DetectorType.SSD,
            errors= str(ex),
            has_errors=True
            ).to_json()
        
@app.activity_trigger(input_name="visioDetectorReqStr")
def run_ssd_detection_activity(visioDetectorReqStr: str) -> str:
    visioDetectorModel = VisioDetectorHttpRequest.from_json(json.loads(visioDetectorReqStr))

    logging.info(f"Running SSD detection for: {visioDetectorModel.file_name}")

    try:
        logging.info(f"Detector Type: {visioDetectorModel.detector_type}")

        file_download_dir = get_child_directory_path(source_img_dir)
        source_img_path = azure_blob_manager.download_and_upload_file(visioDetectorModel.file_name, file_download_dir)

        detector_type = DetectorType.SSD
        
        result_img_dir_path = get_child_directory_path(result_img_dir)
        logging.info(f"result_img_dir_path: {result_img_dir_path}")
        logging.info(f"source_img_path: {source_img_path}")
        
        ssd_prediction_result = ssd_detector.process_image_and_get_predictions(source_img_path, result_img_dir_path)
        logging.info(f"ssd_prediction_result: {ssd_prediction_result}")
        
        if ssd_prediction_result.error_message:
            return ImagePredictionResult(
                image_name=visioDetectorModel.file_name,
                detector_type=detector_type,
                errors=ssd_prediction_result.error_message,
                has_errors=True
            ).to_json()

        image_prediction_response = ImagePredictionResult(
            image_name=visioDetectorModel.file_name,
            detector_type=detector_type,
            classification = ssd_prediction_result.label,
            result_img_name=ssd_prediction_result.det_img_filename,
            result_img_path=ssd_prediction_result.det_img_path,
            prediction=float(ssd_prediction_result.value),
            time_taken=0.5)
        
        predictions_azure_blob_manager.upload_file_to_blob(ssd_prediction_result.det_img_path, ssd_prediction_result.det_img_filename)
        
        logging.info(f"run_ssd_detection_activity.Result: {image_prediction_response}")

        # delete_file_if_exists(downloaded_file_path)
        return image_prediction_response.to_json()

    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return ImagePredictionResult(
            image_name=visioDetectorModel.file_name,
            detector_type=visioDetectorModel.detector_type,
            errors= str(e),
            has_errors=True
            ).to_json()