import cv2
import time

import blobconverter
import depthai as dai
import robothub as rh


class BusinessLogic:
    def __init__(self):
        self.last_image_event_upload_seconds = time.time()
        self.last_video_event_upload_seconds = time.time()

    def process_pipeline_outputs(self, h264_frame: dai.ImgFrame, mjpeg_frame: dai.ImgFrame, object_detections: dai.ImgDetections):
        if rh.LOCAL_DEV:
            cv2.imshow("Live view", cv2.imdecode(mjpeg_frame.getCvFrame(), cv2.IMREAD_COLOR))
            cv2.waitKey(1)
        for detection in object_detections.detections:
            # visualize bounding box in the live view
            bbox = (detection.xmin, detection.ymin, detection.xmax, detection.ymax)
            # self.live_view.add_rectangle(bbox, label=detection.label)
            #
            # current_time_seconds = time.time()
            # # arbitrary condition for sending image events to RobotHub
            # if current_time_seconds - self.last_image_event_upload_seconds > rh.CONFIGURATION["image_event_upload_interval_minutes"] * 60:
            #     if detection.label == 'person':
            #         self.last_image_event_upload_seconds = current_time_seconds
            #         send_image_event(image=packet.frame, title='Person detected')
            # # arbitrary condition for sending video events to RobotHub
            # if current_time_seconds - self.last_video_event_upload_seconds > rh.CONFIGURATION["video_event_upload_interval_minutes"] * 60:
            #     if detection.label == 'person':
            #         self.last_video_event_upload_seconds = current_time_seconds
            #         self.live_view.save_video_event(before_seconds=60, after_seconds=60, title="Interesting video")


class Application(rh.BaseDepthAIApplication):
    business_logic = BusinessLogic()

    def setup_pipeline(self) -> dai.Pipeline:
        """Define the pipeline using DepthAI."""

        pipeline = dai.Pipeline()
        rgb_sensor = create_rgb_sensor(pipeline=pipeline, preview_resolution=(640, 352))
        rgb_h264_encoder = create_h264_encoder(node_input=rgb_sensor.video, pipeline=pipeline)
        rgb_mjpeg_encoder = create_mjpeg_encoder(node_input=rgb_sensor.video, pipeline=pipeline)
        object_detection_nn =  create_object_detecting_nn(node_input=rgb_sensor.preview, pipeline=pipeline, model="yolov7tiny_coco_640x352")

        create_output(pipeline=pipeline, node_input=rgb_h264_encoder.bitstream, stream_name="h264_frames")
        create_output(pipeline=pipeline, node_input=rgb_mjpeg_encoder.bitstream, stream_name="mjpeg_frames")
        create_output(pipeline=pipeline, node_input=object_detection_nn.out, stream_name="object_detections")
        return pipeline

    def manage_device(self, device: dai.Device):
        h264_frames_queue = device.getOutputQueue(name="h264_frames", maxSize=10, blocking=True)
        mjpeg_frames_queue = device.getOutputQueue(name="mjpeg_frames", maxSize=10, blocking=True)
        object_detections_queue = device.getOutputQueue(name="object_detections", maxSize=10, blocking=True)

        while rh.app_is_running():
            h264_frame = h264_frames_queue.get()
            mjpeg_frame = mjpeg_frames_queue.get()
            object_detections = object_detections_queue.get()
            self.business_logic.process_pipeline_outputs(h264_frame=h264_frame, mjpeg_frame=mjpeg_frame, object_detections=object_detections)
            time.sleep(0.001)


def create_rgb_sensor(pipeline: dai.Pipeline,
                      fps: int = 30,
                      resolution: dai.ColorCameraProperties.SensorResolution = dai.ColorCameraProperties.SensorResolution.THE_1080_P,
                      preview_resolution: tuple = (1280, 720),
                      ) -> dai.node.ColorCamera:
    node = pipeline.createColorCamera()
    node.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    node.setInterleaved(False)
    node.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    node.setPreviewNumFramesPool(4)
    node.setPreviewSize(*preview_resolution)
    node.setVideoSize(1920, 1080)
    node.setResolution(resolution)
    node.setFps(fps)
    return node


def create_h264_encoder(node_input: dai.Node.Output, pipeline: dai.Pipeline, fps: int = 30):
    rh_encoder = pipeline.createVideoEncoder()
    rh_encoder_profile = dai.VideoEncoderProperties.Profile.H264_MAIN
    rh_encoder.setDefaultProfilePreset(fps, rh_encoder_profile)
    rh_encoder.input.setQueueSize(2)
    rh_encoder.input.setBlocking(False)
    rh_encoder.setKeyframeFrequency(fps)
    rh_encoder.setRateControlMode(dai.VideoEncoderProperties.RateControlMode.CBR)
    rh_encoder.setNumFramesPool(3)
    node_input.link(rh_encoder.input)
    print(rh_encoder.getWidth())
    print(rh_encoder.getHeight())
    return rh_encoder


def create_mjpeg_encoder(node_input: dai.Node.Output, pipeline: dai.Pipeline, fps: int = 30, quality: int = 100):
    encoder = pipeline.createVideoEncoder()
    encoder_profile = dai.VideoEncoderProperties.Profile.MJPEG
    encoder.setDefaultProfilePreset(fps, encoder_profile)
    encoder.setQuality(quality)
    node_input.link(encoder.input)
    return encoder


def create_image_manip(node_input: dai.Node.Output, pipeline: dai.Pipeline, resize: tuple[int, int], keep_aspect_ration: bool = False,
                       frame_type: dai.RawImgFrame.Type = dai.RawImgFrame.Type.BGR888p, output_frame_dims: int = 3,
                       blocking_input_queue: bool = False, input_queue_size: int = 4, frames_pool: int = 4,
                       wait_for_config: bool = False) -> dai.node.ImageManip:
    image_manip = pipeline.createImageManip()
    image_manip.initialConfig.setResize(*resize)
    image_manip.initialConfig.setFrameType(frame_type)
    image_manip.initialConfig.setKeepAspectRatio(keep_aspect_ration)
    image_manip.setWaitForConfigInput(wait_for_config)
    image_manip.setNumFramesPool(frames_pool)
    image_manip.setMaxOutputFrameSize(resize[0] * resize[1] * output_frame_dims)
    image_manip.inputImage.setBlocking(blocking_input_queue)
    image_manip.inputImage.setQueueSize(input_queue_size)
    node_input.link(image_manip.inputImage)
    return image_manip


def create_object_detecting_nn(node_input: dai.Node.Output, pipeline: dai.Pipeline, model: str) -> dai.node.YoloDetectionNetwork:
    node = pipeline.createYoloDetectionNetwork()
    blob = dai.OpenVINO.Blob(blobconverter.from_zoo(name=model, zoo_type="depthai", shaves=6))
    node.setBlob(blob)
    node_input.link(node.input)
    node.input.setBlocking(False)
    # Yolo specific parameters
    node.setConfidenceThreshold(0.5)
    node.setNumClasses(80)
    node.setCoordinateSize(4)
    node.setAnchors([12.0, 16.0, 19.0, 36.0, 40.0, 28.0, 36.0, 75.0, 76.0, 55.0, 72.0, 146.0, 142.0, 110.0, 192.0, 243.0, 459.0, 401.0])
    node.setAnchorMasks({
                "side80": [0, 1, 2],
                "side40": [3, 4, 5],
                "side20": [6, 7, 8]
            })
    node.setIouThreshold(0.5)
    return node


def create_output(pipeline, node_input: dai.Node.Output, stream_name: str):
    xout = pipeline.createXLinkOut()
    xout.setStreamName(stream_name)
    node_input.link(xout.input)


if __name__ == "__main__":
    app = Application()
    app.run()
