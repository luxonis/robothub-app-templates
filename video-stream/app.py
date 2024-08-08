import logging as log
import time

import blobconverter
import depthai as dai
import object_detector_config as nn_config
import robothub as rh


# [business logic]
class BusinessLogic:
    def __init__(self, frame_buffer: rh.FrameBuffer, live_view: rh.DepthaiLiveView):
        self.live_view: rh.DepthaiLiveView = live_view
        self.frame_buffer: rh.FrameBuffer = frame_buffer

        self.last_image_event_upload_seconds = time.time()
        self.last_video_event_upload_seconds = time.time()

    def process_pipeline_outputs(self, h264_frame: dai.ImgFrame, mjpeg_frame: dai.ImgFrame, object_detections: dai.ImgDetections):
        self.frame_buffer.add_frame(h264_frame) # make sure to store every h264 frame
        for detection in object_detections.detections:
            # visualize bounding box in the live view
            bbox = (detection.xmin, detection.ymin, detection.xmax, detection.ymax)
            self.live_view.add_rectangle(bbox, label=nn_config.labels[detection.label])

            current_time_seconds = time.time()
            # arbitrary condition for sending image events to RobotHub
            if current_time_seconds - self.last_image_event_upload_seconds > rh.CONFIGURATION["image_event_upload_interval_minutes"] * 60:
                if nn_config.labels[detection.label] == 'person':
                    self.last_image_event_upload_seconds = current_time_seconds
                    rh.send_image_event(image=mjpeg_frame.getCvFrame(), title='Person detected')
            # arbitrary condition for sending video events to RobotHub
            if current_time_seconds - self.last_video_event_upload_seconds > rh.CONFIGURATION["video_event_upload_interval_minutes"] * 60:
                if nn_config.labels[detection.label] == 'person':
                    self.last_video_event_upload_seconds = current_time_seconds
                    self.frame_buffer.save_video_event(before_seconds=60, after_seconds=60, title="Interesting video",
                                                       fps=rh.CONFIGURATION["fps"], frame_width=self.live_view.frame_width,
                                                       frame_height=self.live_view.frame_height)
        self.live_view.publish(h264_frame=h264_frame.getCvFrame())

# [/business logic]


# [application]
class Application(rh.BaseDepthAIApplication):

    def __init__(self):
        super().__init__()
        self.live_view = rh.DepthaiLiveView(name="live_view", unique_key="rgb",
                                            width=1920, height=1080)
        frame_buffer = rh.FrameBuffer(maxlen=rh.CONFIGURATION["fps"] * 60 * 2)  # buffer last 2 minutes
        self.business_logic = BusinessLogic(frame_buffer=frame_buffer, live_view=self.live_view)

# [setup pipeline]
    def setup_pipeline(self) -> dai.Pipeline:
        """Define the pipeline using DepthAI."""

        log.info(f"App config: {rh.CONFIGURATION}")
        pipeline = dai.Pipeline()
        rgb_sensor = create_rgb_sensor(pipeline=pipeline, preview_resolution=(640, 352))
        rgb_h264_encoder = create_h264_encoder(node_input=rgb_sensor.video, pipeline=pipeline)
        rgb_mjpeg_encoder = create_mjpeg_encoder(node_input=rgb_sensor.video, pipeline=pipeline)
        object_detection_nn = create_yolov7tiny_coco_nn(node_input=rgb_sensor.preview, pipeline=pipeline)

        create_output(pipeline=pipeline, node_input=rgb_h264_encoder.bitstream, stream_name="h264_frames")
        create_output(pipeline=pipeline, node_input=rgb_mjpeg_encoder.bitstream, stream_name="mjpeg_frames")
        create_output(pipeline=pipeline, node_input=object_detection_nn.out, stream_name="object_detections")
        return pipeline

# [/setup pipeline]

# [main loop]
    def manage_device(self, device: dai.Device):
        log.info(f"{device.getMxId()} creating output queues...")
        h264_frames_queue = device.getOutputQueue(name="h264_frames", maxSize=10, blocking=True)
        mjpeg_frames_queue = device.getOutputQueue(name="mjpeg_frames", maxSize=10, blocking=True)
        object_detections_queue = device.getOutputQueue(name="object_detections", maxSize=10, blocking=True)

        log.info(f"{device.getMxId()} Application started")
        while rh.app_is_running() and self.device_is_running:
            h264_frame = h264_frames_queue.get()
            mjpeg_frame = mjpeg_frames_queue.get()
            object_detections = object_detections_queue.get()
            self.business_logic.process_pipeline_outputs(h264_frame=h264_frame, mjpeg_frame=mjpeg_frame, object_detections=object_detections)
            time.sleep(0.001)

# [/main loop]

# [config change listener]
    def on_configuration_changed(self, configuration_changes: dict) -> None:
        log.info(f"CONFIGURATION CHANGES: {configuration_changes}")
        if "fps" in configuration_changes:
            log.info(f"FPS change needs a new pipeline. Restarting OAK device...")
            self.restart_device()

# [/config change listener]
# [/application]


# [pipeline]
# [rgb sensor]
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
    node.setPreviewKeepAspectRatio(True) #True: letter-boxing, False: resizing. Resizing recommended for square preview and model.
    node.setPreviewSize(*preview_resolution)
    node.setVideoSize(1920, 1080)
    node.setResolution(resolution)
    node.setFps(fps)
    return node
# [/rgb sensor]


# [encoders]
# [h encoder]
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
    return rh_encoder
# [/h encoder]


# [mjpeg encoder]
def create_mjpeg_encoder(node_input: dai.Node.Output, pipeline: dai.Pipeline, fps: int = 30, quality: int = 100):
    encoder = pipeline.createVideoEncoder()
    encoder_profile = dai.VideoEncoderProperties.Profile.MJPEG
    encoder.setDefaultProfilePreset(fps, encoder_profile)
    encoder.setQuality(quality)
    node_input.link(encoder.input)
    return encoder
# [/mjpeg encoder]
# [/encoders]


# [yolo nn]
def create_yolov7tiny_coco_nn(node_input: dai.Node.Output, pipeline: dai.Pipeline) -> dai.node.YoloDetectionNetwork:
    model = "yolov7tiny_coco_640x352"
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
# [/yolo nn]


# [xlink out]
def create_output(pipeline, node_input: dai.Node.Output, stream_name: str):
    xout = pipeline.createXLinkOut()
    xout.setStreamName(stream_name)
    node_input.link(xout.input)
# [/xlink out]
# [/pipeline]


# [local development]
if __name__ == "__main__":
    app = Application()
    app.run()
# [/local development]
