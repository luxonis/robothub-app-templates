from depthai_sdk import OakCamera
from robothub_oak import LiveView, BaseApplication


class ExampleApplication(BaseApplication):
    """
    This is an example application that shows how to use the OakCamera class to create an simple streaming pipeline.
    """

    def setup_pipeline(self, oak: OakCamera):  # This method has to be implemented
        """
        This method is the entrypoint for the device. Note: only one device is supported. If multiple devices are
        connected, this method will be called only once for the first device.
        OakCamera is a class from the DepthAI SDK package that provides a simple interface to create pipelines.
        Documentation for the DepthAI SDK can be found here: https://docs.luxonis.com/projects/sdk/en/latest/.
        """
        color = oak.create_camera(source="color", fps=30, resolution="1080p", encode="mjpeg")

        # Create a live view for the color camera. This will be displayed in the frontend and RobotHub app
        LiveView.create(
            device=oak,
            component=color,
            unique_key="color_stream",  # Unique key is used to identify the live view in the frontend
            name="Color stream",  # Stream name
        )
