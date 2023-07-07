import robothub

from robothub_oak.manager import DEVICE_MANAGER
from robothub_oak.packets import HubPacket

"""
RobotHub-OAK is a high-level library that allows you to easily create applications with Luxonis devices.
It is built on top of the RobotHub and DepthAI SDK libraries and provides a simple interface for creating custom applications.
RobotHub-OAK deals with all the low-level details of the SDK and allows you to focus on the application logic.
"""


class ExampleApplication(robothub.RobotHubApplication):
    """
    This is an example application that streams color, stereo and NN detections from all connected devices.
    All RobotHub applications must inherit from robothub.RobotHubApplication and implement the following methods:
    - on_start: This method is called when the application is started.
    - start_execution: This method is called after on_start and is an entry point for the application.
    - on_stop: This method is called when the application is stopped.
    """
    color_resolution = '1080p'
    mono_resolution = '400p'

    def on_start(self):
        # DEVICE_MANAGER is a global instance of DeviceManager that is used to manage all devices.
        # It can be used to get all devices, get a specific device, start and stop the device manager.

        devices = DEVICE_MANAGER.get_all_devices()  # Get all assigned devices to this application.
        # devices = [DEVICE_MANAGER.get_device(mxid=...), ...] # Get a specific device by its mxid

        for device in devices:
            color = device.get_camera('color', resolution=self.color_resolution, fps=30)
            color.stream_to_hub(name=f'Color stream {device.get_device_name()}')

            stereo = device.get_stereo_camera(resolution=self.mono_resolution, fps=30)
            stereo.stream_to_hub(name=f'Stereo stream {device.get_device_name()}')

            nn = device.create_neural_network('yolov6nr3_coco_640x352', input=color)
            nn.stream_to_hub(name=f'NN stream {device.get_device_name()}')
            nn.add_callback(self.custom_callback)

    def custom_callback(self, packet: HubPacket):
        """
        This is an example of a custom callback that can be used to process packets received from the device.
        This method is called for every packet received. The type of the packet depends on the type of the output.
        For example, if the output is a camera, the packet will be of type `robothub_oak.packets.HubPacket`,
        if the output is a neural network, the packet will be of type `robothub_oak.packets.DetectionPacket`.
        For more information about the different types of packets, refer to the documentation pages.
        :param packet: The packet received from the device.
        """
        print(f'Packet received from device {packet.device.get_device_name()} with type {type(packet)}')

    def start_execution(self):
        DEVICE_MANAGER.start()  # Start all devices and start streaming. Blocks until application is stopped.

    def on_stop(self):
        DEVICE_MANAGER.stop()  # Close all devices and streams.
