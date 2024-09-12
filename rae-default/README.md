# Video Streams

This App launches robot class and uploads a DepthAI pipeline to RAE to stream each camera output to RobotHub.

You can easily replace with you desired sockets that what to be streamed in `streams` dictionary. Also for configuring the pipeline you can edit `build_pipeline()`, such as resolution, fps and more. On how to work with Robot class, please refer to [RAE SDK Docs](https://docs-beta.luxonis.com/develop/use-with-ros/rae-sdk/).

## Requirements

- RAE or a RVC3 device.

## Usage

- Assign devices to the App and launch it.
