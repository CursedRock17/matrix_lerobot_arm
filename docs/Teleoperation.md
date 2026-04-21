# Teleoperation

It's now time to start teleoperating the arm (TODO: Understanding teleop)

## Setup

1) If you've not done so already enter and source your workspace:
    ```bash
    export WORKSPACE=some_name_ws
    cd ~/path_to/${WORKSPACE} && conda activate ${WORKSPACE}
    ```
2) Power on your setup with each of the arms driver board's having their red LED on, and connected to your workspace either independently or through a USB hub (would recommend having a hub)
Ensure the ports show up:
    ```bash
    lerobot-find-port
    ```
3) On Linux, ensure they have the proper permissions:
    ```bash
    sudo chmod 666 /dev/ttyACM*
    ```
4) You must then calibrate the arms, you may use the given files in the `calirbation` section, or just go through the motions that lerobot prompts you in the coming steps. For calibration, there's no rhyme or reason with the SO101, just move them to the max of their range.
    ```bash
    mv calibration_files/robots/so101_follower ~/.cache/huggingface/lerobot/calibration/robots/so_follower
    mv calibration_files/teleoperators/so101_leader ~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader
    ```
5) Once you're all calibrated take a look at the [teleoperation script](../teleop.sh):
    TODO: mention by-id stuff, fix ports
    ```bash
    lerobot-teleoperate \
        --robot.type=so101_follower \
        --robot.port="/dev/ttyACM1" \
        --robot.id="Raven" \
        --robot.cameras="{ top: {type: opencv, index_or_path: /dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_4AF3193F-video-index0, width: 640, height: 480, fps: 30, fourcc: MJPG}, wrist.top: {type: intelrealsense, serial_number_or_name: 353322271691, width: 640, height: 480, fps: 30, use_depth: false, color_mode: RGB}}" \
        --teleop.type=so101_leader \
        --teleop.port="/dev/ttyACM0" \
        --teleop.id="Odin" \
        --display_data=true \
    ```

