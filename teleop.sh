lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port="/dev/ttyACM1" \
    --robot.id="Raven" \
    --robot.cameras="{ top: {type: opencv, index_or_path: /dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_4AF3193F-video-index0, width: 640, height: 480, fps: 30, fourcc: MJPG}, wrist.top: {type: intelrealsense, serial_number_or_name: 353322271691, width: 640, height: 480, fps: 30, use_depth: false, color_mode: RGB}}" \
    --teleop.type=so101_leader \
    --teleop.port="/dev/ttyACM0" \
    --teleop.id="Odin" \
    --display_data=true \
    

# intel config
#--robot.cameras="{ wrist.top: {type: opencv, index_or_path: /dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_4AF3193F-video-index0, width: 640, height: 480, fps: 30, fourcc: MJPG}, base: {type: intelrealsense, serial_number_or_name: 353322271691, width: 640, height: 480, fps: 30, use_depth: false, color_mode: RGB}}" \
#--robot.cameras="{ base: {type: opencv, index_or_path: /dev/video2, width: 640, height: 480, fps: 30, fourcc: mjpg}, wrist.top: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 30, fourcc: mjpg}}" \
