import rosbag
import roslib
# roslib.load_manifest(PKG)
import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

bag_file = '2022-09-30-16-02-46.bag'
bag = rosbag.Bag(bag_file, "r")

bridge = CvBridge()
bag_data = bag.read_messages('/camera/image_color/compressed')
for topic, msg, t in bag_data:
    cv_image = bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
    cv2.imshow("Image window", cv_image)
    timestr = "%.6f" % msg.header.stamp.to_sec()
    image_name = timestr + ".jpg"
    cv2.imwrite(image_name, cv_image)
    cv2.waitKey(3)
