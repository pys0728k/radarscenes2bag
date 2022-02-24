import json
import os
from re import X
import sys
import h5py
import signal

import numpy as np
import numpy.lib.recfunctions as rfn
import matplotlib.pyplot as plt
import enum
from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError
from pkg_resources import resource_filename

from radar_scenes.sequence import Sequence
from radar_scenes.scene import Scene
from radar_scenes.coordinate_transformation import transform_detections_sequence_to_car
from radar_scenes.sensors import get_mounting
from radar_scenes.colors import Colors, brush_for_color
from radar_scenes.evaluation import PredictionFileSchemas
from radar_scenes.labels import Label

import rosbag
import rospkg
import rospy
import tf
import time
from geometry_msgs.msg import Point, Pose, PoseStamped, Transform, TransformStamped, Quaternion
from tf2_msgs.msg import TFMessage
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, PointField, CameraInfo, Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError


RADAR_DEFAULT_MOUNTING = {
    1: {"x": 3.663, "y": -0.873, "yaw": -1.48418552},
    2: {"x": 3.86, "y": -0.70, "yaw": -0.436185662},
    3: {"x": 3.86, "y": 0.70, "yaw": 0.436},
    4: {"x": 3.663, "y": 0.873, "yaw": 1.484},
}

image_conversion = False
path_to_dataset = []

class dataLoader():

    def __init__(self, seq_num):
        super().__init__()
        self.seq_num = seq_num

    def open_sequence(self):
        """
        Dialog for opening a measurement sequence.
        Only json files can be loaded. Actual loading is done by the load_sequence function.
        :return: None
        """
        seq = "sequence_" + str(self.seq_num)
        self.filename = os.path.join(path_to_dataset, "data", seq, "scenes.json")
        if not os.path.exists(self.filename):
            print("Please modify this example so that it contains the correct path to the dataset on your machine.")
            return

        """
        """
        self.filename_h5 = os.path.join(path_to_dataset, "data", seq, "radar_data.h5")
        if not os.path.exists(self.filename_h5):
            print("Please modify this example so that it contains the correct path to the dataset on your machine.")
            return

    def load_sequence(self):
        if not self.filename.endswith(".json") or not os.path.exists(self.filename):
            return

        try:
            self.sequence = Sequence.from_json(self.filename)
            print('Sequence' + str(self.seq_num) + ' Loading Finished.')
        except:
            (type, value, traceback) = sys.exc_info()
            sys.excepthook(type, value, traceback)
            print('Sequence' + str(self.seq_num) + ' Loading Failed.')
    
    def load_timestamps(self):
        cur_timestamp = self.sequence.first_timestamp
        timestamps = [cur_timestamp]
        while True:
            cur_timestamp = self.sequence.next_timestamp_after(cur_timestamp)
            if cur_timestamp is None:
                break
            timestamps.append(cur_timestamp)

        return timestamps


class dataConverter():

    def __init__(self, sequence):
        super().__init__()
        self.sequence = sequence
        self.prev_image = []

    def open_bagfile(self):
        rospack = rospkg.RosPack()
        bag_name = 'RadarScenes-' + self.sequence.sequence_name + '.bag'
        bag_path = os.path.join(os.path.abspath(rospack.get_path('radarscenes2bag')), 'bagfiles', bag_name)
        print(f'Writing to {bag_path}')
        self.bag = rosbag.Bag(bag_path, 'w', compression='lz4')

    def load_single_scene(self, timestamp):
        """
        Constructs a Scene object for measurements of a given timestamp.
        The scene holds radar data, odometry data as well as the name of the camera image belonging to this scene.
        If the timestamp is invalid, None is returned.
        :param timestamp: The timestamp for which a scene is desired.
        :return: The Scene object or None, if the timestamp is invalid.
        """
        if timestamp is None or self.sequence.radar_data is None or self.sequence.odometry_data is None:
            return None
        timestamp = str(timestamp)
        if timestamp not in self.sequence._scenes:
            return None

        scene_dict = self.sequence._scenes[timestamp]
        self.scene = Scene()
        self.scene.timestamp = int(timestamp)
        self.scene.radar_data = self.sequence.radar_data[scene_dict["radar_indices"][0]: scene_dict["radar_indices"][1]]
        self.scene.odometry_data = self.sequence.odometry_data[scene_dict["odometry_index"]]
        self.scene.odometry_timestamp = scene_dict["odometry_timestamp"]
        self.scene.sensor_id = scene_dict["sensor_id"]
        self.scene.camera_image_name = os.path.join(self.sequence._data_folder, "camera", scene_dict["image_name"])

    def write_img_msg(self):
        tf_array = get_tfmessage(cur_sample)
        self.bag.write('/tf', tf_array, stamp)
    
    def get_transform(self, x, y, yaw):
        transf = Transform()
        quat = tf.transformations.quaternion_from_euler(0.0, 0.0, yaw)
        transf.translation.x = x
        transf.translation.y = y
        transf.translation.z = 0.0
        
        transf.rotation.x = quat[0]
        transf.rotation.y = quat[1]
        transf.rotation.z = quat[2]
        transf.rotation.w = quat[3]

        return transf

    def write_tf(self):
        # Get sensor id
        sensor_id = self.scene.sensor_id

        ### Get Transform ###
        transforms = []

        # Get scene time stamp
        secs, msecs = divmod(self.scene.timestamp, 1_000_000)
        nsecs = msecs * 1000
        stamp = rospy.Time(secs, nsecs)

        # create odom transform
        #print(self.scene.odometry_data)
        x = self.scene.odometry_data['x_seq']
        y = self.scene.odometry_data['y_seq']
        yaw = self.scene.odometry_data['yaw_seq']

        odom_tf = TransformStamped()
        odom_tf.header.frame_id = 'map'
        odom_tf.header.stamp = stamp
        odom_tf.child_frame_id = 'base_link'
        odom_tf.transform = self.get_transform(x, y, yaw)
        transforms.append(odom_tf)

        # create radar transform
        x = RADAR_DEFAULT_MOUNTING[sensor_id]['x']
        y = RADAR_DEFAULT_MOUNTING[sensor_id]['y']
        yaw = RADAR_DEFAULT_MOUNTING[sensor_id]['yaw']

        #print(self.scene.radar_data['x_seq'])

        sensor_tf = TransformStamped()
        sensor_tf.header.frame_id = 'base_link'
        sensor_tf.header.stamp = stamp
        sensor_tf.child_frame_id = 'radar_' + str(sensor_id)
        sensor_tf.transform = self.get_transform(x, y, yaw)
        transforms.append(sensor_tf)

        # get transforms for the current sample
        tf_array = TFMessage()
        tf_array.transforms = transforms

        # add transforms from the next sample to enable interpolation
        self.bag.write('/tf', tf_array, stamp)


    def write_odom(self):
        secs, msecs = divmod(self.scene.odometry_data['timestamp'], 1_000_000)
        nsecs = msecs * 1000
        stamp = rospy.Time(secs, nsecs)

        x = self.scene.odometry_data['x_seq']
        y = self.scene.odometry_data['y_seq']
        yaw = self.scene.odometry_data['yaw_seq']
        tf_quat = tf.transformations.quaternion_from_euler(0.0, 0.0, yaw)

        msg = Odometry()
        msg.header.frame_id = 'map'
        msg.header.stamp = stamp
        msg.child_frame_id = 'odom'
        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        msg.pose.pose.position.z = 0.0
        msg.pose.pose.orientation.w = tf_quat[3]
        msg.pose.pose.orientation.x = tf_quat[0]
        msg.pose.pose.orientation.y = tf_quat[1]
        msg.pose.pose.orientation.z = tf_quat[2]

        #print(msg)
        self.bag.write('/odom', msg, stamp)

    def write_radar_msg(self):
        """
        - Structure of radar_data -
        timestamp       : IntegerType
        sensor_id       : IntegerType
        range_sc        : FloatType
        azimuth_sc      : FloatType
        rcs             : FloatType
        vr              : FloatType
        vr_compensated  : FloatType
        x_cc            : FloatType
        y_cc            : FloatType
        x_seq           : FloatType
        y_seq           : FloatType
        uuid            : StringType
        track_id        : StringType
        label_id        : IntegerType
        """
        secs, msecs = divmod(self.scene.radar_data['timestamp'][0], 1_000_000)
        nsecs = msecs * 1000
        stamp = rospy.Time(secs, nsecs)

        point_num = len(self.scene.radar_data)
        msg = PointCloud2()
        msg.header.frame_id = 'base_link'# 'radar_' + str(self.scene.radar_data['sensor_id'][0])
        msg.header.stamp = stamp
        x = self.scene.radar_data['x_cc']
        y = self.scene.radar_data['y_cc']
        z = [0]*point_num
        rcs = self.scene.radar_data['rcs']
        vr = self.scene.radar_data['vr']
        vr_compensated = self.scene.radar_data['vr_compensated']
        range_sc = self.scene.radar_data['range_sc']
        azimuth_sc = self.scene.radar_data['azimuth_sc']
        pointcld = np.array(np.transpose(np.vstack([x, y, z, rcs, vr, vr_compensated, range_sc, azimuth_sc])), dtype=np.float32)

        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name='vr', offset=16, datatype=PointField.FLOAT32, count=1),
            PointField(name='vr_compensated', offset=20, datatype=PointField.FLOAT32, count=1),
            PointField(name='range_sc', offset=24, datatype=PointField.FLOAT32, count=1),
            PointField(name='azimuth_sc', offset=28, datatype=PointField.FLOAT32, count=1),
            #PointField(name='sensor_id', offset=32, datatype=PointField.INT32, count=1),
        ]

        msg.is_bigendian = False
        msg.is_dense = True
        msg.point_step = len(msg.fields) * 4 # 4 bytes per field
        msg.row_step = msg.point_step * point_num
        msg.width = point_num
        msg.height = 1 # unordered
        msg.data = pointcld.tobytes()
        
        topic_id = 'radar_' + str(self.scene.radar_data['sensor_id'][0]) + '/pointcloud'
        self.bag.write(topic_id, msg, stamp)


    def write_image(self):
        jpg_filename = os.path.join(path_to_dataset, "data", self.sequence.sequence_name, "camera", self.scene.camera_image_name)

        if not self.prev_image == self.scene.camera_image_name:
            if not os.path.exists(jpg_filename):
                print("Please modify this example so that it contains the correct path to the dataset on your machine.")
                return
            #print(jpg_filename)
            secs, msecs = divmod(self.scene.timestamp, 1_000_000)
            nsecs = msecs * 1000
            stamp = rospy.Time(secs, nsecs)

            msg = CompressedImage()
            msg.header.frame_id = 'base_link'
            msg.header.stamp = stamp
            msg.format = "jpeg"
            with open(jpg_filename, 'rb') as jpg_file:
                msg.data = jpg_file.read()
            
            bridge = CvBridge()
            img = bridge.compressed_imgmsg_to_cv2(msg)
            img_msg = bridge.cv2_to_imgmsg(img, encoding="passthrough")

            self.bag.write('/image_rect_compressed', img_msg, stamp)
            self.prev_image = self.scene.camera_image_name

    def write_bagfile(self):
        self.write_tf()
        self.write_odom()
        self.write_radar_msg()

        if image_conversion:
            self.write_image()


def main():
    global path_to_dataset, image_conversion

    rospy.init_node("makeBagfile")
    dataset_path = rospy.get_param("~dataset_path")
    image_conversion = rospy.get_param("~image_conversion")
    seq_from = rospy.get_param("~seq_from")
    seq_to = rospy.get_param("~seq_to")

    path_to_dataset = os.path.realpath(dataset_path)

    if seq_from > 0 and seq_to < 159 and seq_from <= seq_to:
        for iter in range(seq_from, seq_to + 1):
            seq_num = iter
            dL = dataLoader(seq_num)
            dL.open_sequence()
            dL.load_sequence()
            timestamps = dL.load_timestamps()

            dC = dataConverter(dL.sequence)
            dC.open_bagfile()
            for ii in timestamps:
                #print(dL.scene.odometry_data['y_seq'])
                #print(dL.scene.__dict__)
                dC.load_single_scene(ii)
                dC.write_bagfile()

            dC.bag.close()

if __name__ == '__main__':
    main()