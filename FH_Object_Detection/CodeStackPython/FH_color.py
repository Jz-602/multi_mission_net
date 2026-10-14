


############################################################################################################
################################################### waymo ##################################################




#
# // Semantic classes for the camera segmentation labels.
# message CameraSegmentation {
#   enum Type {
#     // Anything that does not fit the other classes or is too ambiguous to
#     // label.
#     TYPE_UNDEFINED = 0;
#     // The Waymo vehicle.
#     TYPE_EGO_VEHICLE = 1;
#     // Small vehicle such as a sedan, SUV, pickup truck, minivan or golf cart.
#     TYPE_CAR = 2;
#     // Large vehicle that carries cargo.
#     TYPE_TRUCK = 3;
#     // Large vehicle that carries more than 8 passengers.
#     TYPE_BUS = 4;
#     // Large vehicle that is not a truck or a bus.
#     TYPE_OTHER_LARGE_VEHICLE = 5;
#     // Bicycle with no rider.
#     TYPE_BICYCLE = 6;
#     // Motorcycle with no rider.
#     TYPE_MOTORCYCLE = 7;
#     // Trailer attached to another vehicle or horse.
#     TYPE_TRAILER = 8;
#     // Pedestrian. Does not include objects associated with the pedestrian, such
#     // as suitcases, strollers or cars.
#     TYPE_PEDESTRIAN = 9;
#     // Bicycle with rider.
#     TYPE_CYCLIST = 10;
#     // Motorcycle with rider.
#     TYPE_MOTORCYCLIST = 11;
#     // Birds, including ones on the ground.
#     TYPE_BIRD = 12;
#     // Animal on the ground such as a dog, cat, cow, etc.
#     TYPE_GROUND_ANIMAL = 13;
#     // Cone or short pole related to construction.
#     TYPE_CONSTRUCTION_CONE_POLE = 14;
#     // Permanent horizontal and vertical lamp pole, traffic sign pole, etc.
#     TYPE_POLE = 15;
#     // Large object carried/pushed/dragged by a pedestrian.
#     TYPE_PEDESTRIAN_OBJECT = 16;
#     // Sign related to traffic, including front and back facing signs.
#     TYPE_SIGN = 17;
#     // The box that contains traffic lights regardless of front or back facing.
#     TYPE_TRAFFIC_LIGHT = 18;
#     // Permanent building and walls, including solid fences.
#     TYPE_BUILDING = 19;
#     // Drivable road with proper markings, including parking lots and gas
#     // stations.
#     TYPE_ROAD = 20;
#     // Marking on the road that is parallel to the ego vehicle and defines
#     // lanes.
#     TYPE_LANE_MARKER = 21;
#     // All markings on the road other than lane markers.
#     TYPE_ROAD_MARKER = 22;
#     // Paved walkable surface for pedestrians, including curbs.
#     TYPE_SIDEWALK = 23;
#     // Vegetation including tree trunks, tree branches, bushes, tall grasses,
#     // flowers and so on.
#     TYPE_VEGETATION = 24;
#     // The sky, including clouds.
#     TYPE_SKY = 25;
#     // Other horizontal surfaces that are drivable or walkable.
#     TYPE_GROUND = 26;
#     // Object that is not permanent in its current position and does not belong
#     // to any of the above classes.
#     TYPE_DYNAMIC = 27;
#     // Object that is permanent in its current position and does not belong to
#     // any of the above classes.
#     TYPE_STATIC = 28;
#   }
# }
#
# # RGB colors used to visualize each semantic segmentation class.
# SEGMENTATION_COLOR_MAP = immutabledict.immutabledict({
#     cs_pb2.CameraSegmentation.TYPE_UNDEFINED: [0, 0, 0],
#     cs_pb2.CameraSegmentation.TYPE_EGO_VEHICLE: [102, 102, 102],
#     cs_pb2.CameraSegmentation.TYPE_CAR: [0, 0, 142],
#     cs_pb2.CameraSegmentation.TYPE_TRUCK: [0, 0, 70],
#     cs_pb2.CameraSegmentation.TYPE_BUS: [0, 60, 100],
#     cs_pb2.CameraSegmentation.TYPE_OTHER_LARGE_VEHICLE: [61, 133, 198],
#     cs_pb2.CameraSegmentation.TYPE_BICYCLE: [119, 11, 32],
#     cs_pb2.CameraSegmentation.TYPE_MOTORCYCLE: [0, 0, 230],
#     cs_pb2.CameraSegmentation.TYPE_TRAILER: [111, 168, 220],
#     cs_pb2.CameraSegmentation.TYPE_PEDESTRIAN: [220, 20, 60],
#     cs_pb2.CameraSegmentation.TYPE_CYCLIST: [255, 0, 0],
#     cs_pb2.CameraSegmentation.TYPE_MOTORCYCLIST: [180, 0, 0],
#     cs_pb2.CameraSegmentation.TYPE_BIRD: [127, 96, 0],
#     cs_pb2.CameraSegmentation.TYPE_GROUND_ANIMAL: [91, 15, 0],
#     cs_pb2.CameraSegmentation.TYPE_CONSTRUCTION_CONE_POLE: [230, 145, 56],
#     cs_pb2.CameraSegmentation.TYPE_POLE: [153, 153, 153],
#     cs_pb2.CameraSegmentation.TYPE_PEDESTRIAN_OBJECT: [234, 153, 153],
#     cs_pb2.CameraSegmentation.TYPE_SIGN: [246, 178, 107],
#     cs_pb2.CameraSegmentation.TYPE_TRAFFIC_LIGHT: [250, 170, 30],
#     cs_pb2.CameraSegmentation.TYPE_BUILDING: [70, 70, 70],
#     cs_pb2.CameraSegmentation.TYPE_ROAD: [128, 64, 128],
#     cs_pb2.CameraSegmentation.TYPE_LANE_MARKER: [234, 209, 220],
#     cs_pb2.CameraSegmentation.TYPE_ROAD_MARKER: [217, 210, 233],
#     cs_pb2.CameraSegmentation.TYPE_SIDEWALK: [244, 35, 232],
#     cs_pb2.CameraSegmentation.TYPE_VEGETATION: [107, 142, 35],
#     cs_pb2.CameraSegmentation.TYPE_SKY: [70, 130, 180],
#     cs_pb2.CameraSegmentation.TYPE_GROUND: [102, 102, 102],
#     cs_pb2.CameraSegmentation.TYPE_DYNAMIC: [102, 102, 102],
#     cs_pb2.CameraSegmentation.TYPE_STATIC: [102, 102, 102],
# })


WAYMO_CAMERA_SEGMENTATION_TYPE_TO_INT_DICT = {'TYPE_UNDEFINED': 0, 'TYPE_EGO_VEHICLE': 1, 'TYPE_CAR': 2, 'TYPE_TRUCK': 3, 'TYPE_BUS': 4,
                                              'TYPE_OTHER_LARGE_VEHICLE': 5, 'TYPE_BICYCLE': 6, 'TYPE_MOTORCYCLE': 7, 'TYPE_TRAILER': 8,
                                              'TYPE_PEDESTRIAN': 9, 'TYPE_CYCLIST': 10, 'TYPE_MOTORCYCLIST': 11, 'TYPE_BIRD': 12,
                                              'TYPE_GROUND_ANIMAL': 13, 'TYPE_CONSTRUCTION_CONE_POLE': 14, 'TYPE_POLE': 15, 'TYPE_PEDESTRIAN_OBJECT': 16,
                                              'TYPE_SIGN': 17, 'TYPE_TRAFFIC_LIGHT': 18, 'TYPE_BUILDING': 19, 'TYPE_ROAD': 20, 'TYPE_LANE_MARKER': 21,
                                              'TYPE_ROAD_MARKER': 22, 'TYPE_SIDEWALK': 23, 'TYPE_VEGETATION': 24, 'TYPE_SKY': 25,
                                              'TYPE_GROUND': 26, 'TYPE_DYNAMIC': 27, 'TYPE_STATIC': 28}

# WAYMO_CAMERA_SEGMENTATION_INT_TO_TYPE_DICT = { value: key for index, (key, value) in enumerate(WAYMO_CAMERA_SEGMENTATION_TYPE_TO_INT_DICT.items())}
# print(WAYMO_CAMERA_SEGMENTATION_INT_TO_TYPE_DICT)
WAYMO_CAMERA_SEGMENTATION_INT_TO_TYPE_DICT = {0: 'TYPE_UNDEFINED', 1: 'TYPE_EGO_VEHICLE', 2: 'TYPE_CAR', 3: 'TYPE_TRUCK', 4: 'TYPE_BUS',
                                              5: 'TYPE_OTHER_LARGE_VEHICLE', 6: 'TYPE_BICYCLE', 7: 'TYPE_MOTORCYCLE', 8: 'TYPE_TRAILER',
                                              9: 'TYPE_PEDESTRIAN', 10: 'TYPE_CYCLIST', 11: 'TYPE_MOTORCYCLIST', 12: 'TYPE_BIRD',
                                              13: 'TYPE_GROUND_ANIMAL', 14: 'TYPE_CONSTRUCTION_CONE_POLE', 15: 'TYPE_POLE', 16: 'TYPE_PEDESTRIAN_OBJECT',
                                              17: 'TYPE_SIGN', 18: 'TYPE_TRAFFIC_LIGHT', 19: 'TYPE_BUILDING', 20: 'TYPE_ROAD', 21: 'TYPE_LANE_MARKER',
                                              22: 'TYPE_ROAD_MARKER', 23: 'TYPE_SIDEWALK', 24: 'TYPE_VEGETATION', 25: 'TYPE_SKY',
                                              26: 'TYPE_GROUND', 27: 'TYPE_DYNAMIC', 28: 'TYPE_STATIC'}

WAYMO_CAMERA_SEGMENTATION_TYPE_TO_COLOR_DICT = {'TYPE_UNDEFINED': [0, 0, 0], 'TYPE_EGO_VEHICLE': [102, 102, 102], 'TYPE_CAR': [0, 0, 142], 'TYPE_TRUCK': [0, 0, 70],
                                                'TYPE_BUS': [0, 60, 100], 'TYPE_OTHER_LARGE_VEHICLE': [61, 133, 198], 'TYPE_BICYCLE': [119, 11, 32],
                                                'TYPE_MOTORCYCLE': [0, 0, 230], 'TYPE_TRAILER': [111, 168, 220], 'TYPE_PEDESTRIAN': [220, 20, 60],
                                                'TYPE_CYCLIST': [255, 0, 0], 'TYPE_MOTORCYCLIST': [180, 0, 0], 'TYPE_BIRD': [127, 96, 0], 'TYPE_GROUND_ANIMAL': [91, 15, 0],
                                                'TYPE_CONSTRUCTION_CONE_POLE': [230, 145, 56], 'TYPE_POLE': [153, 153, 153], 'TYPE_PEDESTRIAN_OBJECT': [234, 153, 153],
                                                'TYPE_SIGN': [246, 178, 107], 'TYPE_TRAFFIC_LIGHT': [250, 170, 30], 'TYPE_BUILDING': [70, 70, 70],
                                                'TYPE_ROAD': [128, 64, 128], 'TYPE_LANE_MARKER': [234, 209, 220], 'TYPE_ROAD_MARKER': [217, 210, 233],
                                                'TYPE_SIDEWALK': [244, 35, 232], 'TYPE_VEGETATION': [107, 142, 35], 'TYPE_SKY': [70, 130, 180],
                                                'TYPE_GROUND': [102, 102, 102], 'TYPE_DYNAMIC': [102, 102, 102], 'TYPE_STATIC': [102, 102, 102]}


# message Segmentation {
#   enum Type {
#     TYPE_UNDEFINED = 0;
#     TYPE_CAR = 1;
#     TYPE_TRUCK = 2;
#     TYPE_BUS = 3;
#     // Other small vehicles (e.g. pedicab) and large vehicles (e.g. construction
#     // vehicles, RV, limo, tram).
#     TYPE_OTHER_VEHICLE = 4;
#     TYPE_MOTORCYCLIST = 5;
#     TYPE_BICYCLIST = 6;
#     TYPE_PEDESTRIAN = 7;
#     TYPE_SIGN = 8;
#     TYPE_TRAFFIC_LIGHT = 9;
#     // Lamp post, traffic sign pole etc.
#     TYPE_POLE = 10;
#     // Construction cone/pole.
#     TYPE_CONSTRUCTION_CONE = 11;
#     TYPE_BICYCLE = 12;
#     TYPE_MOTORCYCLE = 13;
#     TYPE_BUILDING = 14;
#     // Bushes, tree branches, tall grasses, flowers etc.
#     TYPE_VEGETATION = 15;
#     TYPE_TREE_TRUNK = 16;
#     // Curb on the edge of roads. This does not include road boundaries if
#     // there’s no curb.
#     TYPE_CURB = 17;
#     // Surface a vehicle could drive on. This include the driveway connecting
#     // parking lot and road over a section of sidewalk.
#     TYPE_ROAD = 18;
#     // Marking on the road that’s specifically for defining lanes such as
#     // single/double white/yellow lines.
#     TYPE_LANE_MARKER = 19;
#     // Marking on the road other than lane markers, bumps, cateyes, railtracks
#     // etc.
#     TYPE_OTHER_GROUND = 20;
#     // Most horizontal surface that’s not drivable, e.g. grassy hill,
#     // pedestrian walkway stairs etc.
#     TYPE_WALKABLE = 21;
#     // Nicely paved walkable surface when pedestrians most likely to walk on.
#     TYPE_SIDEWALK = 22;
#   }
# }
WAYMO_SEGMENTATION_TYPE_TO_INT_DICT = {'TYPE_UNDEFINED': 0, 'TYPE_CAR': 1, 'TYPE_TRUCK': 2, 'TYPE_BUS': 3, 'TYPE_OTHER_VEHICLE': 4,
                                         'TYPE_MOTORCYCLIST': 5, 'TYPE_BICYCLIST': 6, 'TYPE_PEDESTRIAN': 7, 'TYPE_SIGN':8, 'TYPE_TRAFFIC_LIGHT': 9,
                                         'TYPE_POLE': 10, 'TYPE_CONSTRUCTION_CONE': 11, 'TYPE_BICYCLE': 12, 'TYPE_MOTORCYCLE': 13, 'TYPE_BUILDING': 14,
                                         'TYPE_VEGETATION': 15, 'TYPE_TREE_TRUNK': 16, 'TYPE_CURB': 17, 'TYPE_ROAD': 18, 'TYPE_LANE_MARKER': 19,
                                         'TYPE_OTHER_GROUND': 20, 'TYPE_WALKABLE': 21, 'TYPE_SIDEWALK': 22}

# WAYMO_SEGMENTATION_INT_TO_TYPE_DICT = { value: key for index, (key, value) in enumerate(WAYMO_SEGMENTATION_TYPE_TO_INT_DICT.items())}
# print(WAYMO_SEGMENTATION_INT_TO_TYPE_DICT)
WAYMO_SEGMENTATION_INT_TO_TYPE_DICT = {0: 'TYPE_UNDEFINED', 1: 'TYPE_CAR', 2: 'TYPE_TRUCK', 3: 'TYPE_BUS', 4: 'TYPE_OTHER_VEHICLE',
                                         5: 'TYPE_MOTORCYCLIST', 6: 'TYPE_BICYCLIST', 7: 'TYPE_PEDESTRIAN', 8: 'TYPE_SIGN', 9: 'TYPE_TRAFFIC_LIGHT',
                                         10: 'TYPE_POLE', 11: 'TYPE_CONSTRUCTION_CONE', 12: 'TYPE_BICYCLE', 13: 'TYPE_MOTORCYCLE', 14: 'TYPE_BUILDING',
                                         15: 'TYPE_VEGETATION', 16: 'TYPE_TREE_TRUNK', 17: 'TYPE_CURB', 18: 'TYPE_ROAD', 19: 'TYPE_LANE_MARKER',
                                         20: 'TYPE_OTHER_GROUND', 21: 'TYPE_WALKABLE', 22: 'TYPE_SIDEWALK'}

# waymo has 23 semantic classes, 19 can be matched to KITTI classes. The remaining 4 are matched as follows:
#  waymo: TYPE_WALKABLE -> KITTI: terrain
#  waymo: TYPE_TRAFFIC_LIGHT -> KITTI: other-object
#  waymo: TYPE_CONSTRUCTION_CONE -> KITTI: parking
#  waymo: TYPE_CURB -> KITTI: other-structure

#   From waymo labeling specifications:
#   A vehicle bounding box is created if an object can be recognized as a vehicle, from either lidar data or camera images.
#   Trains and trams are not considered vehicles and are not labeled.
#   Motorcycles and motorcyclists are labeled as vehicles.
WAYMO_TYPE_TO_KITTI_TYPE_DICT = {'TYPE_UNDEFINED': 'unlabeled', 'TYPE_CAR': 'car', 'TYPE_TRUCK': 'truck', 'TYPE_BUS': 'bus',
                                 'TYPE_OTHER_VEHICLE': 'other-vehicle', 'TYPE_MOTORCYCLIST': 'motorcyclist', 'TYPE_BICYCLIST': 'bicyclist',
                                 'TYPE_PEDESTRIAN': 'person', 'TYPE_SIGN':'traffic-sign', 'TYPE_TRAFFIC_LIGHT': 'other-object',
                                 'TYPE_POLE': 'pole', 'TYPE_CONSTRUCTION_CONE': 'parking', 'TYPE_BICYCLE': 'bicycle', 'TYPE_MOTORCYCLE': 'motorcycle',
                                 'TYPE_BUILDING': 'building', 'TYPE_VEGETATION': 'vegetation', 'TYPE_TREE_TRUNK': 'trunk',
                                 'TYPE_CURB': 'other-structure', 'TYPE_ROAD': 'road', 'TYPE_LANE_MARKER': 'lane-marking',
                                 'TYPE_OTHER_GROUND': 'other-ground', 'TYPE_WALKABLE': 'terrain', 'TYPE_SIDEWALK': 'sidewalk'}

WAYMO_TYPE_TO_OUR_TYPE_DICT = {'TYPE_UNDEFINED': 'unlabeled/unknown', 'TYPE_CAR': 'vehicle', 'TYPE_TRUCK': 'vehicle', 'TYPE_BUS': 'vehicle',
                                 'TYPE_OTHER_VEHICLE': 'vehicle', 'TYPE_MOTORCYCLIST': 'cyclist', 'TYPE_BICYCLIST': 'cyclist',
                                 'TYPE_PEDESTRIAN': 'pedestrian', 'TYPE_SIGN':'general_obstacle', 'TYPE_TRAFFIC_LIGHT': 'general_obstacle',
                                 'TYPE_POLE': 'general_obstacle', 'TYPE_CONSTRUCTION_CONE': 'general_obstacle', 'TYPE_BICYCLE': 'cyclist', 'TYPE_MOTORCYCLE': 'cyclist',
                                 'TYPE_BUILDING': 'general_obstacle', 'TYPE_VEGETATION': 'vegetation', 'TYPE_TREE_TRUNK': 'general_obstacle',
                                 'TYPE_CURB': 'road', 'TYPE_ROAD': 'road', 'TYPE_LANE_MARKER': 'road',
                                 'TYPE_OTHER_GROUND': 'road', 'TYPE_WALKABLE': 'road2', 'TYPE_SIDEWALK': 'road2'}

OUR_TYPE_TO_INT_DICT = {'unlabeled/unknown': 0, 'vehicle': 1, 'cyclist': 2, 'pedestrian': 3,
                        'general_obstacle': 4, 'road2': 5, 'vegetation':6, 'road': 7, 'free': 8}

OUR_INT_TO_COLOR_DICT = {0:[128, 128, 128], 1:[245, 150, 100], 2:[90, 30, 150], 3:[30, 30, 255],
                         4:[0, 200, 255], 5:[255, 150, 255], 6:[0, 175, 0], 7:[255, 0, 255], 8:[0, 100, 0]}

# KITTI_TYPE_TO_OUR_TYPE_DICT = {'unlabeled': 'unlabeled/unknown', 'outlier': 'unlabeled/unknown', 'car': 'vehicle', 'bicycle': 'cyclist', 'bus': 'vehicle',
#                                  'motorcycle': 'cyclist', 'on-rails': 'vehicle', 'truck': 'vehicle', 'other-vehicle': 'vehicle',
#                                  'person': 'pedestrian', 'bicyclist': 'cyclist', 'motorcyclist': 'cyclist', 'road': 'road',
#                                  'parking': 'road', 'sidewalk': 'road', 'other-ground': 'road', 'building': 'general_obstacle',
#                                  'fence': 'general_obstacle', 'other-structure': 'general_obstacle', 'lane-marking': 'road', 'vegetation': 'vegetation',
#                                  'trunk': 'pole', 'terrain': 'road', 'pole': 'pole', 'traffic-sign': 'general_obstacle', 'other-object': 'general_obstacle'}

############################################################################################################
################################################### Occ3d ##################################################
OCC3D_INT_TO_TYPE_DICT = {0: 'TYPE_GENERALOBJECT', 1: 'TYPE_VEHICLE', 2: 'TYPE_PEDESTRIAN', 3: 'TYPE_SIGN', 4: 'TYPE_CYCLIST',
                         5: 'TYPE_TRAFFIC_LIGHT', 6: 'TYPE_POLE', 7: 'TYPE_CONSTRUCTION_CONE', 8: 'TYPE_BICYCLE', 9: 'TYPE_MOTORCYCLE',
                         10: 'TYPE_BUILDING', 11: 'TYPE_VEGETATION', 12: 'TYPE_TREE_TRUNK', 13: 'TYPE_ROAD', 14: 'TYPE_WALKABLE',
                         15: 'FREE'}

OCC3D_TYPE_TO_OUR_TYPE_DICT = {'TYPE_GENERALOBJECT': 'general_obstacle', 'TYPE_VEHICLE': 'vehicle', 'TYPE_PEDESTRIAN': 'pedestrian', 'TYPE_SIGN': 'general_obstacle',
                               'TYPE_CYCLIST': 'cyclist', 'TYPE_TRAFFIC_LIGHT': 'general_obstacle', 'TYPE_POLE': 'general_obstacle', 'TYPE_CONSTRUCTION_CONE': 'general_obstacle',
                               'TYPE_BICYCLE': 'cyclist', 'TYPE_MOTORCYCLE': 'cyclist', 'TYPE_BUILDING': 'general_obstacle', 'TYPE_VEGETATION': 'vegetation',
                               'TYPE_TREE_TRUNK': 'general_obstacle', 'TYPE_ROAD': 'road', 'TYPE_WALKABLE': 'road2', 'FREE': 'free'}

############################################################################################################
################################################### KITTI ##################################################
# # KITTI has 26 semantic classes, 20 can be matched to waymo class
# KITTI_TYPE_TO_WAYMO_TYPE_DICT = {'unlabeled': 'TYPE_UNDEFINED', 'outlier': 1, 'car': 'TYPE_CAR', 'bicycle': 'TYPE_BICYCLE', 'bus': 'TYPE_BUS',
#                                  'motorcycle': 'TYPE_MOTORCYCLE', 'on-rails': 16, 'truck': 'TYPE_TRUCK', 'other-vehicle': 'TYPE_OTHER_VEHICLE',
#                                  'person': 'TYPE_PEDESTRIAN', 'bicyclist': 'TYPE_BICYCLIST', 'motorcyclist': 'TYPE_MOTORCYCLIST', 'road': 'TYPE_ROAD',
#                                  'parking': 44, 'sidewalk': 'TYPE_SIDEWALK', 'other-ground': 'TYPE_OTHER_GROUND', 'building': 'TYPE_BUILDING',
#                                  'fence': 51, 'other-structure': 52, 'lane-marking': 'TYPE_LANE_MARKER', 'vegetation': 'TYPE_VEGETATION',
#                                  'trunk': 'TYPE_TREE_TRUNK', 'terrain': 'TYPE_WALKABLE', 'pole': 'TYPE_POLE', 'traffic-sign': 'TYPE_SIGN', 'other-object': 99}


KITTI_SEGMENTATION_TYPE_TO_INT_DICT = {'unlabeled': 0, 'outlier': 1, 'car': 10, 'bicycle': 11, 'bus': 13, 'motorcycle':15, 'on-rails': 16,
                                       'truck': 18, 'other-vehicle': 20, 'person': 30, 'bicyclist': 31, 'motorcyclist': 32, 'road': 40,
                                       'parking': 44, 'sidewalk': 48, 'other-ground': 49, 'building': 50, 'fence': 51, 'other-structure': 52,
                                       'lane-marking': 60, 'vegetation': 70, 'trunk': 71, 'terrain': 72, 'pole': 80, 'traffic-sign': 81,
                                       'other-object': 99, 'moving-car': 252, 'moving-bicyclist': 253, 'moving-person': 254, 'moving-motorcyclist': 255,
                                       'moving-on-rails': 256, 'moving-bus': 257, 'moving-truck': 258, 'moving-other-vehicle': 259}

# KITTI_SEGMENTATION_INT_TO_TYPE_DICT = { value: key for index, (key, value) in enumerate(KITTI_SEGMENTATION_TYPE_TO_INT_DICT.items())}
# print(KITTI_SEGMENTATION_INT_TO_TYPE_DICT)
KITTI_SEGMENTATION_INT_TO_TYPE_DICT = {0: 'unlabeled', 1: 'outlier', 10: 'car', 11: 'bicycle', 13: 'bus', 15: 'motorcycle', 16: 'on-rails',
                                       18: 'truck', 20: 'other-vehicle', 30: 'person', 31: 'bicyclist', 32: 'motorcyclist', 40: 'road',
                                       44: 'parking', 48: 'sidewalk', 49: 'other-ground', 50: 'building', 51: 'fence', 52: 'other-structure',
                                       60: 'lane-marking', 70: 'vegetation', 71: 'trunk', 72: 'terrain', 80: 'pole', 81: 'traffic-sign',
                                       99: 'other-object', 252: 'moving-car', 253: 'moving-bicyclist', 254: 'moving-person', 255: 'moving-motorcyclist',
                                       256: 'moving-on-rails', 257: 'moving-bus', 258: 'moving-truck', 259: 'moving-other-vehicle'}

KITTI_SEGMENTATION_INT_TO_COLOR_DICT = {  0 : [0, 0, 0],  1 : [0, 0, 255],  10: [245, 150, 100],  11: [245, 230, 100],  13: [250, 80, 100],  15: [150, 60, 30],
                                          16: [255, 0, 0],  18: [180, 30, 80],  20: [255, 0, 0],  30: [30, 30, 255],  31: [200, 40, 255],  32: [90, 30, 150],
                                          40: [255, 0, 255],  44: [255, 150, 255],  48: [75, 0, 75],  49: [75, 0, 175],  50: [0, 200, 255],  51: [50, 120, 255],
                                          52: [0, 150, 255],  60: [170, 255, 150],  70: [0, 175, 0],  71: [0, 60, 135],  72: [80, 240, 150],  80: [150, 240, 255],
                                          81: [0, 0, 255],  99: [255, 255, 50],  252: [245, 150, 100],  256: [255, 0, 0],  253: [200, 40, 255],  254: [30, 30, 255],
                                          255: [90, 30, 150],  257: [250, 80, 100],  258: [180, 30, 80],  259: [255, 0, 0]} # bgr


# # This file is covered by the LICENSE file in the root of this project.
# name: "kitti"
# labels:
#   0 : "unlabeled"
#   1 : "outlier"
#   10: "car"
#   11: "bicycle"
#   13: "bus"
#   15: "motorcycle"
#   16: "on-rails"
#   18: "truck"
#   20: "other-vehicle"
#   30: "person"
#   31: "bicyclist"
#   32: "motorcyclist"
#   40: "road"
#   44: "parking"
#   48: "sidewalk"
#   49: "other-ground"
#   50: "building"
#   51: "fence"
#   52: "other-structure"
#   60: "lane-marking"
#   70: "vegetation"
#   71: "trunk"
#   72: "terrain"
#   80: "pole"
#   81: "traffic-sign"
#   99: "other-object"
#   252: "moving-car"
#   253: "moving-bicyclist"
#   254: "moving-person"
#   255: "moving-motorcyclist"
#   256: "moving-on-rails"
#   257: "moving-bus"
#   258: "moving-truck"
#   259: "moving-other-vehicle"
# color_map: # bgr
#   0 : [0, 0, 0]
#   1 : [0, 0, 255]
#   10: [245, 150, 100]
#   11: [245, 230, 100]
#   13: [250, 80, 100]
#   15: [150, 60, 30]
#   16: [255, 0, 0]
#   18: [180, 30, 80]
#   20: [255, 0, 0]
#   30: [30, 30, 255]
#   31: [200, 40, 255]
#   32: [90, 30, 150]
#   40: [255, 0, 255]
#   44: [255, 150, 255]
#   48: [75, 0, 75]
#   49: [75, 0, 175]
#   50: [0, 200, 255]
#   51: [50, 120, 255]
#   52: [0, 150, 255]
#   60: [170, 255, 150]
#   70: [0, 175, 0]
#   71: [0, 60, 135]
#   72: [80, 240, 150]
#   80: [150, 240, 255]
#   81: [0, 0, 255]
#   99: [255, 255, 50]
#   252: [245, 150, 100]
#   256: [255, 0, 0]
#   253: [200, 40, 255]
#   254: [30, 30, 255]
#   255: [90, 30, 150]
#   257: [250, 80, 100]
#   258: [180, 30, 80]
#   259: [255, 0, 0]
# content: # as a ratio with the total number of points
#   0: 0.018889854628292943
#   1: 0.0002937197336781505
#   10: 0.040818519255974316
#   11: 0.00016609538710764618
#   13: 2.7879693665067774e-05
#   15: 0.00039838616015114444
#   16: 0.0
#   18: 0.0020633612104619787
#   20: 0.0016218197275284021
#   30: 0.00017698551338515307
#   31: 1.1065903904919655e-08
#   32: 5.532951952459828e-09
#   40: 0.1987493871255525
#   44: 0.014717169549888214
#   48: 0.14392298360372
#   49: 0.0039048553037472045
#   50: 0.1326861944777486
#   51: 0.0723592229456223
#   52: 0.002395131480328884
#   60: 4.7084144280367186e-05
#   70: 0.26681502148037506
#   71: 0.006035012012626033
#   72: 0.07814222006271769
#   80: 0.002855498193863172
#   81: 0.0006155958086189918
#   99: 0.009923127583046915
#   252: 0.001789309418528068
#   253: 0.00012709999297008662
#   254: 0.00016059776092534436
#   255: 3.745553104802113e-05
#   256: 0.0
#   257: 0.00011351574470342043
#   258: 0.00010157861367183268
#   259: 4.3840131989471124e-05
# # classes that are indistinguishable from single scan or inconsistent in
# # ground truth are mapped to their closest equivalent
# learning_map:
#   0 : 0     # "unlabeled"
#   1 : 0     # "outlier" mapped to "unlabeled" --------------------------mapped
#   10: 1     # "car"
#   11: 2     # "bicycle"
#   13: 5     # "bus" mapped to "other-vehicle" --------------------------mapped
#   15: 3     # "motorcycle"
#   16: 5     # "on-rails" mapped to "other-vehicle" ---------------------mapped
#   18: 4     # "truck"
#   20: 5     # "other-vehicle"
#   30: 6     # "person"
#   31: 7     # "bicyclist"
#   32: 8     # "motorcyclist"
#   40: 9     # "road"
#   44: 10    # "parking"
#   48: 11    # "sidewalk"
#   49: 12    # "other-ground"
#   50: 13    # "building"
#   51: 14    # "fence"
#   52: 0     # "other-structure" mapped to "unlabeled" ------------------mapped
#   60: 9     # "lane-marking" to "road" ---------------------------------mapped
#   70: 15    # "vegetation"
#   71: 16    # "trunk"
#   72: 17    # "terrain"
#   80: 18    # "pole"
#   81: 19    # "traffic-sign"
#   99: 0     # "other-object" to "unlabeled" ----------------------------mapped
#   252: 1    # "moving-car" to "car" ------------------------------------mapped
#   253: 7    # "moving-bicyclist" to "bicyclist" ------------------------mapped
#   254: 6    # "moving-person" to "person" ------------------------------mapped
#   255: 8    # "moving-motorcyclist" to "motorcyclist" ------------------mapped
#   256: 5    # "moving-on-rails" mapped to "other-vehicle" --------------mapped
#   257: 5    # "moving-bus" mapped to "other-vehicle" -------------------mapped
#   258: 4    # "moving-truck" to "truck" --------------------------------mapped
#   259: 5    # "moving-other"-vehicle to "other-vehicle" ----------------mapped
# learning_map_inv: # inverse of previous map
#   0: 0      # "unlabeled", and others ignored
#   1: 10     # "car"
#   2: 11     # "bicycle"
#   3: 15     # "motorcycle"
#   4: 18     # "truck"
#   5: 20     # "other-vehicle"
#   6: 30     # "person"
#   7: 31     # "bicyclist"
#   8: 32     # "motorcyclist"
#   9: 40     # "road"
#   10: 44    # "parking"
#   11: 48    # "sidewalk"
#   12: 49    # "other-ground"
#   13: 50    # "building"
#   14: 51    # "fence"
#   15: 70    # "vegetation"
#   16: 71    # "trunk"
#   17: 72    # "terrain"
#   18: 80    # "pole"
#   19: 81    # "traffic-sign"
