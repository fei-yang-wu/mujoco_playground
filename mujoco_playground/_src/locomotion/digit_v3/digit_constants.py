# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Digit constants."""

from etils import epath
from mujoco_playground._src import mjx_env

ROOT_PATH = mjx_env.ROOT_PATH / "locomotion" / "digit_v3" / "xmls"
THIRDARM_WHOLEBODY_XML = ROOT_PATH / "scene_thirdarm_standing.xml"
THIRDARM_SCREWDRIVING_XML = ROOT_PATH / "scene_thirdarm_screwdriving.xml"
NECKARM_WHOLEBODY_XML = ROOT_PATH / "scene_neckarm_1018.xml"
THIRDARM_WHOLEBODY_ROUGHTERRAIN_XML = ROOT_PATH / "scene_thirdarm_roughterrain.xml"
THIRDARM_SHELF_XML = ROOT_PATH / "scene_thirdarm_shelf.xml"
THIRDARM_DOOR_XML = ROOT_PATH / "scene_thirdarm_door.xml"
THIRDARM_FIXED_XML = ROOT_PATH / "scene_thirdarm_fixed.xml"
THIRDARM_TABLE_AND_BOX_XML = ROOT_PATH / "scene_thirdarm_table_and_box.xml"
THIRDARM_TABLE_XML = ROOT_PATH / "scene_thirdarm_table.xml"
BACKARM_1FOOT_WHOLEBODY_XML = ROOT_PATH / "scene_backarm_1foot_1018.xml"
BACKARM_1FOOT_WHOLEBODY_CAREN_XML = ROOT_PATH / "scene_backarm_1foot_1018_CAREN.xml"
BACKARM_1FOOT_WHOLEBODY_CAREN_WALL_XML = ROOT_PATH / "scene_backarm_1foot_CAREN_Wall.xml"


def task_to_xml(task_name: str) -> epath.Path:
  return {
      "thirdarm_wholebody": THIRDARM_WHOLEBODY_XML,
      "thirdarm_screwdriving": THIRDARM_SCREWDRIVING_XML,
      "neckarm_wholebody": NECKARM_WHOLEBODY_XML,
      "backarm_1foot_wholebody": BACKARM_1FOOT_WHOLEBODY_XML,
      "backarm_1foot_wholebody_caren": BACKARM_1FOOT_WHOLEBODY_CAREN_XML,
      "thirdarm_wholebody_roughterrain": THIRDARM_WHOLEBODY_ROUGHTERRAIN_XML,
      "thirdarm_shelf_manipulation": THIRDARM_SHELF_XML,
      "thirdarm_door": THIRDARM_DOOR_XML,
      "thirdarm_fixed": THIRDARM_FIXED_XML,
      "thirdarm_table_and_box": THIRDARM_TABLE_AND_BOX_XML,
      "thirdarm_table": THIRDARM_TABLE_XML,
      "backarm_caren_wall": BACKARM_1FOOT_WHOLEBODY_CAREN_WALL_XML,

  }[task_name]

FEET_SITES = [
    "left_foot",
    "right_foot",
]

FEET_BODIES = [
    "left-foot",
    "right-foot",
]

ARM_GEOMS = [
    "left-shoulder-pitch-c",
    "left-shoulder-yaw-c1",
    "left-elbow-c1",
    "right-shoulder-pitch-c1",
    "right-shoulder-yaw-c1",
    "right-elbow-c1",
]

LEFT_FEET_GEOMS = [
    "left-foot",
]
RIGHT_FEET_GEOMS = [
    "right-foot",
]

RIGHT_HAND_GEOMS = [
    "right-elbow-s1",
]

LEFT_HAND_GEOMS = [
    "left-elbow-s1",
]

LEFT_LEG_GEOMS = [
    "left-shin-c1",
    "left-tarsus-c1",
    "left-tarsus-b1",
]

RIGHT_LEG_GEOMS = [
    "right-shin-c1",
    "right-tarsus-c1",
    "right-tarsus-b1",
]


ROOT_BODY = "torso_link"

GRAVITY_SENSOR = "upvector"
GLOBAL_LINVEL_SENSOR = "global_linvel"
GLOBAL_ANGVEL_SENSOR = "global_angvel"
LOCAL_LINVEL_SENSOR = "local_linvel"
ACCELEROMETER_SENSOR = "accelerometer"
GYRO_SENSOR = "gyro"
