"""Constants for Digit v3 tracking tasks."""

from etils import epath

from mujoco_playground._src import mjx_env


ROOT_PATH = mjx_env.ROOT_PATH / "locomotion" / "digit_v3"
XML_DIR = ROOT_PATH / "xmls"

NECKARM_XML = XML_DIR / "scene_neckarm_1018.xml"
BACKARM_XML = XML_DIR / "scene_backarm_1foot_1018.xml"


def task_to_xml(task_name: str) -> epath.Path:
  return {
      "neckarm": NECKARM_XML,
      "backarm": BACKARM_XML,
  }[task_name]


ROOT_BODY = "base"
KEYFRAME = "home"

TRACKED_BODY_NAMES = (
    "base",
    "left-hand",
    "right-hand",
    "left-foot",
    "right-foot",
    "neckarm-tip",
    "backarm-tip",
)

FEET_BODY_NAMES = (
    "left-foot",
    "right-foot",
)

GRAVITY_SENSOR = "upvector"
GLOBAL_LINVEL_SENSOR = "global_linvel"
GLOBAL_ANGVEL_SENSOR = "global_angvel"
LOCAL_LINVEL_SENSOR = "local_linvel"
ACCELEROMETER_SENSOR = "accelerometer"
GYRO_SENSOR = "gyro"

TERMINATION_NAMES = (
    "base_height",
    "anchor_z_drift",
    "gravity",
    "root_orientation",
    "body_drift",
    "illegal_collision",
    "severe_base_velocity",
    "severe_joint_velocity",
    "timeout",
    "reference_end",
    "nan",
)

BAD_TERMINATION_NAMES = (
    "base_height",
    "anchor_z_drift",
    "gravity",
    "root_orientation",
    "body_drift",
    "illegal_collision",
    "severe_base_velocity",
    "severe_joint_velocity",
    "nan",
)

MOTION_TYPES = {
    "default": 0,
    "locomotion": 1,
    "manipulation": 2,
    "pick": 3,
    "walk": 4,
    "walkpush": 5,
    "screwdriving": 6,
    "wall": 7,
}
