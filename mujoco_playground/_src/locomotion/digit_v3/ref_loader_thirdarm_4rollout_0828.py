import os
import jax
import jax.numpy as jp
import numpy as np
import time
from jax import ShapeDtypeStruct
from functools import partial
from jax import jit
from tqdm import tqdm


class JaxReferenceLoader:
    def __init__(self, ref_traj_dir, subsample_factor=1, device="cpu"):
        """
        JAX-based reference trajectory loader that loads one random trajectory per reset.
        """
        self.subsample_factor = subsample_factor
        self.ref_data = None
        self.device = device
        self.ref_traj_dir = ref_traj_dir

        self.a_pos_index = jp.array([
            7,  8,  9,  14, 18, 23, 30, 31, 32, 33,  # actuator joint pos "hip-roll, yaw, pitch, knee, toe-A, B, shoulder-row, pitch, yaw, elbow"
            34, 35, 36, 41, 45, 50, 57, 58, 59, 60,
            61, 62, 63, 64 # thirdarm
        ]) 
        self.a_vel_index = jp.array([
            6,  7,  8,  12, 16, 20, 26, 27, 28, 29,     # actuator joint vel
            30, 31, 32, 36, 40, 44, 50, 51, 52, 53,
            54, 55, 56, 57 # thirdarm
        ]) 

        self.preloaded_refs = self._preload_trajectories()


    def _preload_trajectories(self):
        """
        Preload all reference trajectories into memory.
        """
        preloaded_refs = {
            "ref_motion_lens": [],
            "ref_motor_joint_pos": [],
            "ref_motor_joint_vel": [],
            "ref_base_local_pos": [],
            "ref_base_local_trans": [],
            "ref_base_local_ori": [],
            "ref_base_local_ee_pos": [],
            "ref_base_robot_lin_vel": [],
            "ref_base_robot_ang_vel": [],
            # Uncomment if needed:
            # "ref_base_robot_ee_pos": [],
            "ref_local_ee_pos": [],
            # "ref_torque": [],
            # "task_label":[],
            "obj_pos_init": [],
            # "obj_pos_goal": [],
            "obj_width": [],
        }

        if os.path.isdir(self.ref_traj_dir):
            npz_files = sorted([
                os.path.join(self.ref_traj_dir, f)
                for f in os.listdir(self.ref_traj_dir)
                if f.endswith(".npz")
            ])
        elif os.path.isfile(self.ref_traj_dir) and self.ref_traj_dir.endswith(".npz"):
            npz_files = [self.ref_traj_dir]
        else:
            raise ValueError(f"Invalid ref_traj_dir: {self.ref_traj_dir}")

        sampling_step = 1  # 200Hz → 50Hz
        for path in tqdm(npz_files, desc="Loading reference trajectories"):
            data = np.load(path)
            ref_len = ref_len = int(data["true_length"]) // sampling_step # int(data["true_length"])  # get original unpadded length
            ref_qpos = jp.array(data["qpos"][::sampling_step])
            ref_qvel = jp.array(data["qvel"][::sampling_step])
            ref_ee_pos = jp.array(data["ee_pos"].reshape(-1, 5, 3)[::sampling_step])
            # ref_torque = jp.array(data["torque"][::sampling_step])
            ref_base_local_quat = jp.array(ref_qpos[:, [6, 3, 4, 5]])  # (w, x, y, z) Notice: Specify the index for your reference trajectory
            # task_label = jp.array(ref_data[:, -1])
            obj_pos_init = jp.array(data["obj_pos_init"][::sampling_step])
            obj_width = jp.array(data["obj_width"][::sampling_step])


            """
            Here "local" means the world frame in Crocoddyl; it is equal to the initial yaw frame in mujoco (Because we random yaw angle during training),
            if the initial yaw = 0 in mujoco, the "local" would be the same with the "world"
            """
            ref_base_local_ori = self.quat2euler(ref_base_local_quat) #
            ref_base_local_pos = jp.array(ref_qpos[:, :3])
            ref_base_local_trans = ref_base_local_pos - ref_base_local_pos[0]
            ref_base_local_ee_pos = ref_ee_pos - ref_base_local_pos[:, None, :]

            ref_base_rot = jax.vmap(self.euler2mat)(ref_base_local_ori)
            ref_base_rot = ref_base_rot.squeeze()
            # ref_base_robot_ee_pos = jp.einsum('bij,bjk->bik', ref_base_rot.transpose(0, 2, 1), ref_base_local_ee_pos.transpose(0, 2, 1)).transpose(0, 2, 1)
            ref_base_robot_lin_vel = jp.einsum('bij,bj->bi', ref_base_rot.transpose(0, 2, 1), ref_qvel[:, :3])
            ref_base_robot_ang_vel = jp.einsum('bij,bj->bi', ref_base_rot.transpose(0, 2, 1), ref_qvel[:, 3:6])

            # for i in range(ref_len):
            #     print('ref_base_local_ori', ref_base_local_ori[i])
            #     print('ref_ee_pos', ref_ee_pos[i])

            preloaded_refs["ref_motion_lens"].append(ref_len)
            preloaded_refs["ref_motor_joint_pos"].append(ref_qpos[:,self.a_pos_index])
            preloaded_refs["ref_motor_joint_vel"].append(ref_qvel[:,self.a_vel_index])
            preloaded_refs["ref_base_local_pos"].append(ref_base_local_pos)
            preloaded_refs["ref_base_local_trans"].append(ref_base_local_trans)
            preloaded_refs["ref_base_local_ori"].append(ref_base_local_ori)
            preloaded_refs["ref_base_local_ee_pos"].append(ref_base_local_ee_pos)   # end effector positions (relative to the base) in the local (initial yaw frame)
            preloaded_refs["ref_base_robot_lin_vel"].append(ref_base_robot_lin_vel) # base linear velocity in the robot frame
            preloaded_refs["ref_base_robot_ang_vel"].append(ref_base_robot_ang_vel) # base angular velocity in the robot frame
            
            # Uncomment if needed:
            # preloaded_refs["ref_base_robot_ee_pos"].append(ref_base_robot_ee_pos)
            preloaded_refs["ref_local_ee_pos"].append(ref_ee_pos)
            # preloaded_refs["ref_torque"].append(ref_torque)
            # preloaded_refs["task_label"].append(task_label)
            preloaded_refs["obj_pos_init"].append(obj_pos_init)
            preloaded_refs["obj_width"].append(obj_width)

        for key in preloaded_refs:
            preloaded_refs[key] = jp.array(preloaded_refs[key])


        return preloaded_refs
        
    
    def quat2euler(self, quat):
        """
        Convert quaternion (w, x, y, z) to Euler angles (roll, pitch, yaw).
        """
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        roll = jp.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        pitch = jp.arcsin(jp.clip(2*(w*y - z*x), -1, 1))
        yaw = jp.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        return jp.stack([roll, pitch, yaw], axis=-1)

    
    def euler2mat(self, euler):
        """
        Convert Euler angles (roll, pitch, yaw) to rotation matrices.
        Supports both single `(3,)` and batched `(T, 3)` inputs.
        """
        roll, pitch, yaw = euler[..., 0], euler[..., 1], euler[..., 2]

        cx, sx = jp.cos(roll), jp.sin(roll)
        cy, sy = jp.cos(pitch), jp.sin(pitch)
        cz, sz = jp.cos(yaw), jp.sin(yaw)

        rot_x = jp.stack([
            jp.ones_like(cx), jp.zeros_like(cx), jp.zeros_like(cx),
            jp.zeros_like(cx), cx, -sx,
            jp.zeros_like(cx), sx, cx
        ], axis=-1).reshape(-1, 3, 3)

        rot_y = jp.stack([
            cy, jp.zeros_like(cy), sy,
            jp.zeros_like(cy), jp.ones_like(cy), jp.zeros_like(cy),
            -sy, jp.zeros_like(cy), cy
        ], axis=-1).reshape(-1, 3, 3)

        rot_z = jp.stack([
            cz, -sz, jp.zeros_like(cz),
            sz, cz, jp.zeros_like(cz),
            jp.zeros_like(cz), jp.zeros_like(cz), jp.ones_like(cz)
        ], axis=-1).reshape(-1, 3, 3)

        return jp.einsum('bij,bjk,bkl->bil', rot_z, rot_y, rot_x)
    




# ***********qpos: 61********************

        # [0-6] base, x y z qw qx qy qz
        # [7] left-hip-roll
        # [8] left-hip-yaw
        # [9] left-hip-pitch
        # [10-13] left-achillies-rod 
        # [14] left-knee
        # [15] left-shin            
        # [16] left-tarsus                     
        # [17] left-heel-spring      
        # [18] left-toe-A                      
        # [19-22] left-toe-A-rod
        # [23] left-toe-B
        # [24-27] left-toe-B-rod
        # [28] left-toe-pitch
        # [29] left-toe-roll
        # [30] left-shoulder-roll
        # [31] left-shoulder-pitch
        # [32] left-shoulder-yaw
        # [33] left-elbow
        # [34] right-hip-roll
        # [35] right-hip-yaw
        # [36] right-hip-pitch
        # [37-40] right-achillies-rod
        # [41] right-knee
        # [42] right-shin
        # [43] right-tarsus
        # [44] right-heel-spring
        # [45] right-toe-A
        # [46-49] right-toe-A-rod
        # [50] right-toe-B
        # [51-54] right-toe-B-rod
        # [55] right-toe-pitch
        # [56] right-toe-roll
        # [57] right-shoulder-roll
        # [58] right-shoulder-pitch
        # [59] right-shoulder-yaw
        # [60] right-elbow



        # qvel 54: env.model.jnt_dofadr, env.init_qvel.size

        # [0-5] base
        # [6] left-hip-roll
        # [7] left-hip-yaw
        # [8] left-hip-pitch            
        # [9-11] left-achillies-rod     
        # [12] left-knee
        # [13] left-shin
        # [14] left-tarsus
        # [15] left-heel-spring
        # [16] left-toe-A
        # [17-19] left-toe-A-rod
        # [20] left-toe-B
        # [21-23] left-toe-B-rod
        # [24] left-toe-pitch
        # [25] left-toe-roll
        # [26] left-shoulder-roll
        # [27] left-shoulder-pitch
        # [28] left-shoulder-yaw
        # [29] left-elbow
        # [30] right-hip-roll
        # [31] right-hip-yaw
        # [32] right-hip-pitch
        # [33-35] right-achillies-rod
        # [36] right-knee
        # [37] right-shin
        # [38] right-tarsus
        # [39] right-heel-spring
        # [40] right-toe-A
        # [41-43] right-toe-A-rod
        # [44] right-toe-B
        # [45-47] right-toe-B-rod
        # [48] right-toe-pitch
        # [49] right-toe-roll
        # [50] right-shoulder-roll
        # [51] right-shoulder-pitch
        # [52] right-shoulder-yaw
        # [53] right-elbow

        # motor/actuators: 20
        # joint_names = ["left-hip-roll", 
        #                 "left-hip-yaw", 
        #                 "left-hip-pitch", 
        #                 "left-knee",
        #                 "left-toe-A", 
        #                 "left-toe-B",  
        #                 "left-shoulder-roll",
        #                 "left-shoulder-pitch", 
        #                 "left-shoulder-yaw", 
        #                 "left-elbow", 
        #                 "right-hip-roll", 
        #                 "right-hip-yaw", 
        #                 "right-hip-pitch", 
        #                 "right-knee", 
        #                 "right-toe-A",
        #                 "right-toe-B", 
        #                 "right-shoulder-roll", 
        #                 "right-shoulder-pitch",
        #                 "right-shoulder-yaw", 
        #                 "right-elbow"
        #                 ]