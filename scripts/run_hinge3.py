"""
Quick script to test out some basic wing hinge modeling in Mujoco

I want to get something minimal working to confirm:
    1) Blender -> Mujoco pipeline
    2) ability to get motion I want without model dictated to me.

To do this, looking at PWP, Ax1, Ax2, and SLA. Want to actuate
SLA and see if this can produce reasonable movements

"""
# ----------------------------------------
# IMPORTS

import os
import mujoco
import math
import signal
import mujoco_viewer

import numpy as np
import matplotlib.pyplot as plt
import pathlib

# ---------------------------------------
# CONSTANTS
XML_FN = 'hinge3'
CURR_PATH = pathlib.Path(__file__).parent
XML_PATH = CURR_PATH.parent / 'models'
print(XML_PATH)

RENDER_MODE = 'window'  # 'offscreen'

CAM_DISTANCE = 50
CAM_AZIMUTH = 20
CAM_ELEVATION = -45
CAM_LOOKAT = [0.0, 0.0, 0.0]

PERIOD = 0.5

TRANS_MOTION_AMPLITUDE = 0.3
TRANS_MOTION_PHASE = np.deg2rad(180.0)
TRANS_MOTION_OFFSET = TRANS_MOTION_AMPLITUDE
TRANS_JOINT_NAME = "sla_world_slide"  # "sla_world_hinge"

TRANS_POS_ACTUATOR_NAME = 'SLA_trans_position'
TRANS_VEL_ACTUATOR_NAME = 'SLA_trans_velocity'

ROT_MOTION_AMPLITUDE = np.deg2rad(7)
ROT_MOTION_PHASE = np.deg2rad(180.0)
ROT_MOTION_OFFSET = ROT_MOTION_AMPLITUDE
ROT_JOINT_NAME = "sla_world_hinge"  # "sla_world_hinge"

ROT_POS_ACTUATOR_NAME = 'SLA_rot_position'
ROT_VEL_ACTUATOR_NAME = 'SLA_rot_velocity'

done = False


# ---------------------------------------
# FUNCTIONS
def sigint_handler(signum, frame):
    """
    SIGINT handler. Sets done to True to quit simulation.
    """
    global done
    done = True


def soft_start(t, ts):
    """
    Soft startup function for actuators. Ramps from 0.0 to 1.0 during interval from t=0
    to t=ts.
    """
    rval = 0.0
    if t < ts:
        rval = t / ts
    else:
        rval = 1.0
    return rval


def sin_motion(t, amplitude, phase, offset, period):
    start_value = soft_start(data.time, period)
    pos = amplitude * math.sin(2.0 * math.pi * t / period + phase) + offset
    vel = (2.0 * math.pi / period) * amplitude * math.cos(2.0 * math.pi * t / period + phase)
    pos *= start_value
    vel *= start_value
    return pos, vel


def cos_motion(t, amplitude, phase, offset, period):
    start_value = soft_start(data.time, period)
    pos = amplitude * math.cos(2.0 * math.pi * t / period + phase) + offset
    vel = -(2.0 * math.pi / period) * amplitude * math.sin(2.0 * math.pi * t / period + phase)
    pos *= start_value
    vel *= start_value
    return pos, vel


def load_mujoco_model(xml_fn, xml_path=XML_PATH):
    """
    Convenience function to grab mujoco model and
    data from an xml file

    """
    # NB: folder and xml filename should be the same
    xml_path_full = os.path.join(xml_path, xml_fn, '%s.xml' % (xml_fn))

    # generate model and data
    mod = mujoco.MjModel.from_xml_path(xml_path_full)
    dat = mujoco.MjData(mod)

    # return
    return mod, dat


# ---------------------------------------
# MAIN SCRIPT
if __name__ == "__main__":
    # load mujoco model
    model, data = load_mujoco_model(XML_FN)

    # create viewer
    viewer = mujoco_viewer.MujocoViewer(model, data)
    viewer.scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
    viewer.render_mode = RENDER_MODE

    print(viewer)
    signal.signal(signal.SIGINT, sigint_handler)

    viewer.cam.distance = CAM_DISTANCE
    viewer.cam.azimuth = CAM_AZIMUTH
    viewer.cam.elevation = CAM_ELEVATION
    viewer.cam.lookat = CAM_LOOKAT

    # initialize lists to store data on how well control is working
    setpos_list = list()
    pos_list = list()
    err_list = list()
    t_list = list()

    # reset data
    mujoco.mj_resetData(model, data)

    # begin running model
    while not done:
        # increment one time step forward (I think this is 2 ms?)
        mujoco.mj_step(model, data)

        # update actuator values
        setpos_trans, setvel_trans = sin_motion(
            data.time,
            TRANS_MOTION_AMPLITUDE,
            TRANS_MOTION_PHASE,
            TRANS_MOTION_OFFSET,
            PERIOD
        )

        setpos_rot, setvel_rot = sin_motion(
            data.time,
            ROT_MOTION_AMPLITUDE,
            ROT_MOTION_PHASE,
            ROT_MOTION_OFFSET,
            PERIOD
        )

        try:
            data.actuator(TRANS_POS_ACTUATOR_NAME).ctrl = setpos_trans
            data.actuator(TRANS_VEL_ACTUATOR_NAME).ctrl = setvel_trans

            data.actuator(ROT_POS_ACTUATOR_NAME).ctrl = setpos_rot
            data.actuator(ROT_VEL_ACTUATOR_NAME).ctrl = setvel_rot
        except KeyError:
            pass

        # render frame
        try:
            if viewer.render_mode == 'window':
                frame = None
                viewer.render()
            else:
                frame = viewer.read_pixels()
        except:
            done = True

        # tell us where how well we're controlling things
        pos = data.joint(ROT_JOINT_NAME).qpos[0]
        err = setpos_rot - pos
        print(f'{np.rad2deg(setpos_rot):1.4f}, {np.rad2deg(pos):1.4f}, {np.rad2deg(err):1.4f}')

        # add error calc to lists
        setpos_list.append(np.rad2deg(setpos_rot))
        pos_list.append(np.rad2deg(pos))
        err_list.append(np.rad2deg(err))
        t_list.append(data.time)

    # close viewer
    viewer.close()

    # generate plot
    fig, (ax1, ax2) = plt.subplots(2, 1)

    setpos_array = np.asarray(setpos_list)
    pos_array = np.asarray(pos_list)
    err_array = np.asarray(err_list)
    t = np.asarray(t_list)

    ax1.plot(t, setpos_array, label='set pos')
    ax1.plot(t, pos_array, label='meas pos')
    ax1.set_ylabel('deg')
    ax1.legend()

    ax2.plot(t, err_array, label='error')
    ax2.set_xlabel('time')
    ax2.set_ylabel('deg')
    ax2.legend()

    plt.show()
