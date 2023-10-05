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
import pathlib
import PIL.Image

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------
# CONSTANTS
XML_FN = 'hinge2'
CURR_PATH = pathlib.Path(__file__).parent
XML_PATH = CURR_PATH.parent / 'models'
print(XML_PATH)

RENDER_MODE = 'offscreen'  #  'window' | 'offscreen'
SAVE_FLAG = True  # only used in RENDER_MODE = 'offscreen'

CAM_DISTANCE = 5
CAM_AZIMUTH = 110
CAM_ELEVATION = -15  # -45
CAM_LOOKAT = [-2.0, -5.0, 8.0]  # [0.0, 0.0, 0.0]

# WINDOW_HEIGHT =
# WINDOW_WIDTH =
MOTION_AMPLITUDE = np.deg2rad(8)  # 5
MOTION_PHASE = np.deg2rad(180.0)
MOTION_OFFSET = MOTION_AMPLITUDE
PERIOD = 1.0  # 0.5
JOINT_NAME = "sla_world_hinge"

POS_ACTUATOR_NAME = 'SLA_position'
VEL_ACTUATOR_NAME = 'SLA_velocity'

# some parameters that will only get used if we're rendering offscreen and saving
DURATION = 3*PERIOD  # 65  # in the units of model.time
START_RECORDING = PERIOD  # time after which to begin saving images
SAVE_PATH = XML_PATH / XML_FN / 'render_images'
if SAVE_FLAG and not os.path.exists(SAVE_PATH) and (RENDER_MODE == 'offscreen'):
    os.mkdir(SAVE_PATH)

# crop image to save?
CROP_BOX = (390, 0, 1700, 1080)  # None  # (704, 215, 1345, 860)

# Flag for ending while loop of simulation
done = False


# ---------------------------------------
# FUNCTIONS
def sigint_handler(signum, frame):
    """
    SIGINT handler. Sets done to True to quit simulation.
    """
    global done
    done = True


def soft_start(t,ts):
    """
    Soft startup function for actuators. Ramps from 0.0 to 1.0 during interval from t=0
    to t=ts.
    """
    rval = 0.0
    if t < ts:
        rval = t/ts
    else:
        rval = 1.0
    return rval


def sin_motion(t, amplitude, phase, offset, period):
    start_value = soft_start(data.time, period)
    pos = amplitude*math.sin(2.0*math.pi*t/period + phase) + offset
    vel = (2.0*math.pi/period)*amplitude*math.cos(2.0*math.pi*t/period + phase)
    pos *= start_value
    vel *= start_value
    return pos, vel


def cos_motion(t, amplitude, phase, offset, period):
    start_value = soft_start(data.time, period)
    pos = amplitude*math.cos(2.0*math.pi*t/period + phase) + offset
    vel = -(2.0*math.pi/period)*amplitude*math.sin(2.0*math.pi*t/period + phase)
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
    viewer = mujoco_viewer.MujocoViewer(model, data, RENDER_MODE)
    viewer.scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False

    # viewer.render_mode = RENDER_MODE

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

    # use a frame counter variable
    frame_counter = 0

    # begin running model
    while not done:
        # increment one time step forward (I think this is 2 ms?)
        mujoco.mj_step(model, data)

        # check if we've exceeded duration (only in offscreen render mode)
        if (viewer.render_mode == 'offscreen') and (data.time >= DURATION):
            done = True

        # update actuator values
        setpos, setvel = sin_motion(
            data.time,
            MOTION_AMPLITUDE,
            MOTION_PHASE,
            MOTION_OFFSET,
            PERIOD
        )
        try:
            data.actuator(POS_ACTUATOR_NAME).ctrl = setpos
            data.actuator(VEL_ACTUATOR_NAME).ctrl = setvel
        except KeyError:
            pass

        # render frame
        try:
            if viewer.render_mode == 'window':
                frame = None
                viewer.render()
            else:
                frame = viewer.read_pixels()

                # save?
                if SAVE_FLAG and (data.time >= START_RECORDING):
                    # where to save image
                    # save_fn = (XML_FN + '_t={:1.3f}.png'.format(data.time))
                    save_fn = (XML_FN + '_%05d.png'%frame_counter)
                    save_path_full = SAVE_PATH / save_fn

                    # convert to PIL format and save
                    img = PIL.Image.fromarray(frame)
                    if CROP_BOX is not None:
                        img = img.crop(CROP_BOX)
                    img.save(save_path_full)
        except:
            done = True

        # tell us where how well we're controlling things
        pos = data.joint(JOINT_NAME).qpos[0]
        err = setpos - pos
        # print(f'{np.rad2deg(setpos):1.4f}, {np.rad2deg(pos):1.4f}, {np.rad2deg(err):1.4f}')

        # add error calc to lists
        setpos_list.append(np.rad2deg(setpos))
        pos_list.append(np.rad2deg(pos))
        err_list.append(np.rad2deg(err))
        t_list.append(data.time)

        # increment frame counter
        frame_counter += 1

    # close viewer
    viewer.close()

    # -------------------------------------------------------------------
    # generate plot showing angle error
    fig, (ax1, ax2) = plt.subplots(2,1)

    setpos_array = np.asarray(setpos_list)
    pos_array = np.asarray(pos_list)
    err_array = np.asarray(err_list)
    t = np.asarray(t_list)
    print(t.size)

    ax1.plot(t, setpos_array, label='set pos')
    ax1.plot(t, pos_array, label='meas pos')
    ax1.set_ylabel('deg')
    ax1.legend()

    ax2.plot(t, err_array, label='error')
    ax2.set_xlabel('time')
    ax2.set_ylabel('deg')
    ax2.legend()

    plt.show()
