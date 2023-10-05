"""
Quick script to test out some basic wing hinge modeling in Mujoco

I want to get something minimal working to confirm:
    1) Blender -> Mujoco pipeline
    2) ability to get motion I want without model dictated to me.

To do this, looking at just pleural wing process (PWP) and second
 axillary sclerite (Ax2). Want Ax2 to rotate around PWP.

Note that this is essentially a copy of the jupyter notebook
test_basic_hinge_joint.ipynb. This just uses mujoco-viewer and
makes it faster to test once something in the model updates

"""
# ----------------------------------------
# IMPORTS

import os
import mujoco
import math
import signal
import mujoco_viewer
import pathlib

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------
# CONSTANTS
XML_FN = 'hinge1'
CURR_PATH = pathlib.Path(__file__).parent
XML_PATH = CURR_PATH.parent / 'models'
print(XML_PATH)

RENDER_MODE = 'window'  # 'offscreen'

CAM_DISTANCE = 50
CAM_AZIMUTH = 20
CAM_ELEVATION = -45
CAM_LOOKAT = [0.0, 0.0, 0.0]

AX2_AMPLITUDE = np.deg2rad(40)
AX2_PHASE = np.deg2rad(0.0)
AX2_OFFSET = -AX2_AMPLITUDE
PERIOD = 0.5
            
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
        ax2_setpos, ax2_setvel = sin_motion(
            data.time,
            AX2_AMPLITUDE,
            AX2_PHASE,
            AX2_OFFSET,
            PERIOD
        )
        try:
            data.actuator('Ax2_position').ctrl = ax2_setpos
            data.actuator('Ax2_velocity').ctrl = ax2_setvel
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
        ax2_pos = data.joint("pwp_ax2_joint").qpos[0]
        ax2_err = ax2_setpos - ax2_pos
        print(f'{np.rad2deg(ax2_setpos):1.4f}, {np.rad2deg(ax2_pos):1.4f}, {np.rad2deg(ax2_err):1.4f}')

        # add error calc to lists
        setpos_list.append(np.rad2deg(ax2_setpos))
        pos_list.append(np.rad2deg(ax2_pos))
        err_list.append(np.rad2deg(ax2_err))
        t_list.append(data.time)

    # close viewer
    viewer.close()

    # generate plot
    fig, (ax1, ax2) = plt.subplots(2,1)

    setpos = np.asarray(setpos_list)
    pos = np.asarray(pos_list)
    err = np.asarray(err_list)
    t = np.asarray(t_list)

    ax1.plot(t, setpos, label='set pos')
    ax1.plot(t, pos, label='meas pos')
    ax1.set_ylabel('deg')
    ax1.legend()

    ax2.plot(t, err, label='error')
    ax2.set_xlabel('time')
    ax2.set_ylabel('deg')
    ax2.legend()

    plt.show()
