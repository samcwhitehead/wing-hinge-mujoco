"""
Quick script to test out some basic wing hinge modeling in Mujoco

Here just recreating basic setup from mujoco colab tutorial:
https://colab.research.google.com/github/deepmind/mujoco/blob/main/python/tutorial.ipynb#scrollTo=SdkwLeGUp9B2

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

# ---------------------------------------
# CONSTANTS
XML_FN = 'basic_demo0'
XML_PATH = '/home/sam/Documents/code/wing-hinge-mujoco/models'

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

    # reset data
    mujoco.mj_resetData(model, data)
    
    # begin running model
    while not done:
        # increment one time step forward (I think this is 2 ms?)
        mujoco.mj_step(model, data)

        # render frame
        try:
            if viewer.render_mode == 'window':
                frame = None
                viewer.render()
            else:
                frame = viewer.read_pixels()
        except:
            done = True

    # close viewer
    viewer.close()

