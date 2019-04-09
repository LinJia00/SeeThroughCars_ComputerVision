# Canny Edge Detection:
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_canny/py_canny.html#canny

import cv2
import numpy as np
import argparse
import configparser
import sys
import os
from datetime import datetime
from Functions.FileCamera import FileCamera, FileStereoCamera


def main():
    description = "The main program of See-Through Car project for CS6320 CV."
    epilog = """\
Config the running environment in the configuration file.

Examples:
  To run the program:
    %(prog)s
  To replay the output video together with the input video:
    %(prog)s -p
    """
    # args parse
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=description, epilog=epilog)
    parser.add_argument("-p", "--play", action="store_true",
                        help="Play mode, play inputs and output.")
    parser.add_argument("-v", "--verbose", help="Verbose mode.",
                        action="store_true")
    args = parser.parse_args()
    if args.verbose:
        print("enabling verbose mode.")
    # config running environment.
    config = configparser.ConfigParser()
    config.read('./config.ini')
    # initialize
    camera_F = FileStereoCamera(config['frontCamera']['FileName'],
                                startFrame=float(config['frontCamera']['startFrame']))
    camera_B = FileCamera(config['backCamera']['FileName'],
                          startFrame=float(config['backCamera']['startFrame']))
    frameTime = 1.0 / float(config['output']['FPS'])
    pause = False
    displayShape = (int(config['output']['width']), int(config['output']['height']))
    subWinShape = tuple(ti//2 for ti in displayShape)
    textFont = cv2.FONT_HERSHEY_SIMPLEX
    textPos = (10, 30)
    textColor = (255, 255, 255)
    textScale = 1
    # run
    while(True):
        camera_F.addTime(frameTime)
        image_FL = camera_F.read('left')
        image_FR = camera_F.read('right')
        image_FD = camera_F.read('depth')
        camera_B.addTime(frameTime)
        image_B = camera_B.read()
        # Display
        timeString = datetime.fromtimestamp(camera_F.posTime()).strftime('%M:%S:%f')
        text = "Left: {:5d}, {}".format(camera_F.posFrame(), timeString)
        disp_UpLeft = cv2.putText(cv2.resize(image_FL, subWinShape),
                                  text, textPos, textFont,
                                  textScale, textColor)
        disp_UpRight = cv2.putText(cv2.resize(image_FD, subWinShape),
                                   'Front Depth', textPos, textFont,
                                   textScale, textColor)
        timeString = datetime.fromtimestamp(camera_B.posTime()).strftime('%M:%S:%f')
        text = "Back: {:5d}, {}".format(camera_B.posFrame(), timeString)
        disp_DownLeft = cv2.putText(cv2.resize(image_B, subWinShape),
                                    text, textPos, textFont,
                                    textScale, textColor)
        disp_DownRight = cv2.putText(cv2.resize(image_B, subWinShape),
                                     'Result', textPos, textFont,
                                     textScale, textColor)
        disp_Up = np.concatenate((disp_UpLeft, disp_UpRight), axis=1)
        disp_Down = np.concatenate((disp_DownLeft, disp_DownRight), axis=1)
        display = np.concatenate((disp_Up[:, :, :3], disp_Down), axis=0)
        cv2.imshow('', display)
        if pause:
            waitTime = 0
        else:
            waitTime = 1
        key = cv2.waitKey(waitTime)
        if key == ord('q'):
            break
        elif key == ord('s'):
            pause = True
        elif key == ord('p'):
            pause = False
    camera_B.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
