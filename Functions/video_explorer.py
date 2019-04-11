'''
The goal of this code is to explorer a video. It provide following functions:
    1. play video (p)
    2. reverse video (r)
    2. stop video (s)
    3. next frame (d)
    4. previous frame (a)
    5. next 10th frame (->)
    6. previous 10th frame (<-)
    7. print current frame number (c)
    8. save current frame as a file ( )
    8. quit (q)

The frame number is 0 based.

example:
py Functions\video_explorer.py Inputs\2019-04-03-16-38-48.MP4
'''
import argparse
import numpy as np
import cv2
import time
import os


if __name__ == '__main__':
    # read arguments.
    startFrame = 0
    path = './Outputs/'
    baseFileName = 'cheesboard'
    extFileName = '.tif'
    description = "Video explorer."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('infile', type=argparse.FileType('r'))
    parser.add_argument("-v", "--verbose",
                        help="pause and show when outputing images",
                        action="store_true")
    args = parser.parse_args()
    cap = cv2.VideoCapture(args.infile.name)
    cap.set(cv2.CAP_PROP_POS_FRAMES, startFrame)
    playSpeed = 0
    textFont = cv2.FONT_HERSHEY_SIMPLEX
    textPos = (10, 30)
    textColor = (255, 255, 255)
    textScale = 1
    while cap.isOpened():
        current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC)
        ret, frame = cap.read()
        frame = cv2.putText(frame, str(current_frame) + str(current_time), textPos, textFont,
                            textScale, textColor)
        cv2.imshow('frame', frame)
        if playSpeed == 0:
            waitTime = 0
        else:
            waitTime = 1
        key = cv2.waitKey(waitTime)
        if key == ord('q'):
            break
        elif key == ord('p'):
            playSpeed = 1
        elif key == ord('r'):
            playSpeed = -1
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame - 1)
        elif key == ord('s'):
            playSpeed = 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        elif key == ord('a'):
            playSpeed = 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame - 1)
        elif key == ord('d'):
            playSpeed = 0
        elif key == ord('c'):
            print(current_frame)
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        elif key == ord(' '):
            outfile = os.path.join(path, baseFileName + str(current_frame) + extFileName)
            cv2.imwrith(outfile, frame)
        else:
            if playSpeed == 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            elif playSpeed == -1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame - 1)

    cap.release()
    cv2.destroyAllWindows()
