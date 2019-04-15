'''
Class for output.
'''
import numpy as np
import cv2
import pyzed.sl as sl
from datetime import datetime
import configparser
from Functions.FileCamera import FileCamera, FileStereoCamera


class Output(object):
    """Normal single camera."""

    def __init__(self, camera_f, camera_b, config):
        self.filename = None
        self.camera_f = camera_f
        self.camera_b = camera_b
        self.layout = config['layout']
        self.res = (int(config['width']), int(config['height']))
        self.drawKP = config['draw keypoints'] == 'yes'
        self.showCamInfo = config['show cam info'] == 'yes'
        self.showSysInfo = config['show sys info'] == 'yes'
        self.colors = np.random.randint(0, 255, (500, 3)).tolist()
        self.textFont = cv2.FONT_HERSHEY_SIMPLEX
        self.textPos = (10, 30)
        self.textColor = (255, 255, 255)
        self.textScale = 1
        self.textPosGlobal = (800, 710)
        self.pause = True

    def display(self, result, data):
        self.data = data
        if self.layout == 'L':
            img = self._l()
        elif self.layout == 'LR BL':
            img = self._lr_bl()
        elif self.layout == 'MLR LR':
            img = self._mlr_lr()
        elif self.layout == 'MLR MLB':
            img = self._mlr_mlb()
        else:
            raise ValueError("Don't support layout:", self.layout)
        if self.showSysInfo:
            timeStr = datetime.fromtimestamp(data['time']).strftime('%M:%S:%f')
            text = "Frame: {:4d}; Time: {}".format(data['nFrames'], timeStr[:-3])
            cv2.putText(img, text, self.textPosGlobal, self.textFont,
                        self.textScale, self.textColor)
        # show
        cv2.imshow('', img)
        if self.pause:
            waitTime = 0
        else:
            waitTime = 1
        key = cv2.waitKey(waitTime)
        if key == ord('q'):
            return True
        elif key == ord('s'):
            self.pause = True
        elif key == ord('p'):
            self.pause = False
        return False

    def _lr_bl(self):
        img_lr = self._lr()
        img_bl = self._bl()
        img = np.concatenate((img_lr, img_bl), axis=0)
        if img.shape[:2] != self.res:
            img = cv2.resize(img, self.res)
        return img

    def _mlr_lr(self):
        img_mlr = self._mlr()
        img_lr = self._lr()
        img = np.concatenate((img_mlr, img_lr), axis=0)
        if img.shape[:2] != self.res:
            img = cv2.resize(img, self.res)
        return img

    def _mlr_mlb(self):
        img_lr = self._mlr()
        img_bl = self._mlb()
        img = np.concatenate((img_lr, img_bl), axis=0)
        if img.shape[:2] != self.res:
            img = cv2.resize(img, self.res)
        return img

    def _l(self):
        img = self.camera_f.read('left')[:, :, :3]
        if img.shape[:2] != self.res:
            img = cv2.resize(img, self.res)
        if self.showCamInfo:
            self._addCamTitle(img, 'Left', self.camera_f, self.textPos)
        return img

    def _r(self):
        img = self.camera_f.read('right')[:, :, :3]
        if img.shape[:2] != self.res:
            img = cv2.resize(img, self.res)
        if self.showCamInfo:
            self._addCamTitle(img, 'Right', self.camera_f, self.textPos)
        return img

    def _b(self):
        img = self.camera_b.read()
        if img.shape[:2] != self.res:
            img = cv2.resize(img, self.res)
        if self.showCamInfo:
            self._addCamTitle(img, 'Back', self.camera_b, self.textPos)
        return img

    def _lr(self):
        img_l = self.camera_f.read('left')[:, :, :3]
        img_r = self.camera_f.read('right')[:, :, :3]
        img_l = cv2.resize(img_l, (self.res[0]//2, self.res[1]//2))
        img_r = cv2.resize(img_r, (self.res[0]//2, self.res[1]//2))
        img = np.concatenate((img_l, img_r), axis=1)
        return img

    def _mlr(self):
        img_l = self.camera_f.read('left')[:, :, :3]
        img_r = self.camera_f.read('right')[:, :, :3]
        matches = self.data['lr_matches']
        kp_l = self.data['kp_l2r']
        kp_r = self.data['kp_r']
        img = self._drawMatch(matches, img_l, kp_l, img_r, kp_r)
        img = cv2.resize(img, (self.res[0], self.res[1]//2))
        text = "Left-Right Match, n_matches: {:3d}".format(len(matches))
        cv2.putText(img, text, self.textPos, self.textFont, self.textScale,
                    self.textColor)
        return img

    def _bl(self):
        img_b = self.camera_b.read()[:, :, :3]
        img_l = self.camera_f.read('left')[:, :, :3]
        img_l = cv2.resize(img_l, (self.res[0]//2, self.res[1]//2))
        img_b = cv2.resize(img_b, (self.res[0]//2, self.res[1]//2))
        img = np.concatenate((img_b, img_l), axis=1)
        return img

    def _mlb(self):
        img_b = self.camera_b.read()[:, :, :3]
        img_l = self.camera_f.read('left')[:, :, :3]
        matches = self.data['bl_matches']
        kp_b = self.data['kp_b']
        kp_l = self.data['kp_l2b']
        img = self._drawMatch(matches, img_l, kp_l, img_b, kp_b)
        img = cv2.resize(img, (self.res[0], self.res[1]//2))
        text = "Left-Back Match, n_matches: {:3d}".format(len(matches))
        cv2.putText(img, text, self.textPos, self.textFont, self.textScale,
                    self.textColor)
        return img

    def _addCamTitle(self, img, title, cam, pos):
        time = datetime.fromtimestamp(cam.posTime())
        timeStr = time.strftime('%M:%S:%f')
        nFrame = cam.posFrame()
        text = "{}: {:5d}, {}".format(title, nFrame, timeStr[:-3])
        cv2.putText(img, text, pos, self.textFont, self.textScale,
                    self.textColor)

    def _drawMatch(self, matches, img1, kp1, img2, kp2):
        for i in range(len(matches)):
            matches[i][0].queryIdx = i
        draw_params = dict(matchColor=(-1, -1, -1),
                           singlePointColor=(255, 0, 0),
                           matchesMask=None,
                           flags=0)
        return cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches,
                                  None, **draw_params)
