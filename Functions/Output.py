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
        self.layout = config['output']['layout']
        self.res = (int(config['output']['width']), int(config['output']['height']))
        self.drawKP = config['output']['draw keypoints'] == 'yes'
        self.drawMask = config['output']['draw mask'] == 'yes'
        self.drawPnP = config['output']['draw repreject'] == 'yes'
        self.maskColor = (0, 0, 255)
        mask_b = config['backCamera']['mask']
        mask_b = np.fromstring(mask_b, dtype=int, sep=' ')
        self.mask_b1 = tuple(mask_b[:2].tolist())
        self.mask_b2 = tuple(mask_b[2:].tolist())
        mask_f = config['frontCamera']['mask']
        mask_f = np.fromstring(mask_f, dtype=int, sep=' ')
        self.mask_f1 = tuple(mask_f[:2].tolist())
        self.mask_f2 = tuple(mask_f[2:].tolist())
        self.showCamInfo = config['output']['show cam info'] == 'yes'
        self.showSysInfo = config['output']['show sys info'] == 'yes'
        self.head_params = dict(fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                org=(10, 30), fontScale=1,
                                color=(255, 255, 255))
        self.foot_params = dict(fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                org=(550, 710), fontScale=1,
                                color=(255, 255, 255))
        self.matche_params = dict(matchColor=(-1, -1, -1), flags=4,
                                  singlePointColor=(150, 150, 150))
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
        elif self.layout == 'MLB L3 B3':
            img = self._mlb_l3_b3()
        else:
            raise ValueError("Don't support layout:", self.layout)
        if self.showSysInfo:
            timeStr = datetime.fromtimestamp(data['time']).strftime('%M:%S:%f')
            text = "Frame: {:4d}; Time: {}; PnP: {}/{}".format(data['nFrames'],
                    timeStr[:-3], len(data['pnp_inlier']), len(data['pnp_2d']))
            cv2.putText(img, text, **self.foot_params)
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

    def _mlb_l3_b3(self):
        img_u = self._mlb()
        img_d = self._l3_b3()
        img = np.concatenate((img_u, img_d), axis=0)
        if img.shape[:2] != self.res:
            img = cv2.resize(img, self.res)
        return img

    def _l(self):
        img = self.camera_f.read('left')[:, :, :3]
        if img.shape[:2] != self.res:
            img = cv2.resize(img, self.res)
        if self.showCamInfo:
            self._addCamTitle(img, 'Left', self.camera_f)
        return img

    def _r(self):
        img = self.camera_f.read('right')[:, :, :3]
        if img.shape[:2] != self.res:
            img = cv2.resize(img, self.res)
        if self.showCamInfo:
            self._addCamTitle(img, 'Right', self.camera_f)
        return img

    def _b(self):
        img = self.camera_b.read()
        if img.shape[:2] != self.res:
            img = cv2.resize(img, self.res)
        if self.showCamInfo:
            self._addCamTitle(img, 'Back', self.camera_b)
        return img

    def _lr(self):
        img_l = self.camera_f.read('left')[:, :, :3]
        img_r = self.camera_f.read('right')[:, :, :3]
        if self.drawKP:
            img_l = cv2.drawKeypoints(img_l, self.data['kp_l'], None, color=(0, 255, 0), flags=4)
            img_r = cv2.drawKeypoints(img_r, self.data['kp_r'], None, color=(0, 255, 0), flags=4)
        img_l = cv2.resize(img_l, (self.res[0]//2, self.res[1]//2))
        img_r = cv2.resize(img_r, (self.res[0]//2, self.res[1]//2))
        img = np.concatenate((img_l, img_r), axis=1)
        return img

    def _mlr(self):
        img_l = self.camera_f.read('left')[:, :, :3]
        img_r = self.camera_f.read('right')[:, :, :3]
        if self.drawMask:
            cv2.rectangle(img_r, self.mask_f1, self.mask_f2, self.maskColor, 2)
            cv2.rectangle(img_l, self.mask_f1, self.mask_f2, self.maskColor, 2)
        matches = self.data['lr_matches']
        kp_l = self.data['kp_l']
        kp_r = self.data['kp_r']
        mask = self.data['lr_mask']
        img = cv2.drawMatchesKnn(img_l, kp_l, img_r, kp_r, matches, None,
                                 matchesMask=mask.tolist(),
                                 **self.matche_params)
        img = cv2.resize(img, (self.res[0], self.res[1]//2))
        text = "Left-Right Match, n_matches: {:3d}".format(mask.sum())
        cv2.putText(img, text, **self.head_params)
        return img

    def _bl(self):
        img_b = self.camera_b.read()[:, :, :3]
        img_l = self.camera_f.read('left')[:, :, :3]
        if self.drawKP:
            img_l = cv2.drawKeypoints(img_l, self.data['kp_l'], None, color=(0, 255, 0), flags=4)
            img_b = cv2.drawKeypoints(img_b, self.data['kp_b'], None, color=(0, 255, 0), flags=4)
        img_l = cv2.resize(img_l, (self.res[0]//2, self.res[1]//2))
        img_b = cv2.resize(img_b, (self.res[0]//2, self.res[1]//2))
        img = np.concatenate((img_b, img_l), axis=1)
        return img

    def _l3_b3(self):
        xyz = self.camera_f.read('3D')[:, :, :3]
        img_1 = self.camera_f.read('left')[:, :, :3]
        mask = np.isnan(xyz)[:, :, 0]
        img_1[mask] = np.zeros(3)
        pt_xyz = xyz[~mask]
        pt_rgb = img_1[~mask]
        img_2 = np.zeros_like(img_1, dtype=np.uint8)
        if self.data['R, T']:
            R, T = self.data['R, T']
            Mtx = np.array(self.camera_b.mtx)
            Disto = np.array(self.camera_b.disto)
            proj_pts, jac = cv2.projectPoints(pt_xyz, R, T, Mtx, Disto)
            proj_pts = proj_pts.reshape((-1, 2)).astype(int)
            Y = np.clip(proj_pts[:, 1], 0, 720-1)
            X = np.clip(proj_pts[:, 0], 0, 1280-1)
            img_2[Y, X] = pt_rgb
        img_l = cv2.resize(img_1, (self.res[0]//2, self.res[1]//2))
        img_2 = cv2.resize(img_2, (self.res[0]//2, self.res[1]//2))
        if self.showCamInfo:
            self._addCamTitle(img_1, 'Left 3D', self.camera_f)
            self._addCamTitle(img_2, 'Front to Back', self.camera_f)
        img = np.concatenate((img_l, img_2), axis=1)
        return img

    def _mlb(self):
        img_b = self.camera_b.read()[:, :, :3]
        img_l = self.camera_f.read('left')[:, :, :3]
        if self.drawMask:
            cv2.rectangle(img_b, self.mask_b1, self.mask_b2, self.maskColor, 2)
            cv2.rectangle(img_l, self.mask_f1, self.mask_f2, self.maskColor, 2)
        if self.drawPnP:
            for i in range(len(self.data['pnp_inlier'])):
                cv2.circle(img_b, tuple(self.data['pnp_inlier'][i]), 10, (0, 255, 0), -1)
                cv2.circle(img_b, tuple(self.data['repreject_p2'][i][0]), 8, (0, 0, 255), -1)
        matches = self.data['fb_matches']
        kp_b = self.data['kp_b']
        kp_l = self.data['kp_fl']
        mask = self.data['fb_mask']
        img = cv2.drawMatchesKnn(img_l, kp_l, img_b, kp_b, matches, None,
                                 matchesMask=mask.tolist(),
                                 **self.matche_params)
        img = cv2.resize(img, (self.res[0], self.res[1]//2))
        text = "Left-Back Match, n_matches: {:3d}".format(mask.sum())
        cv2.putText(img, text, **self.head_params)
        return img

    def _addCamTitle(self, img, title, cam):
        time = datetime.fromtimestamp(cam.posTime())
        timeStr = time.strftime('%M:%S:%f')
        nFrame = cam.posFrame()
        text = "{}: {:5d}, {}".format(title, nFrame, timeStr[:-3])
        cv2.putText(img, text, **self.head_params)
