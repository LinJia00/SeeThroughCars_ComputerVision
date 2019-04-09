'''
Class for normal single camera.
'''
import cv2
import pyzed.sl as sl


class FileCamera(object):
    """Normal single camera."""

    def __init__(self, fileName, startFrame=0):
        self.fileName = fileName
        self.cap = cv2.VideoCapture(self.fileName)
        #self.FPS = self.cap.get(cv2.CAP_PROP_FPS)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, startFrame - 1)
        self.cap.read()
        self.camTime = self.cap.get(cv2.CAP_PROP_POS_MSEC)
        #print(startFrame, self.cap.get(cv2.CAP_PROP_POS_FRAMES), self.camTime)

    def addTime(self, time):
        self.camTime += 1000 * time
        while self.cap.isOpened():
            if self.cap.get(cv2.CAP_PROP_POS_MSEC) > self.camTime:
                break
            ret, self.image = self.cap.read()
            assert ret, "Failed to read image."

    def read(self):
        return self.image

    def posFrame(self):
        return int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

    def posTime(self):
        return self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000

    def isOpened(self):
        return self.cap.isOpened()

    def release(self):
        self.cap.release()


class FileStereoCamera(object):
    """Normal single camera."""

    def __init__(self, fileName, FPS=30, startFrame=0):
        self.fileName = fileName
        self.cam = sl.Camera()
        init = sl.InitParameters(svo_input_filename=self.fileName,
                                 svo_real_time_mode=False)
        status = self.cam.open(init)
        assert status == sl.ERROR_CODE.SUCCESS, repr(status)
        self.cam.set_svo_position(startFrame-1)
        self.runtime = sl.RuntimeParameters()
        self.cam.grab(self.runtime)
        self.camTime = self.cam.get_timestamp(sl.TIME_REFERENCE.TIME_REFERENCE_IMAGE)
        self.mat_L = sl.Mat()
        self.mat_R = sl.Mat()
        self.mat_D = sl.Mat()

    def addTime(self, time):
        self.camTime += int(time * 1e9)
        while True:
            camTime = self.cam.get_timestamp(sl.TIME_REFERENCE.TIME_REFERENCE_IMAGE)
            if camTime > self.camTime:
                break
            err = self.cam.grab(self.runtime)
            assert err == sl.ERROR_CODE.SUCCESS, "Failed to grab frame: {}.".format(err)
        err = self.cam.retrieve_image(self.mat_D, sl.VIEW.VIEW_DEPTH)
        assert err == sl.ERROR_CODE.SUCCESS, "Failed to retrieve image: {}.".format(err)
        err = self.cam.retrieve_image(self.mat_R, sl.VIEW.VIEW_RIGHT)
        assert err == sl.ERROR_CODE.SUCCESS, "Failed to retrieve image: {}.".format(err)
        err = self.cam.retrieve_image(self.mat_L)
        assert err == sl.ERROR_CODE.SUCCESS, "Failed to retrieve image: {}.".format(err)

    def read(self, view='left'):
        if view == 'depth':
            return self.mat_D.get_data()
        if view == 'right':
            return self.mat_R.get_data()
        else:
            return self.mat_L.get_data()

    def posFrame(self):
        return int(self.cam.get_svo_position())

    def posTime(self):
        return self.cam.get_timestamp(sl.TIME_REFERENCE.TIME_REFERENCE_IMAGE) / 1e9

    def isOpened(self):
        return self.cam.isOpened()
