'''
Class for normal single camera.
'''
import cv2
import pyzed.sl as sl


class FileCamera(object):
    """Normal single camera."""

    def __init__(self, fileName, startFrame=0, startTime=0):
        self.fileName = fileName
        self.mtx = [[972.822, 0, 636.455], [0, 879.505, 216.031], [0, 0, 1]]
        self.disto = [0.1356, -0.7613, 0.0273, -0.0368, 0]
        self.cap = cv2.VideoCapture(self.fileName)
        #self.FPS = self.cap.get(cv2.CAP_PROP_FPS)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, startFrame - 1)
        self.cap.read()
        self.camTime = self.cap.get(cv2.CAP_PROP_POS_MSEC)
        self.camTime += startTime*1000
        #print(startFrame, self.cap.get(cv2.CAP_PROP_POS_FRAMES), self.camTime)

    def addTime(self, time):
        self.camTime += 1000 * time
        while self.cap.isOpened():
            if self.cap.get(cv2.CAP_PROP_POS_MSEC) > self.camTime:
                break
            ret, self.image = self.cap.read()
            assert ret, "Failed to read image."

    def read(self):
        return cv2.cvtColor(self.image, cv2.COLOR_RGBA2RGB)

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

    def __init__(self, fileName, FPS=30, startFrame=0, startTime=0):
        self.fileName = fileName
        self.cam = sl.Camera()
        init = sl.InitParameters(svo_input_filename=self.fileName,
                                 svo_real_time_mode=False)
        init.depth_mode = sl.DEPTH_MODE.DEPTH_MODE_ULTRA
        init.coordinate_units = sl.UNIT.UNIT_METER
        init.depth_minimum_distance = 2 # meter
        status = self.cam.open(init)
        assert status == sl.ERROR_CODE.SUCCESS, repr(status)
        left_cam = self.cam.get_camera_information().calibration_parameters.left_cam
        self.left_mtx = [[left_cam.fx, 0, left_cam.cx], [0, left_cam.fy, left_cam.cy], [0, 0, 1]]
        self.left_disto = left_cam.disto
        self.T = self.cam.get_camera_information().calibration_parameters.T
        self.R = self.cam.get_camera_information().calibration_parameters.R
        self.cam.set_depth_max_range_value(40) # meter
        self.width = self.cam.get_resolution().width;
        self.height = self.cam.get_resolution().height;
        self.cam.set_svo_position(startFrame-1)
        self.runtime = sl.RuntimeParameters()
        self.runtime.sensing_mode = sl.SENSING_MODE.SENSING_MODE_FILL
        self.cam.grab(self.runtime)
        self.camTime = self.cam.get_timestamp(sl.TIME_REFERENCE.TIME_REFERENCE_IMAGE)
        self.camTime += startTime*1e9
        self.mat_L = sl.Mat()
        self.mat_R = sl.Mat()
        self.mat_D = sl.Mat()
        self.mat_3D = sl.Mat()

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
        err = self.cam.retrieve_measure(self.mat_3D, sl.MEASURE.MEASURE_XYZ)
        assert err == sl.ERROR_CODE.SUCCESS, "Failed to retrieve image: {}.".format(err)

    def read(self, view='left'):
        if view == 'depth':
            return cv2.cvtColor(self.mat_D.get_data(), cv2.COLOR_RGBA2RGB)
        if view == 'right':
            return cv2.cvtColor(self.mat_R.get_data(), cv2.COLOR_RGBA2RGB)
        if view == '3D':
            return self.mat_3D.get_data()
        else:
            return cv2.cvtColor(self.mat_L.get_data(), cv2.COLOR_RGBA2RGB)

    def posFrame(self):
        return int(self.cam.get_svo_position())

    def posTime(self):
        return self.cam.get_timestamp(sl.TIME_REFERENCE.TIME_REFERENCE_IMAGE) / 1e9

    def isOpened(self):
        return self.cam.isOpened()
