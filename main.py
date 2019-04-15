# Canny Edge Detection:
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_canny/py_canny.html#canny

import cv2
import numpy as np
import argparse
import configparser
import sys
import os
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from Functions.FileCamera import FileCamera, FileStereoCamera
from Functions.Output import Output


def filter_distance(matches, ratio):
    mask = np.zeros((len(matches), 2), dtype=int)
    for i, m in enumerate(matches):
        if len(m) == 2:
            if m[0].distance < ratio*m[1].distance:
                mask[i, 0] = 1
    '''
    dist = np.array([m[0].distance for m in matches])
    thres_dist = (dist.sum() / dist.shape[0]) * ratio
    mask = dist < thres_dist
    matches = np.array(matches)[mask]
    kp_1 = kp_1[mask]
    '''
    return mask


def LRFilter(matches, kp_l, kp_r, vert_dist_thres):
    mask = np.zeros(len(matches), dtype=int)
    y_l = np.array([kp.pt[1] for kp in kp_l])
    y_r = np.array([kp_r[m[0].trainIdx].pt[1] if len(m) else float('inf') for m in matches])
    # y_r = np.array([kp_r[m[0].trainIdx].pt[1] for m in matches])
    vert_dists = np.abs(y_l - y_r)
    mask[vert_dists < vert_dist_thres] = 1
    return mask.reshape((-1, 1))


def plot3DPoints(h, w, pts3, ptcolor):
    my_dpi = 72
    fig = plt.figure(figsize=(w/my_dpi, h/my_dpi), dpi=my_dpi)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts3[:, 0], pts3[:, 1], pts3[:, 2], c=ptcolor, marker='o')
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


def plot2DPoints(h, w, pts2, ptcolor):
    my_dpi = 72
    fig = plt.figure(figsize=(w/my_dpi, h/my_dpi), dpi=my_dpi)
    plt.scatter(pts2[:, 0], pts2[:, 1], c=ptcolor, marker='o')
    plt.canvas.draw()
    data = np.fromstring(plt.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(plt.canvas.get_width_height()[::-1] + (3,))
    return data


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
    pause = True
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
    startTime = float(config['general']['start time'])
    working_res = config['output']['res']
    working_res = np.fromstring(working_res, dtype=int, sep=' ')
    downScale = int(config['general']['down scale'])
    if downScale > 1:
        working_res = np.rint(working_res/downScale).astype(int)
    working_res = tuple(working_res.tolist())
    # initialize
    camera_F = FileStereoCamera(config['frontCamera']['FileName'],
                                startFrame=float(config['frontCamera']['startFrame']),
                                startTime=startTime)
    camera_B = FileCamera(config['backCamera']['FileName'],
                          startFrame=float(config['backCamera']['startFrame']),
                          startTime=startTime)
    output = Output(camera_F, camera_B, config)
    frameTime = 1.0 / float(config['output']['FPS'])
    data = dict(time=.0, nFrames=0)
    feature_method = config['general']['feature points method']
    FLANN_INDEX_KDTREE = 1
    FLANN_INDEX_LSH = 6
    if feature_method == 'orb':
        detector = cv2.ORB_create(nfeatures=2000)
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                            table_number=6,  # 12
                            key_size=12,     # 20
                            multi_probe_level=1)  # 2
    elif feature_method == 'sift':
        detector = cv2.xfeatures2d.SIFT_create()
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    else:
        raise ValueError("Don't support method:", detector)
    search_params = dict(checks=50)   # or pass empty dictionary
    mathcer = cv2.FlannBasedMatcher(index_params, search_params)
    mtx_b = np.array([[972.822, 0, 636.455], [0, 879.505, 216.031], [0, 0, 1]])
    dist_b = np.array([0.1356, -0.7613, 0.0273, -0.0368, 0])  #np.zeros(5)
    rvecs, tvecs = None, None
    drawKP = config['general']['draw keypoints'] = 'yes'
    mask_b = config['backCamera']['mask']
    mask_b = np.fromstring(mask_b, dtype=int, sep=' ')
    mask_b1 = tuple(mask_b[:2].tolist())
    mask_b2 = tuple(mask_b[2:].tolist())
    mask_b = np.zeros((720, 1280), dtype="uint8")
    cv2.rectangle(mask_b, mask_b1, mask_b2, 255, -1)
    mask_f = config['frontCamera']['mask']
    mask_f = np.fromstring(mask_f, dtype=int, sep=' ')
    mask_f1 = tuple(mask_f[:2].tolist())
    mask_f2 = tuple(mask_f[2:].tolist())
    mask_f = np.zeros((720, 1280), dtype="uint8")
    cv2.rectangle(mask_f, mask_f1, mask_f2, 255, -1)
    if downScale > 1:
        mask_b = cv2.resize(mask_b, working_res)
        mask_f = cv2.resize(mask_f, working_res)

    useExtrinsicGuess = 0
    # run
    while(True):
        data['time'] += frameTime
        data['nFrames'] += 1
        camera_F.addTime(frameTime)
        image_FL = camera_F.read('left')
        image_FR = camera_F.read('right')
        image_FD = camera_F.read('depth')
        camera_B.addTime(frameTime)
        image_B = camera_B.read()
        if downScale > 1:
            image_FL = cv2.resize(image_FL, working_res)
            image_FR = cv2.resize(image_FR, working_res)
            image_FD = cv2.resize(image_FD, working_res)
            image_B = cv2.resize(image_B, working_res)

        # Front Features
        data['kp_l'], des_L = detector.detectAndCompute(image_FL, mask=mask_f)
        data['kp_r'], des_R = detector.detectAndCompute(image_FR, mask=mask_f)
        data['kp_l'] = np.array(data['kp_l'])
        data['kp_r'] = np.array(data['kp_r'])
        data['lr_matches'] = mathcer.knnMatch(des_L, des_R, k=1)
        data['lr_mask'] = LRFilter(data['lr_matches'], data['kp_l'], data['kp_r'], 1)
        print('number of Front features:', data['lr_mask'].sum())
        data['kp_fl'] = data['kp_l'][data['lr_mask'][:, 0] == 1]
        des_L = des_L[data['lr_mask'][:, 0] == 1]
        # 3d map
        '''
        p2d = np.array([kp.pt for kp in kp_L])
        #kp2d[:, 0], kp2d[:, 1] = kp2d[:, 1], kp2d[:, 0].copy()
        p2d = np.rint(p2d).astype(int)  # index in (collumn/x, row/y)
        print('number of Front features:', p2d.shape[0])
        P3D_L = camera_F.read('3D')[:, :, :3]  # has shape rows/y x collumns/x
        p_3d = P3D_L[p2d[:, 1], p2d[:, 0], :]
        mask = ~np.isnan(p_3d[:, 0])
        kp_L = kp_L[mask]
        des_L = des_L[mask]
        #kp3d = kp3d[~np.isnan(kp3d)].reshape((-1, 4))[:, :3]
        print('number of 3D features:', p_3d.shape[0])
        print(np.count_nonzero(np.isnan(P3D_L))//3, P3D_L.shape[0]*P3D_L.shape[1])
        '''
        # Back Features
        kp_B, des_B = detector.detectAndCompute(image_B, mask=mask_b)
        matches = mathcer.knnMatch(des_L, des_B, k=2)
        data['fb_matches'] = matches
        data['kp_b'] = kp_B
        data['fb_mask'] = filter_distance(matches, .8)
        if downScale > 1:
            for kp in data['kp_b']:
                kp.pt = (kp.pt[0]*downScale, kp.pt[1]*downScale)
            for kp in data['kp_l']:
                kp.pt = (kp.pt[0]*downScale, kp.pt[1]*downScale)
            for kp in data['kp_r']:
                kp.pt = (kp.pt[0]*downScale, kp.pt[1]*downScale)
        '''
        good_matches = [[matches[i][0]] for i in range(len(matches)) if mask[i, 0]]
        #matches = matches[mask == 1].reshape((-1, 1))
        mask = np.sum(mask, axis=1)
        kp_L = kp_L[mask == 1]
        # p2_b = [kp_B[m[0].trainIdx] for m in matches]
        p2_l = np.array([kp.pt for kp in kp_L])
        print('number of matched 3D features:', p_3d.shape[0])
        # Back pose estimate
        if p_3d.shape[0] > 4:
            if useExtrinsicGuess:
                values = cv2.solvePnPRansac(p_3d, p_2d, mtx_b, dist_b, rvecs,
                                            tvecs, 1, reprojectionError=5)
                ret, rvecs, tvecs, inliers = values
            else:
                values = cv2.solvePnPRansac(p_3d, p_2d, mtx_b, dist_b,
                                            iterationsCount=2000,
                                            reprojectionError=5)
                ret, rvecs, tvecs, inliers = values
        else:
            inliers = None
        if inliers is None:
            print('number of inliers:', 0)
        elif len(inliers) < 10:
            print('number of inliers:', 0)
        else:
            useExtrinsicGuess = 1
            p_3d = p_3d[inliers.reshape(-1)]
            print('number of inliers:', len(inliers))
            # project 3D points to image plane
            proj_pts, jac = cv2.projectPoints(p_3d, rvecs, tvecs, mtx_b, dist_b)
            for point in proj_pts:
                cv2.circle(image_B, tuple(point[0]), 3, (0, 0, 255), -1)
        '''
        # Display
        stop = output.display(None, data)
        if stop:
            break
        '''
        timeString = datetime.fromtimestamp(camera_F.posTime()).strftime('%M:%S:%f')
        text = "Left: {:5d}, {}".format(camera_F.posFrame(), timeString)
        if drawKP:
            cv2.drawKeypoints(image_FL, kp_L, image_FL, color=(0, 255, 0), flags=0)
        disp_UpLeft = cv2.resize(image_FL, subWinShape)
        cv2.putText(disp_UpLeft, text, textPos, textFont, textScale, textColor)
        #print(p_3d)
        mask = ~np.isnan(P3D_L)[:, :, 0]
        pt3d = P3D_L[mask]
        ptcolor = image_FL[mask].astype(np.int)
        proj_pts, jac = cv2.projectPoints(pt3d, camera_F.R, camera_F.T, np.array(camera_F.left_mtx), camera_F.left_disto)
        proj_pts = proj_pts.reshape((-1,2)).astype(int)
        disp_UpRight = np.zeros_like(image_FL, dtype=np.uint8)
        #print(disp_UpRight.shape)
        #print(proj_pts[0], ptcolor[0], disp_UpRight[proj_pts[0][1], proj_pts[0][0]])
        #print(proj_pts.shape, ptcolor.shape, disp_UpRight[proj_pts.astype(int)].shape)
        disp_UpRight[np.clip(proj_pts[:,1], 0, 720-1), np.clip(proj_pts[:,0], 0, 1280-1)] = ptcolor
        disp_UpRight = cv2.resize(disp_UpRight, subWinShape)
        mask = ~np.isnan(P3D_L)[:, :, 0]
        pt3d = P3D_L[mask]
        ptcolor = image_FL[mask].astype(float) / 255
        print(pt3d[0,0], ptcolor[0,0])
        disp_UpRight = plot3DPoints(360, 640, pt3d, ptcolor)
        '''
        '''
        text = "Right: {:5d}, {}".format(camera_F.posFrame(), timeString)
        if drawKP:
            cv2.drawKeypoints(image_FR, kp_R, image_FR, color=(0, 255, 0), flags=0)
        disp_UpRight = cv2.resize(image_FR, subWinShape)
        cv2.putText(disp_UpRight, text, textPos, textFont, textScale, textColor)
        '''
        '''
        disp_UpRight = cv2.resize(image_FD, subWinShape)
        disp_DownRight = cv2.resize(image_B, subWinShape)
        cv2.putText(disp_DownRight, 'Result', textPos, textFont, textScale, textColor)
        timeString = datetime.fromtimestamp(camera_B.posTime()).strftime('%M:%S:%f')
        text = "Back: {:5d}, {}".format(camera_B.posFrame(), timeString)
        if drawKP:
            cv2.drawKeypoints(image_B, kp_B, image_B, color=(0, 255, 0), flags=0)
        disp_DownLeft = cv2.resize(image_B, subWinShape)
        cv2.putText(disp_DownLeft, text, textPos, textFont, textScale, textColor)
        disp_Up = np.concatenate((disp_UpLeft, disp_UpRight), axis=1)
        disp_Down = np.concatenate((disp_DownLeft, disp_DownRight), axis=1)
        #img_LR = drawMatch(matches, image_FL, kp_L, image_FR, kp_R)
        #img_LR = cv2.resize(img_LR, (2*subWinShape[0], subWinShape[1]))
        display = np.concatenate((disp_Up[:, :, :3], disp_Down), axis=0)
        # cv2.imshow('', img)
        '''
    camera_B.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
