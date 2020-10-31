# -*- coding: utf-8 -*-

import sys
from hw1_ui import Ui_MainWindow
import cv2 as cv
import numpy as np
import glob
import os
from PyQt5.QtWidgets import QMainWindow, QApplication

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.onBindingUI()

    def onBindingUI(self):
        self.btn1_1.clicked.connect(self.on_btn1_1_click)
        self.btn1_2.clicked.connect(self.on_btn1_2_click)
        self.btn1_3.clicked.connect(self.on_btn1_3_click)
        self.btn1_4.clicked.connect(self.on_btn1_4_click)
        self.btn2_1.clicked.connect(self.on_btn2_1_click)
        self.btn3_1.clicked.connect(self.on_btn3_1_click)
        self.btn3_2.clicked.connect(self.on_btn3_2_click)
        self.btn4_1.clicked.connect(self.on_btn4_1_click)
        self.btn4_2.clicked.connect(self.on_btn4_2_click)

    def on_btn1_1_click(self):

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((8*11,3), np.float32)
        objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        #read all the images from folder
        images= glob.glob('../Q1_Image/*.bmp')

        i=0
        for fname in images:
            img = cv.imread(fname)
            if img is None:
                print("Failed to load", fn)
                return None
            gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv.findChessboardCorners(gray, (11,8),None)

            # If found, add object points, image points (after refining them)
            i=i+1
            if ret == True:
                objpoints.append(objp)
                corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),(cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                imgpoints.append(corners2)

                # Draw and display the corners
                img = cv.drawChessboardCorners(img, (11,8), corners2,ret)

                cv.namedWindow(str(i),cv.WINDOW_GUI_NORMAL )
                cv.imshow(str(i),img)
                cv.waitKey(500)

    cv.destroyAllWindows()

    def on_btn1_2_click(self):
        # termination criteria
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((8*11,3), np.float32)
        objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        #read all the images from folder
        images= glob.glob('../images/CameraCalibration/*.bmp')

        i=0
        for fname in images:
            img = cv.imread(fname)
            gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv.findChessboardCorners(gray, (11,8),None)

            # If found, add object points, image points (after refining them)
            i=i+1
            if ret == True:
                objpoints.append(objp)
                corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                imgpoints.append(corners2)
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
        print(mtx)
        cv.waitKey(500)
    cv.destroyAllWindows()

    def on_btn1_3_click(self):
        # get the input from ui item
        number = int(self.cboxImgNum.currentText())

         # termination criteria
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((8*11,3), np.float32)
        objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        # read images
        path = '../images/CameraCalibration/'+ str(number)+'.bmp'
        img = cv.imread(path)
        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        print(path)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (11,8),None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)
            ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

            # get rotation matrix and plus tranalation matrix
            R, jacobian = cv.Rodrigues(rvecs[0])
            extrinsic = np.hstack((R,tvecs[0]))
            print(extrinsic)


    def on_btn1_4_click(self):
        # termination criteria
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((8*11,3), np.float32)
        objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        #read all the images from folder
        images = glob.glob('../images/CameraCalibration/*.bmp')
        for fname in images:
            img = cv.imread(fname)
            gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv.findChessboardCorners(gray, (11,8),None)

            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)
                corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                imgpoints.append(corners2)
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
        print(dist)


    def on_btn2_1_click(self):
        # termination criteria
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((8*11,3), np.float32)
        objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        #read all the images from folder
        images = glob.glob('../images/CameraCalibration/*.bmp')

        for fname in images:
            img = cv.imread(fname)
            gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv.findChessboardCorners(gray, (11,8),None)

            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)

                corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                imgpoints.append(corners2)

        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

        # Function to draw the axis
        def draw(img, corners, imgpts):
            imgpts = np.int32(imgpts).reshape(-1,2)

            # draw ground floor in green
            img = cv.drawContours(img, [imgpts[:4]],-1,(0,0,255),10)

            # draw pillars in blue color
            for i,j in zip(range(4),range(4,8)):
                img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(0,0,255),10)

            # draw top layer in red color
            img = cv.drawContours(img, [imgpts[4:]],-1,(0,0,255),10)
            return img

        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((8*11,3), np.float32)
        objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)
        axis = np.float32([[0,0,0], [0,2,0], [2,2,0], [2,0,0],[0,0,-2],[0,2,-2],[2,2,-2],[2,0,-2] ])

        # declare a array to store video frame
        Video_img=[]
        for fname in glob.glob('../images/Augment/*.bmp'):
            img = cv.imread(fname)
            gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
            ret, corners = cv.findChessboardCorners(gray, (11,8),None)

            if ret == True:
                corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

                # Find the rotation and translation vectors.
                _,rvecs, tvecs, inliers = cv.solvePnPRansac(objp, corners2, mtx, dist)

                # project 3D points to image plane
                imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)

                img = draw(img,corners2,imgpts)
                Video_img.append(img)
                cv.imshow('Video',img)
                cv.waitKey(500)

        # making vidoe
        height,width,layers=Video_img[1].shape
        video=cv.VideoWriter('video.mp4',-1,2,(width,height))
        for j in range(0,5):
            video.write(Video_img[j])

        cv.destroyAllWindows()

    def on_btn3_1_click(self):
        # read the image
        img = cv.imread('../images/OriginalTransform.png')

        # read the transform data from ui
        edtAngle = float(self.edtAngle.text())
        edtScale = float(self.edtScale.text())
        edtTx = float(self.edtTx.text())
        edtTy = float(self.edtTy.text())

        # making translate matrix
        H = np.float32([[1,0,edtTx],[0,1,edtTy]])

        # translate the small squared image
        rows,cols = img.shape[:2]
        Translate_img = cv.warpAffine(img,H,(rows,cols))

        # making rotate and scale matrix
        rows,cols = Translate_img.shape[:2]
        M = cv.getRotationMatrix2D((130+edtTy,125+edtTy),edtAngle,edtScale)

        #rotate and scale the small squared image
        result = cv.warpAffine(Translate_img,M,(rows,cols))

        #show the result
        cv.imshow('Original Image', img)
        cv.imshow('Rotation + Translate + Scale Imag',result)

    def on_btn3_2_click(self):
        # declare 2 point array
        pts1=[]
        pts2 = np.float32([[20,20],[450,20],[450,450],[20,450]])

        def CallBack(event,x,y,flags,param):

            # if clicked doing the following things
            if event == cv.EVENT_LBUTTONDOWN:
                nonlocal pts1
                pts1.append([x,y])

                # if clicked th images for four times, then wraping the origin to the perspective image
                if len(pts1)==4:
                    pts1 = np.float32(pts1)
                    M = cv.getPerspectiveTransform(pts1,pts2)
                    dst = cv.warpPerspective(img,M,(450,450))
                    print(pts1)
                    cv.imshow('Perspective Result Image', dst)

        img = cv.imread('../images/OriginalPerspective.png')
        cv.namedWindow('origin')
        cv.imshow('origin',img)

        # add `setMouseCallback()` to the window
        cv.setMouseCallback('origin',CallBack)

    def on_btn4_1_click(self):
        # read left and right images
        imgL = cv.imread('../images/imL.png',0)
        imgR = cv.imread('../images/imR.png',0)

        # making disparity map
        stereo = cv.StereoSGBM_create(numDisparities=48, blockSize=3) #the third parameter
        disparity = stereo.compute(imgL,imgR)

        # normalization
        normalized_img = np.zeros((800, 800))
        normalized_img = cv.normalize(disparity, normalized_img, 0, 255, cv.NORM_MINMAX,cv.CV_8U)

        cv.imshow('Without L-R Disparity Check',normalized_img)


    def on_btn4_2_click(self):
        # read left and right images
        imgL = cv.imread('../images/imL.png',0)
        imgR = cv.imread('../images/imR.png',0)

        # making disparity map without checked
        stereo = cv.StereoSGBM_create(numDisparities=48, blockSize=3, disp12MaxDiff=0) #the third parameter
        disparity = stereo.compute(imgL,imgR)

        # making disparity map with checked
        stereo_checked = cv.StereoSGBM_create(numDisparities=48, blockSize=3, disp12MaxDiff=2)
        disparity_checked = stereo_checked.compute(imgL, imgR)

        # normalization
        normalized_img = np.zeros((800, 800))
        normalized_img = cv.normalize(disparity, normalized_img, 0, 255, cv.NORM_MINMAX,cv.CV_8U)
        cv.imshow('Without the left-right disparity check',normalized_img)

        normalized_checked = np.zeros((800, 800))
        normalized_checked = cv.normalize(disparity_checked, normalized_checked, 0, 255, cv.NORM_MINMAX,cv.CV_8U)
        cv.imshow('With the left-right disparity check',normalized_checked)

        # count the difference
        diff = cv.absdiff(normalized_img, normalized_checked)
        (x,y) = np.where(diff>0)

        diff_img = cv.cvtColor(normalized_checked,cv.COLOR_GRAY2RGB)

        for i in range(len(x)):
            diff_img[x[i],y[i]] = (0,0,255)
        cv.imshow('Mark The Diff',diff_img)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
