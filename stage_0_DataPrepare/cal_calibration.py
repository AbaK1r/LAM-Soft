from pathlib import Path

import cv2
import numpy as np

CHECKERBOARD = (10, 15)
subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
_img_shape = None
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.
images = [str(i) for i in Path('datasets/C3VD/cfhq190l_10x10mm_checkerboard_images/').glob('*.tiff')]
for fname in images:
    img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
    if _img_shape is None:
        _img_shape = img.shape[:2]
        print(f"Image shape: {_img_shape}")
    else:
        assert _img_shape == img.shape[:2], "All images must share the same size."
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                             cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    # If found, add object points, image points (after refining them)
    if ret:
        objpoints.append(objp)
        cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), subpix_criteria)
        imgpoints.append(corners)

N_OK = len(objpoints)
K = np.zeros((3, 3))
D = np.zeros((4, 1))
rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
rms, _, _, _, _ = \
    cv2.fisheye.calibrate(
        objpoints,
        imgpoints,
        gray.shape[::-1],
        K,
        D,
        rvecs,
        tvecs,
        calibration_flags,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    )
print(f"image shape: {_img_shape}")
print("Found " + str(N_OK) + " valid images for calibration")
print("DIM=" + str(_img_shape[::-1]))
print("K=np.array(" + str(K.tolist()) + ")")
print("D=np.array(" + str(D.tolist()) + ")")
print(D)
np.savez("datasets/C3VD/C3VD_calibration.npz", K=K, D=D)
# K=np.array([[774.7611890373379, 0.0, 678.3142750557644], [0.0, 774.5409935830137, 544.3831888773642], [0.0, 0.0, 1.0]])
# D=np.array([[-0.20259135467873074], [0.020961026919539487], [0.0006368736404869556], [0.0007208005955989016]])
# [[-0.20259135]
#  [ 0.02096103]
#  [ 0.00063687]
#  [ 0.0007208 ]]