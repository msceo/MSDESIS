import cv2
import numpy as np
import open3d as o3d

def DispVisualizer(disparity, calib_path):
    # Load calibration data 
    calib_data = cv2.FileStorage(calib_path, cv2.FILE_STORAGE_READ)
    if len(disparity.shape) == 4:
        disparity = disparity[0, 0].astype(np.float32)
    else:
        disparity = disparity.astype(np.float32)
    T = calib_data.getNode('T').mat()
    K = calib_data.getNode('K1').mat()
    if K is None:
        K = calib_data.getNode('M1').mat()

    # Camera Intrinsics
    fx = K[0][0]
    fy = K[1][1]
    cx = K[0][2]
    cy = K[1][2]
    baseline = -T[0][0] / fx
    focal_length = (fx + fy) / 2
    
    # Generate pointcloud
    depth = np.zeros(disparity.shape, dtype=np.float32)
    depth[disparity > 0] = baseline* focal_length / disparity[disparity > 0]
    intrinsics = o3d.camera.PinholeCameraIntrinsic(width=disparity.shape[1], height=disparity.shape[0], fx=fx, fy=fy, cx=cx, cy=cy)
    pcd = o3d.geometry.PointCloud.create_from_depth_image(o3d.geometry.Image(depth), intrinsics)

    # Visualize and save
    o3d.visualization.draw_geometries([pcd])
    o3d.io.write_point_cloud('segmentation.ply', pcd)

def SegVisualizer(segmentation, calib_path=None):
    with open(calib_path, 'r') as file:
        calib = file.readlines()

    h, w = segmentation.shape
    fx, fy = calib[2].split()[1:3]
    cx, cy = calib[2].split()[1:3]
    fx = float(fx)
    fy = float(fy)
    cx = float(cx)
    cy = float(cy)

    points = []
    for x in range(w):
        for y in range(h):
            if segmentation[y, x] == 255:
                z = fx
                x_ = (x - cx) * z / fx
                y_ = (y - cy) * z / fy
                points.append([x_, y_, z])
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])
    # Save output
    o3d.io.write_point_cloud('segmentation.ply', pcd)


if __name__ == '__main__':
    disp = cv2.imread('outputs/dataset_1/keyframe_1/frame000000.png', cv2.IMREAD_GRAYSCALE)
    DispVisualizer(disp, '/home/ilsr/scared_toolkit/output/dataset_1/keyframe_1/stereo_calib.json')
    # seg = cv2.imread('/home/ilsr/msdesis/outputs/instrument_dataset_10/binary/frame000.png', cv2.IMREAD_GRAYSCALE)
    # SegVisualizer(seg, '/home/ilsr/ris2017_toolkit/RIS_data/test_set/instrument_dataset_10/camera_calibration.txt')
