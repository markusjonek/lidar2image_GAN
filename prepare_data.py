import numpy
import torch
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pypcd import pypcd
import cv2
import time
import os


def get_x_y_pair(pcd_path, img_path, rotation_matrix, intrinsic_matrix):
    print(pcd_path)
    pcd = pypcd.PointCloud.from_path(pcd_path)

    xyz = np.empty((pcd.points, 3), dtype=np.float32)
    xyz[:, 0] = pcd.pc_data['x']
    xyz[:, 1] = pcd.pc_data['y']
    xyz[:, 2] = pcd.pc_data['z']
    intensity = pcd.pc_data['intensity']

    distances = np.sqrt(np.sum(xyz ** 2, axis=1))

    intensity = intensity[xyz[:, 0] > 0]
    distances = distances[xyz[:, 0] > 0]
    xyz = xyz[xyz[:, 0] > 0]

    # Project on image
    img = cv2.imread(img_path)    
    rodrigues, _ = cv2.Rodrigues(rotation_matrix)
    rvec = np.array(rodrigues, dtype=np.float32)
    tvec = np.array([0, 0, 0], dtype=np.float32)
    img_points, _ = cv2.projectPoints(xyz, rvec, tvec, intrinsic_matrix, None)

    # filter out points outside of image
    points_on_img = []
    points_on_img_bgr = []

    for i, img_point in enumerate(img_points):
        pc_x, pc_y, pc_z = xyz[i]
        inten = intensity[i]
        img_x, img_y = img_point[0]
        if 0 <= img_x < img.shape[1] and 0 <= img_y < img.shape[0]:
            r, g, b = img[int(img_y), int(img_x)]
            points_on_img.append([pc_x, pc_y, pc_z, inten])
            points_on_img_bgr.append([pc_x, pc_y, pc_z, r, g, b])

    points_on_img = np.array(points_on_img)
    points_on_img_bgr = np.array(points_on_img_bgr)

    return points_on_img, points_on_img_bgr


def main():
    """
    Colorize point cloud with image.
    :param pc_XYZI: point cloud in XYZI format
    :param img: image
    :param intrinsic_matrix: intrinsic matrix
    :param rotation_matrix: rotation matrix
    :return: colorized point cloud
    """
    
    rotation_mat = np.array(
        [
            [0, -1, 0],
            [0, 0, -1],
            [1, 0,  0]
        ], 
        dtype=np.float32
    )

    fx = 718.856
    fy = 718.856
    cx = 607.1928
    cy = 185.2157

    intrinsic_mat = np.array(
        [
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ]
    )

    img_width = 1242
    img_height = 375

    base_dir = 'data/2011_09_26_drive_0002_sync'

    x_save_dir = 'uncolored_point_clouds'
    os.makedirs(x_save_dir, exist_ok=True)

    y_save_dir = 'colored_point_clouds'
    os.makedirs(y_save_dir, exist_ok=True)

    count = 0

    velodyne_points_dir = 'pcds'
    image_02_dir = os.path.join(base_dir, 'image_02', 'data')

    velodyne_points_files = os.listdir(velodyne_points_dir)
    image_02_files = os.listdir(image_02_dir)

    velodyne_points_files.sort()
    image_02_files.sort()

    for velodyne_points_file, image_02_file in zip(velodyne_points_files, image_02_files):
        velodyne_points_path = os.path.join(velodyne_points_dir, velodyne_points_file)
        image_02_path = os.path.join(image_02_dir, image_02_file)

        points_on_img, points_on_img_bgr = get_x_y_pair(velodyne_points_path, image_02_path, rotation_mat, intrinsic_mat)

        instance_name = velodyne_points_file.split('.')[0]
        npy_file = instance_name + '.npy'

        # save point cloud
        save_path = os.path.join(x_save_dir, npy_file)
        np.save(save_path, points_on_img)

        # save point cloud with color
        save_path = os.path.join(y_save_dir, npy_file)
        np.save(save_path, points_on_img_bgr)

        count += 1
        print(count)
        


def main_full_dataset():
    """
    Colorize point cloud with image.
    :param pc_XYZI: point cloud in XYZI format
    :param img: image
    :param intrinsic_matrix: intrinsic matrix
    :param rotation_matrix: rotation matrix
    :return: colorized point cloud
    """
    
    rotation_mat = np.array(
        [
            [0, -1, 0],
            [0, 0, -1],
            [1, 0,  0]
        ], 
        dtype=np.float32
    )

    fx = 718.856
    fy = 718.856
    cx = 607.1928
    cy = 185.2157

    intrinsic_mat = np.array(
        [
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ]
    )

    img_width = 1242
    img_height = 375

    base_dir = '/lidar_img/raw_data'
    date_dirs = os.listdir(base_dir)

    x_save_dir = 'uncolored_point_clouds'
    os.makedirs(x_save_dir, exist_ok=True)

    y_save_dir = 'colored_point_clouds'
    os.makedirs(y_save_dir, exist_ok=True)

    count = 0

    for date_dir in date_dirs:
        instance_dirs = os.listdir(os.path.join(base_dir, date_dir))
        for instance_dir in instance_dirs:
            velodyne_points_dir = os.path.join(base_dir, date_dir, instance_dir, 'velodyne_points_pcd')
            image_02_dir = os.path.join(base_dir, date_dir, instance_dir, 'image_02', 'data')

            velodyne_points_files = os.listdir(velodyne_points_dir)
            image_02_files = os.listdir(image_02_dir)

            velodyne_points_files.sort()
            image_02_files.sort()

            for velodyne_points_file, image_02_file in zip(velodyne_points_files, image_02_files):

                velodyne_points_path = os.path.join(velodyne_points_dir, velodyne_points_file)
                image_02_path = os.path.join(image_02_dir, image_02_file)

                points_on_img, points_on_img_bgr = get_x_y_pair(velodyne_points_path, image_02_path, rotation_mat, intrinsic_mat)

                instance_name = velodyne_points_file.split('.')[0]
                npy_file = instance_dir + '_' + instance_name + '.npy'

                # save point cloud
                save_path = os.path.join(x_save_dir, npy_file)
                np.save(save_path, points_on_img)

                # save point cloud with color
                save_path = os.path.join(y_save_dir, npy_file)
                np.save(save_path, points_on_img_bgr)

                count += 1
                print(count)




        
if __name__ == '__main__':
    main_full_dataset()