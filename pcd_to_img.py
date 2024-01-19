import numpy as np
import matplotlib.pyplot as plt
from pypcd import pypcd
import cv2
import time
import os
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import griddata


def pcd_to_img(pcd_path, img_width, img_height, rotation_matrix, intrinsic_matrix):
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

    distances = np.log(np.clip(distances, 0, 75))

    rodrigues, _ = cv2.Rodrigues(rotation_matrix)

    rvec = np.array(rodrigues, dtype=np.float32)
    tvec = np.array([0, 0, 0], dtype=np.float32)

    distortion = np.array([0, 0, 0, 0, 0], dtype=np.float32)

    points, _ = cv2.projectPoints(xyz, rvec, tvec, intrinsic_matrix, distortion)

    xy = points.reshape(-1, 2)
    z = intensity.reshape(-1)
    z = distances.reshape(-1)

    grid_x, grid_y = np.mgrid[0:img_width, 150:img_height]
    grid_z = griddata(xy, z, (grid_x, grid_y), method='linear')

    img = grid_z.transpose()

    grid_x, grid_y = np.mgrid[0:img_width, 150:img_height]
    grid_z = griddata(xy, intensity.reshape(-1), (grid_x, grid_y), method='linear')

    img += grid_z.transpose() * 1.7

    img = img / np.max(img)

    return img


def get_total_files(base_dir):
    total_files = 0
    date_dirs = os.listdir(base_dir)

    for date_dir in date_dirs:
        instance_dirs = os.listdir(os.path.join(base_dir, date_dir))

        for instance_dir in instance_dirs:
            pcd_files = os.listdir(os.path.join(base_dir, date_dir, instance_dir, 'velodyne_points', 'data'))
            total_files += len(pcd_files)

    return total_files


def main():
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

    base_dir = 'data'
    date_dirs = os.listdir(base_dir)

    save_dir = 'processed_data'
    os.makedirs(save_dir, exist_ok=True)

    tot = get_total_files(base_dir)
    count = 0

    for date_dir in date_dirs:
        instance_dirs = os.listdir(os.path.join(base_dir, date_dir))

        for instance_dir in instance_dirs:
            pcd_files = os.listdir(os.path.join(base_dir, date_dir, instance_dir, 'velodyne_points', 'data'))

            for pcd_file in pcd_files:
                pcd_path = os.path.join(base_dir, date_dir, instance_dir, pcd_file)
                img = pcd_to_img(pcd_path, img_width, img_height, rotation_mat, intrinsic_mat)

                save_path = os.path.join(SAVE_DIR, date_dir, instance_dir)
                os.makedirs(save_path, exist_ok=True)

                img_name = pcd_file.split('.')[0] + '.png'
                
                cv2.imshow('img', img)

                #cv2.imwrite(os.path.join(save_path, img_name), img)

                count += 1
                print(f'Processed {count}/{tot} files')

                if count == 10:
                    exit()


            
            


if __name__ == "__main__":
    main()