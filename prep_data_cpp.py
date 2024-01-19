from build.prepare_data import getXYpair
import numpy as np
import os
import time

def get_total_num_files(base_dir):
    date_dirs = os.listdir(base_dir)

    total_num_files = 0

    for date_dir in date_dirs:
        instance_dirs = os.listdir(os.path.join(base_dir, date_dir))
        for instance_dir in instance_dirs:
            velodyne_points_dir = os.path.join(base_dir, date_dir, instance_dir, 'velodyne_points_pcd')
            velodyne_points_files = os.listdir(velodyne_points_dir)
            total_num_files += len(velodyne_points_files)

    return total_num_files


def main():
    base_dir = '/lidar_img/raw_data'
    date_dirs = os.listdir(base_dir)

    x_save_dir = 'uncolored_point_clouds'
    os.makedirs(x_save_dir, exist_ok=True)

    y_save_dir = 'colored_point_clouds'
    os.makedirs(y_save_dir, exist_ok=True)

    total_num_files = get_total_num_files(base_dir)
    count = 0

    X = []
    Y = []

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

                points_on_img_bgr, points_on_img = getXYpair(velodyne_points_path, image_02_path)

                instance_name = velodyne_points_file.split('.')[0]
                npy_file = instance_dir + '_' + instance_name + '.npy'

                # save point cloud
                X.append(points_on_img)
                Y.append(points_on_img_bgr)
                
                count += 1
                print(f'{count} / {total_num_files}')

            np.savez('X_data.npz', **{f'{i}': a for i, a in enumerate(X)})
            np.savez('Y_data.npz', **{f'{i}': a for i, a in enumerate(Y)})

if __name__ == '__main__':
    main()