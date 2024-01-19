import os
import sys


def main():
    base_dir = '/lidar_img/raw_data'
    date_dirs = os.listdir(base_dir)

    for date_dir in date_dirs:
        instance_dirs = os.listdir(os.path.join(base_dir, date_dir))
        for instance_dir in instance_dirs:
            # run bash command
            velo_dir = os.path.join(base_dir, date_dir, instance_dir, 'velodyne_points', 'data')
            pcd_dir = os.path.join(base_dir, date_dir, instance_dir, 'pcds')

            cmd = f'/lidar_img/build/read_pointcloud --indir {velo_dir} --outdir {pcd_dir}'
            os.system(cmd)
            print(cmd)


if __name__ == '__main__':
    main()