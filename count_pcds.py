import os

total_files = 0

base_dir = '/lidar_img/raw_data'
date_dirs = os.listdir(base_dir)

for date_dir in date_dirs:
    instance_dirs = os.listdir(os.path.join(base_dir, date_dir))

    for instance_dir in instance_dirs:
        pcd_files = os.listdir(os.path.join(base_dir, date_dir, instance_dir, 'velodyne_points_pcd'))
        total_files += len(pcd_files)

print(total_files)
