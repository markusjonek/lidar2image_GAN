import os
import sys


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
    base_dir = '/lidar_img/raw_data'
    date_dirs = os.listdir(base_dir)

    save_dir = '/lidar_img/processed_data'
    os.makedirs(save_dir, exist_ok=True)

    tot = get_total_files(base_dir)
    count = 0

    for date_dir in date_dirs:
        instance_dirs = os.listdir(os.path.join(base_dir, date_dir))

        for instance_dir in instance_dirs:
            # check if pcd files already exist
            if os.path.exists(os.path.join(base_dir, date_dir, instance_dir, 'velodyne_points_pcd')):
                continue

            indir = os.path.join(base_dir, date_dir, instance_dir, 'velodyne_points', 'data')
            outdir = os.path.join(base_dir, date_dir, instance_dir, 'velodyne_points_pcd')

            os.makedirs(outdir, exist_ok=True)

            os.system(f'/lidar_img/build/read_pointcloud --indir {indir} --outdir {outdir}')

            count += 1
            print(f'Processed {count}/{tot} files')


if __name__ == "__main__":
    main()