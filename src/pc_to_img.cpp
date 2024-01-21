#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <string>
#include <filesystem>

void pcdToImg(const std::string& pcd_path, int img_width, int img_height, 
              const cv::Mat& rotation_matrix, const cv::Mat& intrinsic_matrix, 
              const std::string& save_path) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);

    if (pcl::io::loadPCDFile<pcl::PointXYZI>(pcd_path, *cloud) == -1) {
        PCL_ERROR("Couldn't read file\n");
        return;
    }

    std::vector<cv::Point3f> points;
    points.reserve(cloud->points.size());

    for (const auto& point : cloud->points) {
        if (point.x > 0) {
            points.emplace_back(point.x, point.y, point.z);
        }
    }

    cv::Mat rvec;
    cv::Rodrigues(rotation_matrix, rvec);

    cv::Mat tvec = cv::Mat::zeros(3, 1, CV_32F);
    cv::Mat distortion = cv::Mat::zeros(1, 5, CV_32F);

    std::vector<cv::Point2f> image_points;
    cv::projectPoints(points, rvec, tvec, intrinsic_matrix, distortion, image_points);

    cv::Mat img = cv::Mat::zeros(img_height, img_width, CV_32FC1);

    for (size_t i = 0; i < image_points.size(); ++i) {
        int x = static_cast<int>(image_points[i].x);
        int y = static_cast<int>(image_points[i].y);
        if (x >= 0 && x < img_width && y >= 0 && y < img_height) {
            float intensity = cloud->points[i].intensity;
            cv::circle(img, cv::Point(x, y), 1, cv::Scalar(intensity * 255, 0, 0), -1);
        }
    }

    cv::imwrite(save_path, img);
}

int main() {
    std::string pcd_dir = "/Users/markusjonek/Documents/KTH/lidar2image_GAN/data/pcds/";
    std::string save_dir = "/Users/markusjonek/Documents/KTH/lidar2image_GAN/data/pc_images/";

    cv::Mat rotation_mat = (cv::Mat_<float>(3,3) << 0, -1, 0, 0, 0, -1, 1, 0, 0);
    float fx = 718.856f, fy = 718.856f, cx = 607.1928f, cy = 185.2157f;
    cv::Mat intrinsic_mat = (cv::Mat_<float>(3,3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);

    int img_width = 1242;
    int img_height = 375;

    for (const auto& entry : std::filesystem::directory_iterator(pcd_dir)) {
        if (entry.path().extension() == ".pcd") {
            std::string pcd_path = entry.path().string();
            std::string save_path = save_dir + entry.path().stem().string() + ".png";
            pcdToImg(pcd_path, img_width, img_height, rotation_mat, intrinsic_mat, save_path);
        }
    }

    return 0;
}
