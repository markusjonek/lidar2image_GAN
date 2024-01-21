#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/opencv.hpp>

#include <boost/program_options.hpp>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>

// include pcl packages for voxel
#include <pcl/filters/voxel_grid.h>

#include <pcl/common/io.h>

#include <iostream>
#include <fstream> 
#include <filesystem>
#include <string>
#include <vector>
#include <algorithm>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

typedef pcl::PointCloud<pcl::PointXYZI> PointCloudXYZI;
typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloudXYZRGB;

namespace py = pybind11;

typedef std::pair<py::array_t<float>, py::array_t<float>> PyArrayPair;


void getXYpair(std::string& pcd_path, std::string& img_path, PointCloudXYZRGB::Ptr& cloud_rgb, PointCloudXYZI::Ptr& cloud_i) {
    // read pcd file
    PointCloudXYZI::Ptr cloud(new PointCloudXYZI);
    pcl::io::loadPCDFile<pcl::PointXYZI>(pcd_path, *cloud);

    // read image file
    cv::Mat img = cv::imread(img_path, cv::IMREAD_COLOR);

    std::vector<cv::Point3f> cv_cloud;
    for (const pcl::PointXYZI& point : cloud->points) {
        cv_cloud.push_back(cv::Point3f(point.x, point.y, point.z));
    }

    cv::Mat intrinsic_mat = (cv::Mat_<float>(3, 3) << 718.856, 0, 607.1928, 0, 718.856, 185.2157, 0, 0, 1);

    // Rotation matrix and translation vector
    cv::Mat rotation_mat = (cv::Mat_<float>(3, 3) << 0, -1, 0, 0, 0, -1, 1, 0, 0);
    cv::Mat translation_vec = cv::Mat::zeros(3, 1, CV_32F); // Translation is zero

    // Convert rotation matrix to rotation vector
    cv::Mat rotation_vec;
    cv::Rodrigues(rotation_mat, rotation_vec);

    // Project points
    std::vector<cv::Point2f> projected_cloud;
    cv::projectPoints(cv_cloud, rotation_vec, translation_vec, intrinsic_mat, cv::Mat(), projected_cloud);

    cloud_rgb->clear();
    cloud_i->clear();

    for (int i = 0; i < projected_cloud.size(); i++) {
        int x = projected_cloud[i].x;
        int y = projected_cloud[i].y;
        float intensity = cloud->points[i].intensity;

        if (cloud->points[i].x <= 1) {
            continue;
        }

        if (x >= 0 && x < img.cols && y >= 0 && y < img.rows) {
            pcl::PointXYZRGB point;
            point.x = cv_cloud[i].x;
            point.y = cv_cloud[i].y;
            point.z = cv_cloud[i].z;
            point.r = img.at<cv::Vec3b>(y, x)[2];
            point.g = img.at<cv::Vec3b>(y, x)[1];
            point.b = img.at<cv::Vec3b>(y, x)[0];
            cloud_rgb->points.push_back(point);

            pcl::PointXYZI point_i;
            point_i.x = cv_cloud[i].x;
            point_i.y = cv_cloud[i].y;
            point_i.z = cv_cloud[i].z;
            point_i.intensity = intensity;
            cloud_i->points.push_back(point_i);
        }
    }
}


PyArrayPair getXYpair_python(std::string& pcd_path, std::string& img_path) {
    PointCloudXYZRGB::Ptr cloud_rgb(new PointCloudXYZRGB);
    PointCloudXYZI::Ptr cloud_i(new PointCloudXYZI);

    getXYpair(pcd_path, img_path, cloud_rgb, cloud_i);

    int num_points = cloud_rgb->size();

    // convert to numpy array
    py::array_t<float> cloud_rgb_np({num_points, 6});
    py::array_t<float> cloud_i_np({num_points, 4});

    std::fill(cloud_rgb_np.mutable_data(), cloud_rgb_np.mutable_data() + cloud_rgb_np.size(), 0.0f);
    std::fill(cloud_i_np.mutable_data(), cloud_i_np.mutable_data() + cloud_i_np.size(), 0.0f);

    auto cloud_rgb_np_ptr = cloud_rgb_np.mutable_unchecked<2>();
    auto cloud_i_np_ptr = cloud_i_np.mutable_unchecked<2>();

    for (int i = 0; i < num_points; i++) {
        cloud_rgb_np_ptr(i, 0) = cloud_rgb->points[i].x;
        cloud_rgb_np_ptr(i, 1) = cloud_rgb->points[i].y;
        cloud_rgb_np_ptr(i, 2) = cloud_rgb->points[i].z;
        cloud_rgb_np_ptr(i, 3) = cloud_rgb->points[i].r;
        cloud_rgb_np_ptr(i, 4) = cloud_rgb->points[i].g;
        cloud_rgb_np_ptr(i, 5) = cloud_rgb->points[i].b;

        cloud_i_np_ptr(i, 0) = cloud_i->points[i].x;
        cloud_i_np_ptr(i, 1) = cloud_i->points[i].y;
        cloud_i_np_ptr(i, 2) = cloud_i->points[i].z;
        cloud_i_np_ptr(i, 3) = cloud_i->points[i].intensity;
    }

    return std::make_pair(cloud_rgb_np, cloud_i_np);
}


PYBIND11_MODULE(prepare_data, m) {
    m.def("getXYpair", &getXYpair_python, "getXYpair");
}