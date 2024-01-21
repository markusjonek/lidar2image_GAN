#include <boost/program_options.hpp>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>

// include pcl packages for voxel
#include <pcl/filters/voxel_grid.h>

#include <pcl/common/io.h>

#include <iostream>
#include <fstream>
#include <filesystem>


namespace fs = std::filesystem;
namespace po = boost::program_options;


void convertBinsToPCD(std::vector<std::string> &files, std::string &out_dir){
    // Create the output directory if it doesn't exist
    if (!fs::exists(out_dir)){
        fs::create_directory(out_dir);
    }

    // Iterate through the files
    for (auto &file : files){
        // Create the output file name
        std::string in_file_name = file.substr(file.find_last_of("/") + 1);
        std::string out_file = out_dir + "/" + in_file_name.substr(0, in_file_name.find_last_of(".")) + ".pcd";

        std::cout << "Converting " << file << " to " << out_file << std::endl;

        // Create the point cloud
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);

        // Open the file
        std::ifstream in_file(file, std::ios::binary);

        // Read the file
        for (int i = 0; in_file.good() && !in_file.eof(); i++){
            pcl::PointXYZI point;
            in_file.read((char *) &point.x, 3 * sizeof(float));
            in_file.read((char *) &point.intensity, sizeof(float));
            cloud->push_back(point);
        }

        // Close the file
        in_file.close();
        
        int points_before_voxel = cloud->size();

        pcl::VoxelGrid<pcl::PointXYZI> voxel_grid;
        voxel_grid.setInputCloud(cloud);
        voxel_grid.setLeafSize(0.3f, 0.3f, 0.3f); 
        voxel_grid.filter(*cloud);

        int points_after_voxel = cloud->size();

        std::cout << "Before: " << points_before_voxel << " - After: " << points_after_voxel << std::endl;

        // Save the point cloud
        pcl::io::savePCDFileASCII(out_file, *cloud);
    }
}


int main(int argc, char **argv){
	///The file to read from.
	std::string in_dir;

	///The file to output to.
	std::string out_dir;

	// Declare the supported options.
	po::options_description desc("Program options");
	desc.add_options()
		//Options
		("indir", po::value<std::string>(&in_dir)->required(), "the file to read a point cloud from")
		("outdir", po::value<std::string>(&out_dir)->required(), "the file to write the DoN point cloud & normals to")
		;
	// Parse the command line
	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);

	// Print help
	if (vm.count("help"))
	{
		std::cout << desc << std::endl;
		return 1;
	}

	// Process options.
	po::notify(vm);


    // Get the files in the directory
    std::vector<std::string> files;
    for (const auto &entry : fs::directory_iterator(in_dir)){
        files.push_back(entry.path().string());
    }

    // Convert the files
    convertBinsToPCD(files, out_dir);

    return 0;

}