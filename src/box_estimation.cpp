#include <ros/ros.h>
// PCL specific includes
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/io/pcd_io.h>
#include <Eigen/Dense>
#include <pcl/registration/icp.h>
#include <iostream>
#include <cstring>

ros::Publisher pub;


void display_pointcloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, std::string ss);
pcl::visualization::PCLVisualizer::Ptr visualize_two_pcl(pcl::PointCloud<pcl::PointXYZ>::Ptr pcl1,pcl::PointCloud<pcl::PointXYZ>::Ptr pcl2);
void filter_pointcloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, int *rect);
void cloud_cb(const sensor_msgs::PointCloud2ConstPtr& input);
void get_dominant_plane(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr surface, float distance_thr);
void getIterativeClosestPoint(pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud, int *surface_count);



void getIterativeClosestPoint(pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud, int *surface_count){
  pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
  pcl::PointCloud<pcl::PointXYZ>::Ptr reference_cloud(new pcl::PointCloud<pcl::PointXYZ>);

  int pcl_size = input_cloud->points.size();
  float x_dim = 0.2;
  float y_dim = 0.075;
  float z_dim = 0.1;
  float scale = 10000.0;

  // float total_area = x_dim*y_dim + y_dim*z_dim + z_dim*x_dim;
  // int n1 = (pcl_size*x_dim*y_dim)/total_area;
  // int n2 = (pcl_size*y_dim*z_dim)/total_area;
  // int n3 = pcl_size - n1 - n2;

  int n1 = surface_count[0];
  int n2 = surface_count[1];
  int n3 = surface_count[2];

  float x_mean = 0;
  float y_mean = 0;
  float z_mean = 0;

  std::cout<<"Input Cloud size:"<<input_cloud->points.size() << std::endl;
  *reference_cloud = *input_cloud;

  for(int i = 0; i < pcl_size;i++){
    if(i < n1){
      reference_cloud->points[i].x = (rand()%(int)(x_dim*scale + 1) - x_dim*scale/2)/scale ;
      reference_cloud->points[i].y = (rand()%(int)(y_dim*scale + 1) - y_dim*scale/2)/scale ;
      reference_cloud->points[i].z = -z_dim/2;
    }
    else if(i >= n1 && i< (n1+n2)){
      reference_cloud->points[i].x = (rand()%(int)(x_dim*scale + 1) - x_dim*scale/2)/scale ;
      reference_cloud->points[i].y = -y_dim/2;
      reference_cloud->points[i].z = (rand()%(int)(z_dim*scale + 1) - z_dim*scale/2)/scale ;
    }
    else{
      reference_cloud->points[i].x = -x_dim/2;
      reference_cloud->points[i].y = (rand()%(int)(y_dim*scale + 1) - y_dim*scale/2)/scale ;
      reference_cloud->points[i].z = (rand()%(int)(z_dim*scale + 1) - z_dim*scale/2)/scale ;
    }
    x_mean += input_cloud->points[i].x;
    y_mean += input_cloud->points[i].y;
    z_mean += input_cloud->points[i].z;
  }

  x_mean /= pcl_size;
  y_mean /= pcl_size;
  z_mean /= pcl_size;

  Eigen::Matrix< float, 4, 4 > initial_guess;
  initial_guess(0,0) = 1.0 ; 
  initial_guess(1,1) = 1.0 ;
  initial_guess(2,2) = 1.0 ;
  initial_guess(3,3) = 1.0 ;
  initial_guess(0,1) = 0.0 ;
  initial_guess(0,2) = 0.0 ;
  initial_guess(1,0) = 0.0 ;
  initial_guess(1,2) = 0.0 ;
  initial_guess(2,0) = 0.0 ;
  initial_guess(2,1) = 0.0 ;
  initial_guess(3,0) = 0.0 ;
  initial_guess(3,1) = 0.0 ;
  initial_guess(3,2) = 0.0 ;
  initial_guess(0,3) = x_mean ;  
  initial_guess(1,3) = y_mean ;
  initial_guess(2,3) = z_mean ;

  display_pointcloud(reference_cloud, "reference cloud at origin");

  icp.setInputSource(reference_cloud);
  icp.setInputTarget(input_cloud);
  pcl::PointCloud<pcl::PointXYZ> aligned_pcl;
  icp.align(aligned_pcl, initial_guess);
  visualize_two_pcl(aligned_pcl.makeShared(), input_cloud);
  // display_pointcloud(aligned_pcl.makeShared());
  // display_pointcloud(input_cloud);
  std::cout << "has converged:" << icp.hasConverged() << " score: "<<icp.getFitnessScore() << std::endl;
  std::cout << icp.getFinalTransformation() << std::endl;

}

void display_pointcloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, std::string ss)
{
  pcl::visualization::CloudViewer viewer (ss);
  viewer.showCloud (cloud);
  while (!viewer.wasStopped ())
  {
  }
}

pcl::visualization::PCLVisualizer::Ptr visualize_two_pcl(pcl::PointCloud<pcl::PointXYZ>::Ptr pcl1,pcl::PointCloud<pcl::PointXYZ>::Ptr pcl2){
  pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("Refined pointcloud using ICP (green) and original pointcloud (red)"));
  // viewer->setBackgroundColor(0,0,0);
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> pcl1_single_color(pcl1,0,255,0);
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> pcl2_single_color(pcl1,255,0,0);

  viewer->addPointCloud<pcl::PointXYZ> (pcl1, pcl1_single_color,"reference_cloud");
  viewer->addPointCloud<pcl::PointXYZ> (pcl2, pcl2_single_color,"input_cloud");

  // viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
  // Eigen::Affine3f tt;
  // tt = Eigen::Translation3f(0.,0.,0.) * Eigen::AngleAxis<float>(M_PI, Eigen::Vector3f::UnitX());
  viewer->addCoordinateSystem (0.1);
  viewer->initCameraParameters ();
  while (!viewer->wasStopped ())
  {
    viewer->spinOnce(100);
    // std::this_thread::sleep_for(100ms);
  }
  return (viewer); 
}

void cloud_cb(const sensor_msgs::PointCloud2ConstPtr& input)
{
  // Create a container for the data.
  pcl::PCLPointCloud2* cloud2 = new pcl::PCLPointCloud2; 
  pcl_conversions::toPCL(*input, *cloud2);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromPCLPointCloud2(*cloud2,*cloud);
  // pcl::io::savePCDFileASCII("/home/parv/ws/src/box_estimation/data/scene.pcd", *cloud);  

  int rect[] = {246, 219, 439, 340};
  pcl::PointCloud<pcl::PointXYZ>::Ptr surface(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::copyPointCloud(*cloud,*surface);
  display_pointcloud(cloud, "captured scene point cloud");
  get_dominant_plane(cloud, surface, 0.01);
  // display_pointcloud(cloud);
  display_pointcloud(surface, "Ground Plane");
  filter_pointcloud(cloud,rect);
  display_pointcloud(cloud, "Points filtered using bounding box");
  pcl::PointCloud<pcl::PointXYZ>::Ptr cube_plane_1(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cube_plane_2(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cube_plane_3(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ> cube;
  get_dominant_plane(cloud, cube_plane_1, 0.01);
  get_dominant_plane(cloud, cube_plane_2, 0.01);
  get_dominant_plane(cloud, cube_plane_3, 0.01);
  cube += *cube_plane_1;
  cube += *cube_plane_2;
  cube += *cube_plane_3;
  display_pointcloud(cube.makeShared(), "Cube filtered with 3 dominant planes");
  std::cout << "ICP started " << rand() << std::endl;
  int surface_count[3];
  surface_count[0] = cube_plane_1->points.size();   
  surface_count[1] = cube_plane_2->points.size();
  surface_count[2] = cube_plane_3->points.size();

  getIterativeClosestPoint(cube.makeShared(), surface_count);
  // pcl::PCLPointCloud2ConstPtr cloudPtr(cloud);
  // // Do data processing here...
  // output = *input;

  // // Publish the data.
  // pub.publish (output);

  // std::cerr << "Point cloud data: " << cloud->points.size () << " points" << std::endl;
  // for (size_t i = 0; i < cloud2->data.size (); ++i)
  //   std::cerr << cloud2->data[i] << std::endl;
  //   std::cerr << "    " << cloud->points[i].x << " "
  //                       << cloud->points[i].y << " "
  //                       << cloud->points[i].z << std::endl;

}

void get_dominant_plane(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr surface, float distance_thr)
{
  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
  // Create the segmentation object
  pcl::SACSegmentation<pcl::PointXYZ> seg;
  // Optional
  seg.setOptimizeCoefficients (true);
  // Mandatory
  seg.setModelType (pcl::SACMODEL_PLANE);
  seg.setMethodType (pcl::SAC_RANSAC);
  // seg.setDistanceThreshold (0.01);
  seg.setDistanceThreshold (distance_thr);

  seg.setInputCloud (cloud);
  seg.segment (*inliers, *coefficients);

  pcl::ExtractIndices<pcl::PointXYZ> extract;
  extract.setInputCloud(cloud);
  extract.setIndices(inliers);
  extract.filter(*surface);
  extract.setNegative(true);
  extract.filter(*cloud);
}


void filter_pointcloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, int *rect)
{
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
  Eigen::Matrix3d K_;

  K_(0,0)=617.8404541;
  K_(0,1)=0.000000000;
  K_(0,2)=323.6599426;
  K_(1,0)=0.000000000;
  K_(1,1)=618.1534424;
  K_(1,2)=241.0394135;
  K_(2,0)=0.000000000;
  K_(2,1)=0.000000000;
  K_(2,2)=1.000000000;

  for (int i = 0; i < (*cloud).size(); i++)
  {
    float x,y,z,x_loc,y_loc;
    x = cloud->points[i].x;
    y = cloud->points[i].y;
    z = cloud->points[i].z;

    Eigen::Vector3d v(x,y,z);
    Eigen::Vector3d P;
    P = K_*v;
    x_loc = P(0)/P(2);
    y_loc = P(1)/P(2);
    
    if (x_loc>=rect[0] && x_loc<=rect[2] && y_loc>=rect[1] && y_loc<=rect[3])
      inliers->indices.push_back(i);
  }

  pcl::ExtractIndices<pcl::PointXYZ> extract;
  extract.setInputCloud(cloud);
  extract.setIndices(inliers);
  extract.filter(*cloud);
}



int main(int argc, char** argv)
{
  // Initialize ROS
  ros::init (argc, argv, "box_estimation");
  ros::NodeHandle nh;
  int rect[] = {246, 219, 439, 340};


  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);

  if (pcl::io::loadPCDFile<pcl::PointXYZ> ("/home/parv/ws/src/box_estimation/data/scene.pcd", *cloud) == -1) //* load the file
  {
    PCL_ERROR ("Couldn't read file /home/parv/ws/src/box_estimation/data/scene.pcd make sure path is updated. Trying live feed for /camera/depth/color/points \n");
    // return (-1);
    // Create a ROS subscriber for the input point cloud
    ros::Subscriber sub = nh.subscribe ("/camera/depth/color/points", 1, cloud_cb);

    // Spin
    ros::spin();  

    return 0;
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr surface(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::copyPointCloud(*cloud,*surface);
  display_pointcloud(cloud, "captured scene point cloud");
  get_dominant_plane(cloud, surface, 0.01);
  // display_pointcloud(cloud);
  display_pointcloud(surface, "Ground Plane");
  filter_pointcloud(cloud,rect);
  display_pointcloud(cloud, "Points filtered using bounding box");
  pcl::PointCloud<pcl::PointXYZ>::Ptr cube_plane_1(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cube_plane_2(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cube_plane_3(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ> cube;
  get_dominant_plane(cloud, cube_plane_1, 0.01);
  get_dominant_plane(cloud, cube_plane_2, 0.01);
  get_dominant_plane(cloud, cube_plane_3, 0.01);
  cube += *cube_plane_1;
  cube += *cube_plane_2;
  cube += *cube_plane_3;
  display_pointcloud(cube.makeShared(), "Cube filtered with 3 dominant planes");
  std::cout << "ICP started " << rand() << std::endl;
  int surface_count[3];
  surface_count[0] = cube_plane_1->points.size();   
  surface_count[1] = cube_plane_2->points.size();
  surface_count[2] = cube_plane_3->points.size();

  getIterativeClosestPoint(cube.makeShared(), surface_count);
  // Create a ROS publisher for the output point cloud
  // pub = nh.advertise<sensor_msgs::PointCloud2> ("output", 1);
  return 0;
}
