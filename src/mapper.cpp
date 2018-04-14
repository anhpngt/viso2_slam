#include <iostream>
#include <string>
#include <vector>

#include <ros/ros.h>
#include <tf/transform_datatypes.h>

#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/PointCloud2.h>
#include <viso2_slam/BoolStamped.h>

#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl_ros/transforms.h>
#include <pcl_conversions/pcl_conversions.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

using namespace std;

namespace viso2_slam
{
class Mapper
{
public:
  typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloud_t;

private:

  // Initialization variables
  message_filters::Subscriber<sensor_msgs::PointCloud2> pointcloud_sub_;
  message_filters::Subscriber<nav_msgs::Path> trajectory_sub_;
  message_filters::Subscriber<viso2_slam::BoolStamped> optimization_sub_;
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, nav_msgs::Path, viso2_slam::BoolStamped> newSyncPolicy;
  typedef message_filters::Synchronizer<newSyncPolicy> newSync;
  boost::shared_ptr<newSync> new_sync_;

  ros::Publisher pointcloud_pub_;

  int queue_size_;
  std::string odom_frame_id_;
  std::string viso2_pointcloud_topic_, viso2_trajectory_topic_, viso2_opt_topic_;

  Eigen::Affine3d tf_world2cam;

  // Data
  PointCloud_t cloud_map_;
  std::vector<PointCloud_t> cloud_vector_;

public:

  Mapper()
  {
    // Parameters
    ros::NodeHandle nh;
    ros::NodeHandle local_nh("~");
    local_nh.param("odom_frame_id", odom_frame_id_, std::string("/odom"));
    local_nh.param("pointcloud_topic", viso2_pointcloud_topic_, std::string("/stereo_slam/point_cloud"));
    local_nh.param("trajectory_topic", viso2_trajectory_topic_, std::string("/pose_optimizer/trajectory"));
    local_nh.param("optimization_topic", viso2_opt_topic_, std::string("/pose_optimizer/optimization"));
    local_nh.param("queue_size", queue_size_, 20);
    
    // Set up sync callback for viso2 odom
    pointcloud_sub_.subscribe(nh, viso2_pointcloud_topic_, 10);
    trajectory_sub_.subscribe(nh, viso2_trajectory_topic_, 10);
    optimization_sub_.subscribe(nh, viso2_opt_topic_, 10);
    new_sync_.reset(new newSync(newSyncPolicy(queue_size_), pointcloud_sub_, trajectory_sub_, optimization_sub_));
    new_sync_->registerCallback(boost::bind(&Mapper::mapperCallback, this, _1, _2, _3));

    // Result publisher
    pointcloud_pub_ = local_nh.advertise<sensor_msgs::PointCloud2>("map", 1);

    pcl::getTransformation(0, 0, 0, -1.57079632679, 0, -1.57079632679, tf_world2cam);
   
    ROS_INFO("Mapper initialization finished! Waiting for incoming topics...");
  }

private:
  void mapperCallback(const sensor_msgs::PointCloud2::ConstPtr &pointcloud_msg,
                      const nav_msgs::Path::ConstPtr &trajectory_msg,
                      const viso2_slam::BoolStamped::ConstPtr &optimization_msg)
  {
    PointCloud_t new_cloud;
    pcl::fromROSMsg(*pointcloud_msg, new_cloud);
    cloud_vector_.push_back(new_cloud);

    if(!optimization_msg->data)
    {
      // No bundle adjustment was performed, simply add cloud to map
      double tf_x, tf_y, tf_z, tf_rl, tf_pt, tf_yw;
      PointCloud_t transformed_cloud;
      Eigen::Affine3d latest_transform;
      geometry_msgs::PoseStamped latest_pose = trajectory_msg->poses.back();
      tf_x = latest_pose.pose.position.x;
      tf_y = latest_pose.pose.position.y;
      tf_z = latest_pose.pose.position.z;
      tf::Quaternion tmp_quat(latest_pose.pose.orientation.x, latest_pose.pose.orientation.y,
                              latest_pose.pose.orientation.z, latest_pose.pose.orientation.w);
      tf::Matrix3x3(tmp_quat).getRPY(tf_rl, tf_pt, tf_yw);
      pcl::getTransformation(tf_x, tf_y, tf_z, tf_rl, tf_pt, tf_yw, latest_transform);
      pcl::transformPointCloud(new_cloud, transformed_cloud, latest_transform * tf_world2cam);
      cloud_map_ += transformed_cloud;
    }
    else
    {
      // Bundle adjustment was performed, re-transform all clouds
      ROS_ASSERT(cloud_vector_.size() == trajectory_msg->poses.size());

      cloud_map_.clear();
      for(size_t i = 0, i_end = cloud_vector_.size(); i < i_end; ++i)
      {
        PointCloud_t transformed_cloud;
        double tf_x, tf_y, tf_z, tf_rl, tf_pt, tf_yw;
        Eigen::Affine3d transform;
        geometry_msgs::PoseStamped pose = trajectory_msg->poses[i];
        tf_x = pose.pose.position.x;
        tf_y = pose.pose.position.y;
        tf_z = pose.pose.position.z;
        tf::Quaternion tmp_quat(pose.pose.orientation.x, pose.pose.orientation.y,
                                pose.pose.orientation.z, pose.pose.orientation.w);
        tf::Matrix3x3(tmp_quat).getRPY(tf_rl, tf_pt, tf_yw);
        pcl::getTransformation(tf_x, tf_y, tf_z, tf_rl, tf_pt, tf_yw, transform);
        pcl::transformPointCloud(cloud_vector_[i], transformed_cloud, transform * tf_world2cam);
        cloud_map_ += transformed_cloud;
      }
      // ROS_INFO("Map succesfully updated");
    }

    if(pointcloud_pub_.getNumSubscribers() > 0)
      publishCloudMap(pointcloud_msg->header);
  }

  void publishCloudMap(std_msgs::Header header)
  {
    sensor_msgs::PointCloud2 map_msg;
    pcl::toROSMsg(cloud_map_, map_msg);
    map_msg.header = header;
    map_msg.header.frame_id = odom_frame_id_;
    pointcloud_pub_.publish(map_msg);
  }
}; // class Mapper
} // namespace viso2_slam

int main(int argc, char** argv)
{
  ros::init(argc, argv, "mapper");
  viso2_slam::Mapper viso2_mapper;
  ros::spin();
  return 0;
}