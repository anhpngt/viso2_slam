#include <iostream>
#include <string>

#include <ros/ros.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <viso2_slam/VisoInfo.h>
#include <tf/tfMessage.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/sync_policies/approximate_time.h>

#include "pose_graph_3d_error_term.h"

using namespace std;

namespace viso2_slam
{
class PoseOptimizer
{

private:

  // Initialization variables
  message_filters::Subscriber<geometry_msgs::PoseStamped> pose_sub_;
  message_filters::Subscriber<VisoInfo> viso2info_sub_;
  typedef message_filters::sync_policies::ApproximateTime<geometry_msgs::PoseStamped, VisoInfo> poseSyncPolicy;
  typedef message_filters::Synchronizer<poseSyncPolicy> poseSync;
  boost::shared_ptr<poseSync> pose_sync_;

  ros::Subscriber gps_sub_;
  ros::Subscriber kitti_tf_sub_; // subscribe to kitti ground-truth pose and convert to gps data
  ros::Publisher optimized_pose_;
  tf::TransformBroadcaster tf_broadcaster_;

  bool publish_tf_;
  bool is_kitti_;
  int queue_size_;
  std::string odom_frame_id_, base_link_frame_id_, gps_frame_id_;
  std::string viso2_pose_topic_, viso2_info_topic_;

  // Processing/Data variables
  bool is_vo_lost_;
  tf::Transform last_optimized_pose_;
  tf::Transform last_gps_pose_;
  tf::Transform prev_pose_;
  ros::Time last_gps_time_;

public:

  PoseOptimizer()
  {
    // Parameters
    ros::NodeHandle nh;
    ros::NodeHandle local_nh("~");
    local_nh.param("odom_frame_id", odom_frame_id_, std::string("/odom"));
    local_nh.param("base_link_frame_id", base_link_frame_id_, std::string("/base_link"));
    local_nh.param("gps_frame_id", gps_frame_id_, std::string("/gps"));
    local_nh.param("pose_topic", viso2_pose_topic_, std::string("/stereo_slam/pose"));
    local_nh.param("info_topic", viso2_info_topic_, std::string("/stereo_slam/info"));
    local_nh.param("publish_tf", publish_tf_, true);
    local_nh.param("is_kitti", is_kitti_, true);
    local_nh.param("queue_size", queue_size_, 5);
    
    // Set up sync callback for viso2 odom
    pose_sub_.subscribe(nh, viso2_pose_topic_, 3);
    viso2info_sub_.subscribe(nh, viso2_info_topic_, 3);
    pose_sync_.reset(new poseSync(poseSyncPolicy(queue_size_), pose_sub_, viso2info_sub_));
    pose_sync_->registerCallback(boost::bind(&PoseOptimizer::viso2PoseCallback, this, _1, _2));

    // Set up gps callback
    if(is_kitti_)
    {
      ROS_INFO("Working with KITTI dataset: Simulating GPS and IMU data based on /tf topic.");
      kitti_tf_sub_ = nh.subscribe("/tf", queue_size_, &PoseOptimizer::kittiTFCallback, this);
    }
    else
    {
      ROS_INFO("Working with NTU dataset: Receiving data from true GPS sensor.");
      gps_sub_ = nh.subscribe("/dji_sdk/gps_pose", queue_size_, &PoseOptimizer::gpsCallback, this);
    }

    // Result publisher
    optimized_pose_ = local_nh.advertise<geometry_msgs::PoseStamped>("optimized_pose", 1);

    // Data initialization
    last_optimized_pose_ = tf::Transform::getIdentity();
    
    ROS_INFO("Pose Optimizer initialization finished! Waiting for incoming topics...");
  }

private:

  void viso2PoseCallback(const geometry_msgs::PoseStamped::ConstPtr &pose_msg, const viso2_slam::VisoInfo::ConstPtr &info_msg)
  {
    // ROS_INFO("Synchronization successful");
    // TODO: add function to depend on gps when lost_track
    if(info_msg->got_lost)
    {
      ROS_WARN("VO lost track.");
      is_vo_lost_ = true;
      return;
    }
    else is_vo_lost_ = false;

    // Transform from camera_link to base_link
    tf::Vector3 vo_xyz = tf::Vector3(pose_msg->pose.position.z, -pose_msg->pose.position.x, -pose_msg->pose.position.y);
    tf::Quaternion vo_quat = tf::Quaternion(pose_msg->pose.orientation.z, -pose_msg->pose.orientation.x,
                                            -pose_msg->pose.orientation.y, pose_msg->pose.orientation.w);
    tf::Transform new_vo_tf = tf::Transform(vo_quat, vo_xyz);
    ros::Time new_vo_time = ros::Time(pose_msg->header.stamp.sec, pose_msg->header.stamp.nsec);

    tf_broadcaster_.sendTransform(tf::StampedTransform(new_vo_tf, new_vo_time, odom_frame_id_, "odom"));


    // Update previous
    prev_pose_ = new_vo_tf;
  }

  void gpsCallback(const geometry_msgs::PoseWithCovarianceStamped& pose_msg)
  {
    tf::Vector3 gps_xyz = tf::Vector3(pose_msg.pose.pose.position.x, pose_msg.pose.pose.position.y, pose_msg.pose.pose.position.z);
    tf::Quaternion gps_quat = tf::Quaternion(pose_msg.pose.pose.orientation.x, pose_msg.pose.pose.orientation.y,
                                             pose_msg.pose.pose.orientation.z, pose_msg.pose.pose.orientation.w);
    last_gps_pose_ = tf::Transform(gps_quat, gps_xyz);
    last_gps_time_ = ros::Time(pose_msg.header.stamp.sec, pose_msg.header.stamp.nsec);
    tf_broadcaster_.sendTransform(tf::StampedTransform(last_gps_pose_, last_gps_time_, odom_frame_id_, gps_frame_id_));
    // ROS_INFO("Received GPS data");
  }

  void kittiTFCallback(const tf::tfMessage::ConstPtr& tf_msg)
  {
    // Note that coordinate for camera frame in /tf is: x-right, y-down, z-forward
    // Look for the correct transform in the array
    for(int i = 0, i_end = tf_msg->transforms.size(); i < i_end; i++)
    {
      if(tf_msg->transforms[i].child_frame_id == "camera_left" && tf_msg->transforms[i].header.frame_id == "world")
      {
        // Simulate GPS data
        geometry_msgs::PoseWithCovarianceStamped gps_pose;
        gps_pose.header.seq = tf_msg->transforms[i].header.seq;
        gps_pose.header.stamp = tf_msg->transforms[i].header.stamp;
        gps_pose.header.frame_id = "/gps";
        gps_pose.pose.pose.position.x = tf_msg->transforms[i].transform.translation.z;
        gps_pose.pose.pose.position.y = -tf_msg->transforms[i].transform.translation.x;
        gps_pose.pose.pose.position.z = -tf_msg->transforms[i].transform.translation.y;

        tf::Quaternion kitti_quat(tf_msg->transforms[i].transform.rotation.x,
                                  tf_msg->transforms[i].transform.rotation.y,
                                  tf_msg->transforms[i].transform.rotation.z,
                                  tf_msg->transforms[i].transform.rotation.w);
        gps_pose.pose.pose.orientation.x = kitti_quat.z();
        gps_pose.pose.pose.orientation.y = -kitti_quat.x();
        gps_pose.pose.pose.orientation.z = -kitti_quat.y();
        gps_pose.pose.pose.orientation.w = kitti_quat.w();
        gpsCallback(gps_pose);
        return;
      }
    }
  }
};
} // namespace viso2_slam

int main(int argc, char** argv)
{
  ros::init(argc, argv, "pose_optimizer");
  viso2_slam::PoseOptimizer pose_optimizer;
  ros::spin();
  return 0;
}