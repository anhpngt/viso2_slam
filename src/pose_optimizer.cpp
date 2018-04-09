#include <iostream>
#include <string>

#include <ros/ros.h>
#include <tf/transform_datatypes.h>

#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <viso2_slam/VisoInfo.h>
#include <tf/tfMessage.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/sync_policies/approximate_time.h>

using namespace std;

namespace viso2_slam
{
class PoseOptimizer
{

private:

  message_filters::Subscriber<geometry_msgs::PoseStamped> pose_sub_;
  message_filters::Subscriber<VisoInfo> viso2info_sub_;
  typedef message_filters::sync_policies::ApproximateTime<geometry_msgs::PoseStamped, VisoInfo> poseSyncPolicy;
  typedef message_filters::Synchronizer<poseSyncPolicy> poseSync;
  boost::shared_ptr<poseSync> pose_sync_;

  ros::Subscriber gps_sub_;
  ros::Subscriber kitti_tf_sub_; // subscribe to kitti ground-truth pose and convert to gps data

  ros::Publisher optimized_pose_;

  bool is_kitti;
  int queue_size_;
  std::string viso2_pose_topic_, viso2_info_topic_;

public:

  PoseOptimizer()
  {
    // Parameters
    ros::NodeHandle nh;
    ros::NodeHandle local_nh("~");
    local_nh.param("pose_topic", viso2_pose_topic_, std::string("/stereo_slam/pose"));
    local_nh.param("info_topic", viso2_info_topic_, std::string("/stereo_slam/info"));
    local_nh.param("is_kitti", is_kitti, true);
    local_nh.param("queue_size", queue_size_, 5);
    
    // Set up sync callback for viso2 odom
    pose_sub_.subscribe(nh, viso2_pose_topic_, 3);
    viso2info_sub_.subscribe(nh, viso2_info_topic_, 3);
    pose_sync_.reset(new poseSync(poseSyncPolicy(queue_size_), pose_sub_, viso2info_sub_));
    pose_sync_->registerCallback(boost::bind(&PoseOptimizer::viso2PoseCallback, this, _1, _2));

    // Set up gps callback
    if(is_kitti)
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
    ROS_INFO("Initialization finished!");
  }

protected:

  void viso2PoseCallback(const geometry_msgs::PoseStamped::ConstPtr &pose_msg, const viso2_slam::VisoInfo::ConstPtr &info_msg)
  {
    ROS_INFO("Synchronization successful");
  }

  void gpsCallback(const geometry_msgs::PoseWithCovarianceStamped& gps_pose)
  {
    ROS_INFO("Received GPS data");
  }

  void kittiTFCallback(const tf::tfMessage::ConstPtr& tf_msg)
  {
    // Note that translation from /tf is: x-right, y-down, z-forward
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