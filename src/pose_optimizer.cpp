#include <iostream>
#include <random>
#include <string>

#include <ros/ros.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <nav_msgs/Path.h>
#include <viso2_slam/VisoInfo.h>
#include <tf/tfMessage.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <kalman.h>
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

  ros::Subscriber gps_sub_; // GPS data
  ros::Subscriber kitti_tf_sub_; // subscribe to kitti ground-truth pose and convert to gps data
  ros::Publisher optimized_pose_pub_; // i dunno whats this for
  ros::Publisher optimized_trajectory_pub_;
  tf::TransformBroadcaster tf_broadcaster_;

  bool publish_tf_;
  bool is_kitti_;
  bool add_gps_noise_;      // Whether to add noise to simulated GPS by Kitti
  int queue_size_;
  std::string odom_frame_id_, base_link_frame_id_, gps_frame_id_;
  std::string viso2_pose_topic_, viso2_info_topic_;
  nav_msgs::Path trajectory_msg_;

  // Processing/Data variables
  bool is_vo_lost_;
  bool is_initialized;
  tf::Transform last_raw_pose_;           // Last optimized pose, pre-optimized
  tf::Transform last_optimized_pose_;     // Last optimized pose, pro-optimized
  tf::Transform last_gps_pose_;
  tf::Transform prev_pose_;
  tf::Transform gps_diff_pose_;
  ros::Time last_pose_time_, last_gps_time_;
  ros::Duration gps_diff_time_;

  std::random_device rd_{};
  std::mt19937 rd_generator_{rd_()}; 
  // std::default_random_engine rd_generator_;
  std::normal_distribution<> p_noise_{0., 0.5};
  // std::normal_distribution<> q_noise_{0., 1. / 180. * 3.141592653589793238463};

  // Optimizer
  Optimizer::VectorofPoses optimizer_poses_;
  Optimizer::VectorOfConstraints optimizer_constraints_;
  size_t total_optimized_size_;
  ceres::Problem *optimizer_problem_ = new ceres::Problem;
  ceres::Solver::Options optimizer_options_;
  ceres::Solver::Summary optimizer_summary_;

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
    local_nh.param("queue_size", queue_size_, 10);
    local_nh.param("publish_tf", publish_tf_, true);
    local_nh.param("is_kitti", is_kitti_, true);
    local_nh.param("add_gps_noise", add_gps_noise_, false);
    
    // Set up sync callback for viso2 odom
    pose_sub_.subscribe(nh, viso2_pose_topic_, 3);
    viso2info_sub_.subscribe(nh, viso2_info_topic_, 3);
    pose_sync_.reset(new poseSync(poseSyncPolicy(queue_size_), pose_sub_, viso2info_sub_));
    pose_sync_->registerCallback(boost::bind(&PoseOptimizer::viso2PoseCallback, this, _1, _2));

    // Callbacks/Subscribers
    if(is_kitti_)
    {
      ROS_INFO("Working with KITTI dataset: Simulating GPS and IMU data based on /tf topic.");
      if(add_gps_noise_)
      {
        ROS_WARN("Gaussian noise will be added to GPS data");
        kitti_tf_sub_ = nh.subscribe("/tf", queue_size_, &PoseOptimizer::kittiTFWithNoiseCallback, this);
      }
      else
        kitti_tf_sub_ = nh.subscribe("/tf", queue_size_, &PoseOptimizer::kittiTFCallback, this);
    }
    else
    {
      ROS_INFO("Working with NTU dataset: Receiving data from true GPS sensor.");
      gps_sub_ = nh.subscribe("/dji_sdk/gps_pose", queue_size_, &PoseOptimizer::gpsCallback, this);
    }

    // Result publisher
    optimized_pose_pub_ = local_nh.advertise<geometry_msgs::PoseStamped>("optimized_pose", 1);
    optimized_trajectory_pub_ = local_nh.advertise<nav_msgs::Path>("trajectory", 1, true);
    trajectory_msg_.header.frame_id = odom_frame_id_;

    // Optimizer
    total_optimized_size_ = 0;
    optimizer_options_.max_num_iterations = 100;
	  optimizer_options_.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
	  optimizer_options_.minimizer_progress_to_stdout = false;

    // Data initialization
    is_vo_lost_ = false;
    is_initialized = false;
    last_raw_pose_ = tf::Transform::getIdentity();
    last_optimized_pose_ = tf::Transform::getIdentity();
    last_gps_pose_ = tf::Transform::getIdentity();
    prev_pose_ = tf::Transform::getIdentity();
    
    ROS_INFO("Pose Optimizer initialization finished! Waiting for incoming topics...");
    // ros::Rate rate(20);
    // while(ros::ok())
    // {
    //   ros::spinOnce();
    //   ROS_INFO("Last update: [%.6lf, %.6lf]", last_pose_time_.toSec(), last_gps_time_.toSec());
    //   rate.sleep();
    // }
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
    tf::Transform new_vo_tf(tf::Quaternion(pose_msg->pose.orientation.z, -pose_msg->pose.orientation.x,
                                           -pose_msg->pose.orientation.y, pose_msg->pose.orientation.w),
                            tf::Vector3(pose_msg->pose.position.z, -pose_msg->pose.position.x, -pose_msg->pose.position.y));
    ros::Time new_vo_time(pose_msg->header.stamp.sec, pose_msg->header.stamp.nsec);

    tf_broadcaster_.sendTransform(tf::StampedTransform(new_vo_tf, new_vo_time, odom_frame_id_, "odom"));

    // Transform pose from odom frame to optimized frame
    tf::Transform current_pose = last_optimized_pose_ * (last_raw_pose_.inverse() * new_vo_tf);
    // double x,y,z,rl,pt,yw;
    // x = current_pose.getOrigin().x();
    // y = current_pose.getOrigin().y();
    // z = current_pose.getOrigin().z();
    // current_pose.getBasis().getRPY(rl, pt, yw, 1);
    tf::Vector3 current_p = current_pose.getOrigin();   
    tf::Quaternion current_q = current_pose.getRotation();
    Eigen::Vector3d eigen_p(current_p.x(), current_p.y(), current_p.z());
    Eigen::Quaterniond eigen_q(current_q.w(), current_q.x(), current_q.y(), current_q.z());

    tf::Transform current_gps_pose = getInterpolatedGPSPose(new_vo_time);
    tf::Transform pose_discrepancy = current_gps_pose.inverse() * current_pose;
    if(pose_discrepancy.getOrigin().length() > 10 && optimizer_poses_.size() > 50)
    {
      // Change final pose to GPS
      tf::Vector3 current_gps_p = current_gps_pose.getOrigin();
      tf::Quaternion current_gps_q = current_gps_pose.getRotation();
      Eigen::Vector3d eigen_gps_p(current_gps_p.x(), current_gps_p.y(), current_gps_p.z());
      Eigen::Quaterniond eigen_gps_q(current_gps_q.w(), current_gps_q.x(), current_gps_q.y(), current_gps_q.z());

      Optimizer::Pose3d new_vertex(eigen_p, eigen_q);
      Optimizer::Pose3d groundtruth_vertex(eigen_gps_p, eigen_gps_q);
      optimizer_poses_.push_back(new_vertex); // add new vertex to optimizer's graph
      // size_t i = optimizer_poses_.size() - 2;
      // size_t j = optimizer_poses_.size() - 1;
      // const Optimizer::Pose3d Sji = optimizer_poses_[j].inverse() * optimizer_poses_[i];
      // Optimizer::Constraint3d constraint(i, j, Sji);
      // optimizer_constraints_.push_back(constraint);
      // optimizer_poses_[j] = groundtruth_vertex;

      // Add pose to trajectory
      trajectory_msg_.header.seq = pose_msg->header.seq;
      trajectory_msg_.header.stamp = pose_msg->header.stamp;
      geometry_msgs::PoseStamped new_pose_msg;
      new_pose_msg.header = trajectory_msg_.header;
      new_pose_msg.pose.position.x = eigen_p.x();
      new_pose_msg.pose.position.y = eigen_p.y();
      new_pose_msg.pose.position.z = eigen_p.z();
      new_pose_msg.pose.orientation.x = eigen_q.x();
      new_pose_msg.pose.orientation.y = eigen_q.y();
      new_pose_msg.pose.orientation.z = eigen_q.z();
      new_pose_msg.pose.orientation.w = eigen_q.w();
      trajectory_msg_.poses.push_back(new_pose_msg);

      optimizer_poses_ = solveOptimizationProblem(optimizer_poses_, groundtruth_vertex);

      // Extract optimized poses from optimzer_poses_
      extractOptimizedTrajectory();

      last_raw_pose_ = new_vo_tf;
      last_optimized_pose_ = current_gps_pose;
      total_optimized_size_ += optimizer_poses_.size();
      optimizer_poses_.clear();
    }
    else // VO has not drift that much
    {
      Optimizer::Pose3d new_vertex(eigen_p, eigen_q);
      optimizer_poses_.push_back(new_vertex); // add new vertex to optimizer's graph
      // size_t graph_size = optimizer_poses_.size();
      // if(graph_size > 1)
      // {
      //   size_t i = graph_size - 2;
      //   size_t j = graph_size - 1;
      //   const Optimizer::Pose3d Sji = optimizer_poses_[j].inverse() * optimizer_poses_[i];
      //   Optimizer::Constraint3d constraint(i, j, Sji);
      //   optimizer_constraints_.push_back(constraint);
      // }

      // Add pose to trajectory
      trajectory_msg_.header.seq = pose_msg->header.seq;
      trajectory_msg_.header.stamp = pose_msg->header.stamp;
      geometry_msgs::PoseStamped new_pose_msg;
      new_pose_msg.header = trajectory_msg_.header;
      new_pose_msg.pose.position.x = eigen_p.x();
      new_pose_msg.pose.position.y = eigen_p.y();
      new_pose_msg.pose.position.z = eigen_p.z();
      new_pose_msg.pose.orientation.x = eigen_q.x();
      new_pose_msg.pose.orientation.y = eigen_q.y();
      new_pose_msg.pose.orientation.z = eigen_q.z();
      new_pose_msg.pose.orientation.w = eigen_q.w();
      trajectory_msg_.poses.push_back(new_pose_msg);
    }
    
    optimized_trajectory_pub_.publish(trajectory_msg_);

    // Update previous
    prev_pose_ = new_vo_tf;
    last_pose_time_ = new_vo_time;
  }

  void gpsCallback(const geometry_msgs::PoseWithCovarianceStamped& pose_msg)
  {
    ros::Time current_time = ros::Time(pose_msg.header.stamp.sec, pose_msg.header.stamp.nsec);
    tf::Vector3 gps_xyz = tf::Vector3(pose_msg.pose.pose.position.x, pose_msg.pose.pose.position.y, pose_msg.pose.pose.position.z);
    tf::Quaternion gps_quat = tf::Quaternion(pose_msg.pose.pose.orientation.x, pose_msg.pose.pose.orientation.y,
                                             pose_msg.pose.pose.orientation.z, pose_msg.pose.pose.orientation.w);
    tf::Transform gps_pose = tf::Transform(gps_quat, gps_xyz);

    if(!is_initialized)
    {
      gps_diff_time_ = ros::Duration(0.1);
      gps_diff_pose_ = tf::Transform::getIdentity();
      is_initialized = true;
    }
    else
    {
      gps_diff_time_ = current_time - last_gps_time_;
      gps_diff_pose_ = last_gps_pose_.inverse() * gps_pose;
    }

    last_gps_pose_ = gps_pose;
    last_gps_time_ = current_time;

    tf_broadcaster_.sendTransform(tf::StampedTransform(gps_pose, current_time, odom_frame_id_, gps_frame_id_));
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

  void kittiTFWithNoiseCallback(const tf::tfMessage::ConstPtr& tf_msg)
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
        gps_pose.pose.pose.position.x = tf_msg->transforms[i].transform.translation.z + p_noise_(rd_generator_);
        gps_pose.pose.pose.position.y = -tf_msg->transforms[i].transform.translation.x + p_noise_(rd_generator_);
        gps_pose.pose.pose.position.z = -tf_msg->transforms[i].transform.translation.y + p_noise_(rd_generator_);

        tf::Quaternion kitti_quat(tf_msg->transforms[i].transform.rotation.x,
                                  tf_msg->transforms[i].transform.rotation.y,
                                  tf_msg->transforms[i].transform.rotation.z,
                                  tf_msg->transforms[i].transform.rotation.w);

        // double rl, pt, yw;
        // tf::Matrix3x3(kitti_quat).getRPY(rl, pt, yw, 1);
        // rl += q_noise_(rd_generator_);
        // pt += q_noise_(rd_generator_);
        // yw += q_noise_(rd_generator_);
        // kitti_quat.setRPY(rl, pt, yw);
        gps_pose.pose.pose.orientation.x = kitti_quat.z();
        gps_pose.pose.pose.orientation.y = -kitti_quat.x();
        gps_pose.pose.pose.orientation.z = -kitti_quat.y();
        gps_pose.pose.pose.orientation.w = kitti_quat.w();
        gpsCallback(gps_pose);
        return;
      }
    }
  }

  Optimizer::VectorofPoses solveOptimizationProblem(const Optimizer::VectorofPoses &NonCorrectedSim3,
                                                    Optimizer::Pose3d endCorrectedPose)
  {
    Optimizer::VectorofPoses poses;
    Optimizer::VectorOfConstraints constraints;
    poses.clear();
    constraints.clear();
    poses.resize(NonCorrectedSim3.size());
    
    // Set KeyFrame poses (vertex)
    // We assume the start frame and end frame are the two ends of the loop
    int endID = NonCorrectedSim3.size()-1;
    
    for(size_t i=0, iend=NonCorrectedSim3.size(); i<iend;i++)
      poses[i] = NonCorrectedSim3[i];

    // edges (constraint just between two neighboring scans)
    for(size_t i=0, iend=NonCorrectedSim3.size()-1; i<iend;i++)
    {
      const Optimizer::Pose3d Swi = poses[i];
      const Optimizer::Pose3d Swj = poses[i+1];
      const Optimizer::Pose3d Sjw = Swj.inverse();
      const Optimizer::Pose3d Sji = Sjw * Swi;

      Optimizer::Constraint3d constraint(i, i+1, Sji);
        
      constraints.push_back(constraint);
    }
    
    // modify the pose of last keyframe to the corrected one
    poses[endID] = endCorrectedPose;

    ceres::LossFunction* loss_function = NULL;
    ceres::LocalParameterization *quaternion_local_parameterization = new ceres::EigenQuaternionParameterization;
    ceres::Problem problem;

    for(Optimizer::VectorOfConstraints::const_iterator constraints_iter = constraints.begin(); constraints_iter != constraints.end(); ++constraints_iter) 
    {
      const Optimizer::Constraint3d& constraint = *constraints_iter;
      const Eigen::Matrix<double, 6, 6> sqrt_information = constraint.information.llt().matrixL();
      // Ceres will take ownership of the pointer.
      ceres::CostFunction* cost_function = Optimizer::PoseGraph3dErrorTerm::Create(constraint.t_be, sqrt_information);

      problem.AddResidualBlock(cost_function, loss_function,
                  poses[constraint.id_begin].p.data(),
                  poses[constraint.id_begin].q.coeffs().data(),
                  poses[constraint.id_end].p.data(),
                  poses[constraint.id_end].q.coeffs().data());

      problem.SetParameterization(poses[constraint.id_begin].q.coeffs().data(),
                  quaternion_local_parameterization);
      problem.SetParameterization(poses[constraint.id_end].q.coeffs().data(),
                  quaternion_local_parameterization);
    }

    // set constant pose for start and end scans
    problem.SetParameterBlockConstant(poses[0].p.data());
    problem.SetParameterBlockConstant(poses[0].q.coeffs().data());
    problem.SetParameterBlockConstant(poses[endID].p.data());
    problem.SetParameterBlockConstant(poses[endID].q.coeffs().data());

    // optimize
    ceres::Solver::Options options;
    options.max_num_iterations = 50;  //can try more iterations if not converge
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    Optimizer::VectorofPoses CorrectedSim3;
    for(size_t i = 0, iend = NonCorrectedSim3.size(); i < iend; i++)
      CorrectedSim3.push_back(poses[i]);
    
    return CorrectedSim3;
  }

  void solveOptimizationProblem() // BROKEN, DO NOT USE!
  {
    ceres::LossFunction* loss_function = NULL;
    ceres::LocalParameterization *quaternion_local_parameterization = new ceres::EigenQuaternionParameterization;

    for(Optimizer::VectorOfConstraints::const_iterator constraints_iter = optimizer_constraints_.begin(); constraints_iter != optimizer_constraints_.end(); ++constraints_iter)
    {
      const Optimizer::Constraint3d& constraint = *constraints_iter;
      const Eigen::Matrix<double, 6, 6> sqrt_information = constraint.information.llt().matrixL();
      // Ceres will take ownership of the pointer.
      ceres::CostFunction* cost_function = Optimizer::PoseGraph3dErrorTerm::Create(constraint.t_be, sqrt_information);

      optimizer_problem_->AddResidualBlock(cost_function, loss_function,
                            optimizer_poses_[constraint.id_begin].p.data(),
                            optimizer_poses_[constraint.id_begin].q.coeffs().data(),
                            optimizer_poses_[constraint.id_end].p.data(),
                            optimizer_poses_[constraint.id_end].q.coeffs().data());

      optimizer_problem_->SetParameterization(optimizer_poses_[constraint.id_begin].q.coeffs().data(),
                                              quaternion_local_parameterization);
      optimizer_problem_->SetParameterization(optimizer_poses_[constraint.id_end].q.coeffs().data(),
                                              quaternion_local_parameterization);
    }
    optimizer_problem_->SetParameterBlockConstant(optimizer_poses_[0].p.data());
    optimizer_problem_->SetParameterBlockConstant(optimizer_poses_[0].q.coeffs().data());
    optimizer_problem_->SetParameterBlockConstant(optimizer_poses_[optimizer_poses_.size()-1].p.data());
    optimizer_problem_->SetParameterBlockConstant(optimizer_poses_[optimizer_poses_.size()-1].q.coeffs().data());

    // Optimize graph, resulting pose will remain in optimizer_poses_
    ceres::Solve(optimizer_options_, optimizer_problem_, &optimizer_summary_);
    ROS_INFO("Pose graph optimized!");
    delete loss_function;
    delete quaternion_local_parameterization;
    cout << "dude" << endl;
    delete optimizer_problem_;
    optimizer_problem_ = new ceres::Problem;
    cout << "really? " << endl;
  }

  void extractOptimizedTrajectory()
  {
    if(optimizer_poses_.size() + total_optimized_size_ != trajectory_msg_.poses.size())
    {
      ROS_ERROR("Size error between optimizer poses and trajectory poses.");
      return;
    }
    for(size_t i = 0, i_end = optimizer_poses_.size(); i < i_end; i++)
    {
      Optimizer::Pose3d pose = optimizer_poses_[i];
      size_t j = i + total_optimized_size_;
      trajectory_msg_.poses[j].pose.position.x = pose.p.x();
      trajectory_msg_.poses[j].pose.position.y = pose.p.y();
      trajectory_msg_.poses[j].pose.position.z = pose.p.z();
      trajectory_msg_.poses[j].pose.orientation.x = pose.q.x();
      trajectory_msg_.poses[j].pose.orientation.y = pose.q.y();
      trajectory_msg_.poses[j].pose.orientation.z = pose.q.z();
      trajectory_msg_.poses[j].pose.orientation.w = pose.q.w();
    }
    ROS_INFO("Trajectory updated after optimization");
  }

  tf::Transform getInterpolatedGPSPose(ros::Time required_time)
  {
    if(required_time == last_gps_time_)
      return last_gps_pose_;
    
    double time_ratio = (required_time.toSec() - last_gps_time_.toSec()) / gps_diff_time_.toSec();
    tf::Vector3 del_trl = gps_diff_pose_.getOrigin() * time_ratio;
    double diff_roll, diff_pitch, diff_yaw;
    gps_diff_pose_.getBasis().getRPY(diff_roll, diff_pitch, diff_yaw);
    tf::Quaternion del_rot;
    del_rot.setRPY(diff_roll * time_ratio, diff_pitch * time_ratio, diff_yaw * time_ratio);

    tf::Transform del_pose(del_rot, del_trl);
    return last_gps_pose_ * del_pose;
  }
}; // class PoseOptimizer
} // namespace viso2_slam

int main(int argc, char** argv)
{
  ros::init(argc, argv, "pose_optimizer");
  viso2_slam::PoseOptimizer pose_optimizer;
  ros::spin();
  return 0;
}