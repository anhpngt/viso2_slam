#include <iostream>
#include <random>
#include <string>

#include <ros/ros.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

#include <viso2_slam/BoolStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <nav_msgs/Path.h>
#include <viso2_slam/VisoInfo.h>
#include <tf/tfMessage.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include "pose_graph_3d_error_term.h"
#include <Eigen/Dense>

using namespace std;

namespace viso2_slam
{
class PoseKalmanFilter
{
 private:
  // Matrices for computation
  Eigen::MatrixXd A_, H_, Q_, R_, P_, P0_;
  Eigen::MatrixXd I_;

  // System dimensions
  int m_, n_;

  // Initial and current time
  double t0_, t_;
  double dt_;

  // Check initialization
  bool initialized_;

  // Estimated states
  Eigen::VectorXd x_hat_, x_hat_new_;
  
 public:
  /**
   * Create a Kalman filter with the specified matrices.
   *   A - System dynamics matrix
   *   H - To output measurement matrix from state matrix
   *   Q - Process noise covariance
   *   R - Measurement noise covariance
   *   P - Estimate error covariance
   */
  PoseKalmanFilter(double dt,
                   const Eigen::MatrixXd& A,
                   const Eigen::MatrixXd& H,
                   const Eigen::MatrixXd& Q,
                   const Eigen::MatrixXd& R,
                   const Eigen::MatrixXd& P)
  : A_(A), H_(H), Q_(Q), R_(R), P_(P), P0_(P), I_(n_, n_),
    m_(H_.rows()), n_(A.rows()), dt_(dt), initialized_(false),
    x_hat_(n_), x_hat_new_(n_)
  {
    if(Q.rows() != 12 || Q.cols() != 12)
      cout << "[KalmanFilter] WARN: <Q> has invalid dimension!" << endl;
    if(R.rows() != 12 || R.cols() != 12)
      cout << "[KalmanFilter] WARN: <R> has invalid dimension!" << endl;
    if(P.rows() != 12 || P.cols() != 12)
      cout << "[KalmanFilter] WARN: <P> has invalid dimension!" << endl;
    if(m_ != 12 || n_ != 12)
      cout << "[KalmanFilter] WARN: System dimensions are invalid!" << endl;
    A_ = createSystemDynamics(dt);
    H_ = Eigen::MatrixXd::Identity(12, 12);
    I_ = Eigen::MatrixXd::Identity(12, 12);
  };

  PoseKalmanFilter() {};

  /**
   * Initialization. Required.
   */
  void init()
  {
    x_hat_.setZero();
    t0_ = 0;
    t_ = 0;
    initialized_ = true;
  }

  void init(const double t0, const Eigen::VectorXd& x0)
  {
    x_hat_ = x0;
    t0_ = t0;
    t_ = t0;
    initialized_ = true;
  }

  void init(const double t0, const tf::Transform& tf_z0)
  {
    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(12);
    x0.head(6) = fromTFTransformToEigen(tf_z0);
    init(t0, x0);
  }

  /**
   * Update the estimated state with measurement value.
   */
  void update(const Eigen::VectorXd& z)
  {
    if(!initialized_)
      throw std::runtime_error("Filter is not initialized!");

    x_hat_new_ = A_ * x_hat_;
    P_ = A_ * P_ * A_.transpose() + Q_;
    Eigen::MatrixXd K = P_ * H_.transpose() * (H_ * P_ * H_.transpose() + R_).inverse();
    x_hat_new_ += K * (z - H_ * x_hat_new_);
    P_ = (I_ - K * H_) * P_;
    x_hat_ = x_hat_new_;
  }
  
  void update(const double t, const tf::Transform& tfPose, const tf::Transform& tfDelPose)
  {
    double dt = t - t_;
    A_ = createSystemDynamics(dt);
    Eigen::VectorXd z(12);
    z << fromTFTransformToEigen(tfPose), fromTFTransformToEigen(tfDelPose);
    update(z);
    t_ = t;
    dt_ = dt;
  }

  /**
   * Return the current state.
   */
  Eigen::VectorXd getRawState() {return x_hat_;}
  
  tf::Transform getPoseState()
  {
    tf::Vector3 x_pos(x_hat_[0], x_hat_[1], x_hat_[2]);
    tf::Quaternion x_quat;
    x_quat.setRPY(x_hat_[3], x_hat_[4], x_hat_[5]);
    return tf::Transform(x_quat, x_pos);
  }
  
  double getTime() {return t_;}

 private:

  /**
   * Create a new System dynamic matrix from new dt
   */
  Eigen::MatrixXd createSystemDynamics(double dt)
  {
    Eigen::MatrixXd A = Eigen::MatrixXd::Identity(12, 12);
    double k = dt / dt_;
    A(0, 6)   = k;
    A(1, 7)   = k;
    A(2, 8)   = k;
    A(3, 9)   = k;
    A(4, 10)  = k;
    A(5, 11)  = k;
    return A;
  }

  /**
   * Convert tf::Transform to 6-dof Eigen::VectorXd
   */
  void fromTFTransformToEigen(const tf::Transform& tf, Eigen::VectorXd& eigenvec)
  {
    eigenvec = Eigen::VectorXd(6);
    tf::Vector3 tf_xyz = tf.getOrigin();
    eigenvec[0] = tf_xyz.x();
    eigenvec[1] = tf_xyz.y();
    eigenvec[2] = tf_xyz.z();
    tf.getBasis().getRPY(eigenvec[3], eigenvec[4], eigenvec[5], 1);
  }

  Eigen::VectorXd fromTFTransformToEigen(const tf::Transform& tf)
  {
    Eigen::VectorXd eigenvec = Eigen::VectorXd(6);
    tf::Vector3 tf_xyz = tf.getOrigin();
    eigenvec[0] = tf_xyz.x();
    eigenvec[1] = tf_xyz.y();
    eigenvec[2] = tf_xyz.z();
    tf.getBasis().getRPY(eigenvec[3], eigenvec[4], eigenvec[5], 1);
    return eigenvec;
  }
}; // class PoseKalmanFilter

class PoseOptimizer
{

 private:

  // Initialization variables
  message_filters::Subscriber<geometry_msgs::PoseStamped> pose_sub_;
  message_filters::Subscriber<VisoInfo> viso2info_sub_;
  typedef message_filters::sync_policies::ApproximateTime<geometry_msgs::PoseStamped, VisoInfo> poseSyncPolicy;
  typedef message_filters::Synchronizer<poseSyncPolicy> poseSync;
  boost::shared_ptr<poseSync> pose_sync_;

  ros::Subscriber gps_sub_;           // GPS data
  ros::Subscriber kitti_tf_sub_;      // subscribe to kitti ground-truth pose and convert to gps data
  ros::Publisher optimized_pose_pub_;
  ros::Publisher optimized_trajectory_pub_;
  ros::Publisher new_optimization_pub_;
  tf::TransformBroadcaster tf_broadcaster_;

  bool publish_tf_;
  bool is_kitti_;
  bool add_gps_noise_, is_gps_noisy_;      // Whether to add noise to simulated GPS by Kitti
  int queue_size_;
  std::string odom_frame_id_, base_link_frame_id_, gps_frame_id_;
  std::string viso2_pose_topic_, viso2_info_topic_;
  nav_msgs::Path trajectory_msg_;

  // Processing/Data variables
  bool is_vo_lost_, is_gps_lost_;
  bool is_initialized_;
  tf::Transform last_raw_pose_;           // Last optimized pose, pre-optimized
  tf::Transform last_optimized_pose_;     // Last optimized pose, pro-optimized
  tf::Transform last_gps_pose_;
  tf::Transform prev_pose_;
  tf::Transform gps_diff_pose_;
  ros::Time last_pose_time_, last_gps_time_;
  ros::Duration gps_diff_time_;
  PoseKalmanFilter kalman_;

  std::random_device rd_{};
  std::mt19937 rd_generator_{rd_()}; 
  // std::default_random_engine rd_generator_;
  std::normal_distribution<> p_noise_{0., 0.5};
  std::normal_distribution<> q_noise_{0., 1. / 180. * 3.141592653589793238463};

  // Optimizer
  Optimizer::VectorofPoses optimizer_poses_;
  Optimizer::VectorOfConstraints optimizer_constraints_;
  size_t total_optimized_size_;
  ceres::Problem *optimizer_problem_ = new ceres::Problem; // BROKEN
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
    is_gps_noisy_ = false;
    if(is_kitti_)
    {
      ROS_INFO("Working with KITTI dataset: Simulating GPS and IMU data based on /tf topic.");
      if(add_gps_noise_)
      {
        ROS_WARN("Gaussian noise will be added to GPS data.");
        kitti_tf_sub_ = nh.subscribe("/tf", queue_size_, &PoseOptimizer::kittiTFWithNoiseCallback, this);

        // Add covariance matrix for Kalman Filter
        Eigen::MatrixXd Q(12, 12);      // Process noise covariance
        Eigen::MatrixXd R(12, 12);      // Measurement noise covariance
        Eigen::MatrixXd P(12, 12);      // Estimate error covariance

        Q << 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5;

        R << 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0.001, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0.001, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0.001, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0.001, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.001, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.001;

        P = Eigen::MatrixXd::Identity(12, 12);

        kalman_ = PoseKalmanFilter(0.1,
                                   Eigen::MatrixXd::Zero(12, 12),
                                   Eigen::MatrixXd::Zero(12, 12),
                                   Q, R, P);
        is_gps_noisy_ = true;
      }
      else
        kitti_tf_sub_ = nh.subscribe("/tf", queue_size_, &PoseOptimizer::kittiTFCallback, this);
    }
    else
    {
      ROS_WARN("Working with NTU dataset: Receiving data from true GPS sensor.");
      gps_sub_ = nh.subscribe("/dji_sdk/gps_pose", queue_size_, &PoseOptimizer::gpsCallback, this);
    }

    // Result publisher
    optimized_pose_pub_ = local_nh.advertise<geometry_msgs::PoseStamped>("optimized_pose", 1);
    optimized_trajectory_pub_ = local_nh.advertise<nav_msgs::Path>("trajectory", 1, true);
    new_optimization_pub_ = local_nh.advertise<viso2_slam::BoolStamped>("optimization", 1);
    trajectory_msg_.header.frame_id = odom_frame_id_;

    // Optimizer
    total_optimized_size_ = 0;
    optimizer_options_.max_num_iterations = 100;
	  optimizer_options_.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
	  optimizer_options_.minimizer_progress_to_stdout = false;

    // Data initialization
    is_vo_lost_ = false;
    is_gps_lost_ = true;
    is_initialized_ = false;
    last_raw_pose_ = tf::Transform::getIdentity();
    last_optimized_pose_ = tf::Transform::getIdentity();
    last_gps_pose_ = tf::Transform::getIdentity();
    prev_pose_ = tf::Transform::getIdentity();
    
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
    tf::Transform new_vo_tf(tf::Quaternion(pose_msg->pose.orientation.z, -pose_msg->pose.orientation.x,
                                           -pose_msg->pose.orientation.y, pose_msg->pose.orientation.w),
                            tf::Vector3(pose_msg->pose.position.z, -pose_msg->pose.position.x, -pose_msg->pose.position.y));
    ros::Time new_vo_time(pose_msg->header.stamp.sec, pose_msg->header.stamp.nsec);

    tf_broadcaster_.sendTransform(tf::StampedTransform(new_vo_tf, new_vo_time, odom_frame_id_, "odom"));

    // Transform pose from odom frame to optimized frame
    tf::Transform current_pose = last_optimized_pose_ * (last_raw_pose_.inverse() * new_vo_tf);
    tf::Vector3 current_p = current_pose.getOrigin();   
    tf::Quaternion current_q = current_pose.getRotation();
    Eigen::Vector3d eigen_p(current_p.x(), current_p.y(), current_p.z());
    Eigen::Quaterniond eigen_q(current_q.w(), current_q.x(), current_q.y(), current_q.z());

    if((new_vo_time - last_gps_time_).toSec() > 1.0)
    {
      ROS_WARN("Lost GPS signal");
      is_gps_lost_ = true;
    }

    if(!is_gps_lost_ && is_initialized_) // KF applied
    {
      tf::Transform current_gps_pose = getInterpolatedGPSPose(new_vo_time);
      tf::Transform relative_vo_pose = prev_pose_.inverse() * new_vo_tf;

      kalman_.update(new_vo_time.toSec(), current_gps_pose, relative_vo_pose);
      tf::Transform filtered_pose = kalman_.getPoseState();

      // Add pose to trajectory
      trajectory_msg_.header.seq = pose_msg->header.seq;
      trajectory_msg_.header.stamp = pose_msg->header.stamp;
      geometry_msgs::PoseStamped new_pose_msg;
      new_pose_msg.header = trajectory_msg_.header;
      new_pose_msg.pose.position.x = filtered_pose.getOrigin().x();
      new_pose_msg.pose.position.y = filtered_pose.getOrigin().y();
      new_pose_msg.pose.position.z = filtered_pose.getOrigin().z();
      new_pose_msg.pose.orientation.x = filtered_pose.getRotation().x();
      new_pose_msg.pose.orientation.y = filtered_pose.getRotation().y();
      new_pose_msg.pose.orientation.z = filtered_pose.getRotation().z();
      new_pose_msg.pose.orientation.w = filtered_pose.getRotation().w();
      trajectory_msg_.poses.push_back(new_pose_msg);

      last_raw_pose_ = new_vo_tf;
      last_optimized_pose_ = filtered_pose;

      viso2_slam::BoolStamped bool_msg;
      bool_msg.header = trajectory_msg_.header;
      bool_msg.data = false;
      new_optimization_pub_.publish(bool_msg);
    }
    else // Pure VO
    {
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

      viso2_slam::BoolStamped bool_msg;
      bool_msg.header = trajectory_msg_.header;
      bool_msg.data = false;
      new_optimization_pub_.publish(bool_msg);
    }
    
    optimized_trajectory_pub_.publish(trajectory_msg_);
    optimized_pose_pub_.publish(trajectory_msg_.poses.back());

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

    if(!is_initialized_)
    {
      gps_diff_time_ = ros::Duration(0.1);
      gps_diff_pose_ = tf::Transform::getIdentity();
      is_initialized_ = true;
      kalman_.init(current_time.toSec(), gps_pose);
    }
    else
    {
      gps_diff_time_ = current_time - last_gps_time_;
      gps_diff_pose_ = last_gps_pose_.inverse() * gps_pose;
    }

    last_gps_pose_ = gps_pose;
    last_gps_time_ = current_time;
    is_gps_lost_ = false;

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

        double rl, pt, yw;
        tf::Matrix3x3(kitti_quat).getRPY(rl, pt, yw, 1);
        rl += q_noise_(rd_generator_);
        pt += q_noise_(rd_generator_);
        yw += q_noise_(rd_generator_);
        kitti_quat.setRPY(rl, pt, yw);
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
    
    for(size_t i = 0, i_end = NonCorrectedSim3.size(); i < i_end; ++i)
      poses[i] = NonCorrectedSim3[i];

    // Edges (constraint just between two neighboring scans)
    for(size_t i = 0, i_end = NonCorrectedSim3.size() - 1; i < i_end; ++i)
    {
      const Optimizer::Pose3d Swi = poses[i];
      const Optimizer::Pose3d Swj = poses[i+1];
      const Optimizer::Pose3d Sjw = Swj.inverse();
      const Optimizer::Pose3d Sji = Sjw * Swi;

      Optimizer::Constraint3d constraint(i, i+1, Sji);
        
      constraints.push_back(constraint);
    }
    
    // Modify the pose of last keyframe to the corrected one
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

    // Set constant pose for start and end scans
    problem.SetParameterBlockConstant(poses[0].p.data());
    problem.SetParameterBlockConstant(poses[0].q.coeffs().data());
    problem.SetParameterBlockConstant(poses[endID].p.data());
    problem.SetParameterBlockConstant(poses[endID].q.coeffs().data());

    // Optimize
    ceres::Solver::Options options;
    options.max_num_iterations = 50;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    Optimizer::VectorofPoses CorrectedSim3;
    for(size_t i = 0, i_end = NonCorrectedSim3.size(); i < i_end; ++i)
      CorrectedSim3.push_back(poses[i]);
    
    return CorrectedSim3;
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