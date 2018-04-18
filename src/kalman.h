/**
 * Kalman filter implementation using Eigen. Based on the following
 * introductory paper:
 *
 *     http://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf
 * 
 * Modified by phamngtuananh.
 * @author: Hayk Martirosyan
 * @date: 2014.11.15
 * A good read for the implementation of 6-DOF system can be found at
 * http://campar.in.tum.de/Chair/KalmanFilter
 */

#ifndef KALMAN_H_
#define KALMAN_H_

#include <Eigen/Dense>
#include <tf/transform_datatypes.h>

#pragma once

namespace viso2_slam
{
class PoseKalmanFilter
{
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
                   const Eigen::MatrixXd& P);

  PoseKalmanFilter();

  /**
   * Initialization. Required.
   */
  void init();
  void init(const double t0, const Eigen::VectorXd& x0);
  void init(const double t0, const tf::Transform& tf_z0);

  /**
   * Update the estimated state with measurement value.
   */
  void update(const Eigen::VectorXd& z);
  void update(const double dt, const Eigen::VectorXd& z, const Eigen::MatrixXd& A);
  void update(const double t, const tf::Transform& tf);

  /**
   * Create a new System dynamic matrix from new dt
   */
  Eigen::MatrixXd createSystemDynamics(double dt);

  /**
   * Convert tf::Transform to 6-dof Eigen::VectorXd
   */
  void fromTFTransformToEigen(const tf::Transform& tf, Eigen::VectorXd& eigenvec);
  Eigen::VectorXd fromTFTransformToEigen(const tf::Transform& tf);

  /**
   * Return the current state.
   */
  Eigen::VectorXd getRawState();
  tf::Transform getPoseState();
  double getTime();

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
};
} // namespace viso2_slam

#endif // # KALMAN_H_