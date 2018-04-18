/**
* Implementation of KalmanFilter class. Modified by phamngtuananh.
*
* @author: Hayk Martirosyan
* @date: 2014.11.15
*/

#include <iostream>
#include <stdexcept>

#include "kalman.h"

using namespace std;
namespace viso2_slam
{
PoseKalmanFilter::PoseKalmanFilter(double dt,
                                   const Eigen::MatrixXd& A,
                                   const Eigen::MatrixXd& H,
                                   const Eigen::MatrixXd& Q,
                                   const Eigen::MatrixXd& R,
                                   const Eigen::MatrixXd& P)
  : A_(A), H_(H), Q_(Q), R_(R), P_(P), P0_(P), I_(n_, n_),
    m_(H_.rows()), n_(A.rows()), dt_(dt), initialized_(false),
    x_hat_(n_), x_hat_new_(n_)
{
  if(Q.rows() != 18 || Q.cols() != 18)
    cout << "[KalmanFilter] WARN: <Q> has invalid dimension!" << endl;
  if(R.rows() != 6 || R.cols() != 6)
    cout << "[KalmanFilter] WARN: <R> has invalid dimension!" << endl;
  if(P.rows() != 18 || P.cols() != 18)
    cout << "[KalmanFilter] WARN: <P> has invalid dimension!" << endl;
  if(m_ != 6 || n_ != 18)
    cout << "[KalmanFilter] WARN: System dimensions are invalid!" << endl;
  A_ = createSystemDynamics(dt);
  H_ = Eigen::MatrixXd::Identity(6, 18);
  I_ = Eigen::MatrixXd::Identity(18, 18);
}

PoseKalmanFilter::PoseKalmanFilter() {}

///////////////////////////////////////////////////////////////////////////////
void PoseKalmanFilter::init()
{
  x_hat_.setZero();
  t0_ = 0;
  t_ = 0;
  initialized_ = true;
}

void PoseKalmanFilter::init(const double t0, const Eigen::VectorXd& x0)
{
  x_hat_ = x0;
  t0_ = t0;
  t_ = t0;
  initialized_ = true;
}

void PoseKalmanFilter::init(const double t0, const tf::Transform& tf_z0)
{
  Eigen::VectorXd x0 = Eigen::VectorXd::Zero(18);
  x0.head(6) = fromTFTransformToEigen(tf_z0);
  init(t0, x0);
}

///////////////////////////////////////////////////////////////////////////////
void PoseKalmanFilter::update(const Eigen::VectorXd& z)
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

void PoseKalmanFilter::update(const double dt, const Eigen::VectorXd& z, const Eigen::MatrixXd& A)
{
  A_ = A;
  dt_ = dt;
  t_ += dt_;
  update(z);
}

void PoseKalmanFilter::update(const double t, const tf::Transform& tf_z)
{
  double dt = t - t_;
  t_ = t;
  A_ = createSystemDynamics(dt);
  Eigen::VectorXd z = fromTFTransformToEigen(tf_z);
  update(z);
}

///////////////////////////////////////////////////////////////////////////////
Eigen::MatrixXd PoseKalmanFilter::createSystemDynamics(double dt)
{
  Eigen::MatrixXd A = Eigen::MatrixXd::Identity(18, 18);
  double dt2 = dt * dt / 2;
  A(0, 6)   = dt;
  A(1, 7)   = dt;
  A(2, 8)   = dt;
  A(3, 9)   = dt;
  A(4, 10)  = dt;
  A(5, 11)  = dt;
  A(6, 12)  = dt;
  A(7, 13)  = dt;
  A(8, 14)  = dt;
  A(9, 15)  = dt;
  A(10, 16) = dt;
  A(11, 17) = dt;
  A(0, 12) = dt2;
  A(1, 13) = dt2;
  A(2, 14) = dt2;
  A(3, 15) = dt2;
  A(4, 16) = dt2;
  A(5, 17) = dt2;
  return A;
}

///////////////////////////////////////////////////////////////////////////////
void PoseKalmanFilter::fromTFTransformToEigen(const tf::Transform& tf, Eigen::VectorXd& eigenvec)
{
  eigenvec = Eigen::VectorXd(6);
  tf::Vector3 tf_xyz = tf.getOrigin();
  eigenvec[0] = tf_xyz.x();
  eigenvec[1] = tf_xyz.y();
  eigenvec[2] = tf_xyz.z();
  tf.getBasis().getRPY(eigenvec[3], eigenvec[4], eigenvec[5], 1);
}

Eigen::VectorXd PoseKalmanFilter::fromTFTransformToEigen(const tf::Transform& tf)
{
  Eigen::VectorXd eigenvec = Eigen::VectorXd(6);
  tf::Vector3 tf_xyz = tf.getOrigin();
  eigenvec[0] = tf_xyz.x();
  eigenvec[1] = tf_xyz.y();
  eigenvec[2] = tf_xyz.z();
  tf.getBasis().getRPY(eigenvec[3], eigenvec[4], eigenvec[5], 1);
  return eigenvec;
}

///////////////////////////////////////////////////////////////////////////////
Eigen::VectorXd PoseKalmanFilter::getRawState() {return x_hat_;}

tf::Transform PoseKalmanFilter::getPoseState()
{
  tf::Vector3 x_pos(x_hat_[0], x_hat_[1], x_hat_[2]);
  tf::Quaternion x_quat;
  x_quat.setRPY(x_hat_[3], x_hat_[4], x_hat_[5]);
  return tf::Transform(x_quat, x_pos);
}

double PoseKalmanFilter::getTime() {return t_;}
} // namespace viso2_slam