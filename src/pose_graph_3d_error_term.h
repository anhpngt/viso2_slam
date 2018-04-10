#ifndef _POSE_GRAPH_3D_ERROR_TERM_H_
#define _POSE_GRAPH_3D_ERROR_TERM_H_

#include <string>
#include <fstream>
#include <sstream>
#include <iostream> 
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <ceres/local_parameterization.h>

using namespace std;

namespace Optimizer
{
// 3D pose by Eigen 
struct Pose3d
{
  Eigen::Vector3d p;
  Eigen::Quaterniond q;

  Pose3d()
  {
      q.setIdentity();
      p.fill(0.);
  };

  // Pose3d(const Pose3d &_p):
  //       p(_p.p), q(_p.q), n(_p.n)
  // {};

  Pose3d(const Pose3d &_p):
        p(_p.p), q(_p.q)
  {};

  // Pose3d(const Eigen::Vector3d &_p, 
  //        const Eigen::Quaterniond &_q,
  //        const Eigen::Vector3d &_n):
  //       p(_p), q(_q), n(_n)
  // {};

  Pose3d(const Eigen::Vector3d &_p, const Eigen::Quaterniond &_q):
        p(_p), q(_q)
  {};

  Pose3d(const Eigen::Vector3d &t, const Eigen::Matrix3d &R): 
        p(t), q(Eigen::Quaterniond(R))
  {};

  Pose3d inverse() const
  {
    return Pose3d(q.conjugate()*(-1.*p), q.conjugate());
  }

  Pose3d operator *(const Pose3d& other) const 
  {
    Pose3d ret;
    ret.q = q*other.q;
    ret.p = (q*other.p)+p;
    return ret;
  }

  Eigen::Vector3d map (const Eigen::Vector3d& xyz) const 
  {
    return (q*xyz) + p;
  }

  inline const Eigen::Vector3d& translation() const {return p;}

  inline Eigen::Vector3d& translation() {return p;}

  inline const Eigen::Quaterniond& rotation() const {return q;}

  inline Eigen::Quaterniond& rotation() {return q;}

  friend ostream& operator<<(ostream& os, const Pose3d& ret) { 
	  os << "p = (" << ret.p.x() << "," << ret.p.y() << "," << ret.p.z() << ")," 
	     << "q = (" << ret.q.x() << "," << ret.q.y() << "," << ret.q.z() << "," << ret.q.w() << ")";  
      return os;  
  } 
};
typedef struct Pose3d Pose3d;

typedef vector<Pose3d, Eigen::aligned_allocator<Pose3d>> VectorofPoses;
typedef vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> VectorofNormalVectors;

// The constraint between two vertices in the pose graph. The constraint is the
// transformation from vertex id_begin to vertex id_end.
struct Constraint3d {
  int id_begin;
  int id_end;

  // The transformation that represents the pose of the end frame E w.r.t. the
  // begin frame B. In other words, it transforms a vector in the E frame to
  // the B frame.
  Pose3d t_be;

  // The inverse of the covariance matrix for the measurement. The order of the
  // entries are x, y, z, delta orientation.
  Eigen::Matrix<double, 6, 6> information;

  Constraint3d() 
  {
	  id_begin = -1;
	  id_end = -1;
      t_be.q.setIdentity();
      t_be.p.fill(0.);
	  information.setIdentity();
  };

  Constraint3d(int ib, int ie, Pose3d t_be_, Eigen::Matrix<double, 6, 6> inf):
      id_begin(ib), id_end(ie), t_be(t_be_), information(inf)
  {};

  Constraint3d(int ib, int ie, Pose3d t_be_):
      id_begin(ib), id_end(ie), t_be(t_be_)
  {
      information.setIdentity();
  };
};

typedef vector<Constraint3d, Eigen::aligned_allocator<Constraint3d>> VectorOfConstraints;

class PoseGraph3dErrorTerm 
{
 public:
  PoseGraph3dErrorTerm(const Pose3d& t_ab_measured,
                       const Eigen::Matrix<double, 6, 6>& sqrt_information)
      : t_ab_measured_(t_ab_measured), sqrt_information_(sqrt_information) {};

  template <typename T>
  bool operator()(const T* const p_a_ptr, const T* const q_a_ptr,
                  const T* const p_b_ptr, const T* const q_b_ptr,
                  T* residuals_ptr) const 
  {
    Eigen::Map<const Eigen::Matrix<T, 3, 1> > p_a(p_a_ptr);
    Eigen::Map<const Eigen::Quaternion<T> > q_a(q_a_ptr);

    Eigen::Map<const Eigen::Matrix<T, 3, 1> > p_b(p_b_ptr);
    Eigen::Map<const Eigen::Quaternion<T> > q_b(q_b_ptr);

	  // Compute inversion of b (T)
    Eigen::Quaternion<T> q_b_inverse = q_b.conjugate();
    Eigen::Matrix<T, 3, 1> p_b_inverse = q_b_inverse*(-p_b);

    // Compute the relative rotation between the two frames.
    Eigen::Quaternion<T> q_ab_estimated = q_b_inverse * q_a;

    // Represent the displacement between the two frames in the A frame.
    Eigen::Matrix<T, 3, 1> p_ab_estimated = q_b_inverse * p_a + p_b_inverse;

    // Compute the error between the two orientation estimates.
    Eigen::Quaternion<T> delta_q =
        t_ab_measured_.q.template cast<T>() * q_ab_estimated.conjugate();

    // Compute the residuals.
    // [ position         ]   [ delta_p          ]
    // [ orientation (3x1)] = [ 2 * delta_q(0:2) ]
    Eigen::Map<Eigen::Matrix<T, 6, 1> > residuals(residuals_ptr);
    residuals.template block<3, 1>(0, 0) =
        p_ab_estimated - t_ab_measured_.p.template cast<T>();
    residuals.template block<3, 1>(3, 0) = T(2.0) * delta_q.vec();

    // Scale the residuals by the measurement uncertainty.
    residuals.applyOnTheLeft(sqrt_information_.template cast<T>());

    return true;
  }

  static ceres::CostFunction* Create(const Pose3d& t_ab_measured,
                                     const Eigen::Matrix<double, 6, 6>& sqrt_information) 
  {
    return new ceres::AutoDiffCostFunction<PoseGraph3dErrorTerm, 6, 3, 4, 3, 4>(
                      new PoseGraph3dErrorTerm(t_ab_measured, sqrt_information));
  }

 private:
  // The measurement for the position of B relative to A in the A frame.
  const Pose3d t_ab_measured_;
  // The square root of the measurement information matrix.
  const Eigen::Matrix<double, 6, 6> sqrt_information_;
};

} // namespace Optimizer

#endif // _POSE_GRAPH_3D_ERROR_TERM_H_