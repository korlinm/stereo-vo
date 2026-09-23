#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <optional>
#include <stdexcept>

namespace {

// Calibration values needed to build the Q reprojection matrix, parsed at
// runtime from a sequence's calib.txt rather than hardcoded — different
// KITTI sequences (and definitely different datasets) have different
// focal lengths/baselines.
struct StereoCalibration {
    double focal_length;
    double cx;
    double cy;
    double cx_right;
    double tx;  // signed baseline in meters (focal length already divided out)
};

// calib.txt lines look like "P0: v0 v1 v2 ... v11" - a label token followed
// by the 12 values of a row-major flattened 3x4 projection matrix. This pulls
// out just the 12 numbers so callers can index into them.
std::vector<double> parseCalibLine(const std::string& line) {
    std::istringstream stream(line);
    std::string label;
    stream >> label;  // discard "P0:" / "P1:" etc.

    std::vector<double> values;
    double value;
    while (stream >> value) {
        values.push_back(value);
    }
    return values;
}

// Reads P0 (left/image_0) and P1 (right/image_1) from calib.txt. In each
// flattened 3x4 P matrix: index 0 = focal length, index 2 = cx, index 6 = cy,
// index 3 = Tx. KITTI bakes focal length into Tx (Tx = -focal_length *
// baseline_meters), so it's divided back out here - the same fix already
// verified against buildReprojectionMatrix()'s expectations.
StereoCalibration loadCalibration(const std::string& calib_path) {
    std::ifstream file(calib_path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open calib.txt at: " + calib_path);
    }

    std::vector<double> p0;
    std::vector<double> p1;
    std::string line;
    while (std::getline(file, line)) {
        if (line.rfind("P0:", 0) == 0) {
            p0 = parseCalibLine(line);
        } else if (line.rfind("P1:", 0) == 0) {
            p1 = parseCalibLine(line);
        }
    }

    constexpr size_t kExpectedValues = 12;
    if (p0.size() != kExpectedValues || p1.size() != kExpectedValues) {
        throw std::runtime_error("calib.txt missing P0/P1 or malformed");
    }

    StereoCalibration calib;
    calib.focal_length = p0[0];
    calib.cx = p0[2];
    calib.cy = p0[6];
    calib.cx_right = p1[2];
    calib.tx = p1[3] / calib.focal_length;
    return calib;
}

constexpr int kFrameNumberWidth = 6;

// Builds paths like "<sequence_dir>/image_0/000000.png".
std::string frameImagePath(const std::string& sequence_dir,
                            const std::string& camera_folder,
                            int frame_index) {
    std::ostringstream path;
    path << sequence_dir << "/" << camera_folder << "/"
         << std::setw(kFrameNumberWidth) << std::setfill('0') << frame_index
         << ".png";
    return path.str();
}

// Helper for loading image 
cv::Mat loadGrayscale(const std::string& path) {
    cv::Mat image = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        throw std::runtime_error("Could not load image: " + path);
    }
    return image;
}


// StereoSGBM tuning parameters (see cv::StereoSGBM::create docs for meaning).
constexpr int kMinDisparity = 0;
constexpr int kNumDisparities = 96;   // must be divisible by 16
constexpr int kBlockSize =  11;        // odd, typically 5-11
constexpr int kDisp12MaxDiff = 1;
constexpr int kPreFilterCap = 4;
constexpr int kUniquenessRatio = 10;
constexpr int kSpeckleWindowSize = 100;
constexpr int kSpeckleRange = 32;

constexpr float kMaxValidDepthMeters = 100.0f;

cv::Mat computeDisparity(const cv::Mat& left, const cv::Mat& right) {
    const cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(
        kMinDisparity,
        kNumDisparities,
        kBlockSize,
        8 * 3 * kBlockSize * kBlockSize,   // P1 smoothness penalty
        32 * 3 * kBlockSize * kBlockSize,  // P2 smoothness penalty
        kDisp12MaxDiff,
        kPreFilterCap,
        kUniquenessRatio,
        kSpeckleWindowSize,
        kSpeckleRange
    );

    cv::Mat disparity_raw;
    sgbm->compute(left, right, disparity_raw);

    // SGBM returns fixed-point disparity (16x scale); convert to real disparity values.
    cv::Mat disparity_float;
    disparity_raw.convertTo(disparity_float, CV_32F, 1.0 / 16.0);
    return disparity_float;
}

cv::Mat buildReprojectionMatrix(const StereoCalibration& calib) {
    // OpenCV's Q-matrix convention wants the signed baseline (-baseline_meters)
    // in the bottom-left slot, not focal_length * baseline. calib.tx already
    // has focal length divided back out by loadCalibration(). Skipping that
    // step silently inflates every depth by ~f (a 5m-deep scene comes out as
    // ~3600m) — verified by a scratch diagnostic build before this fix landed.
    return (cv::Mat_<double>(4, 4) <<
        1, 0, 0, -calib.cx,
        0, 1, 0, -calib.cy,
        0, 0, 0, calib.focal_length,
        0, 0, -1.0 / calib.tx, (calib.cx - calib.cx_right) / calib.tx
    );
}

cv::Mat disparityToPointCloud(const cv::Mat& disparity_float,
                               const StereoCalibration& calib) {
    const cv::Mat Q = buildReprojectionMatrix(calib);
    cv::Mat points_3d;
    cv::reprojectImageTo3D(disparity_float, points_3d, Q, true);
    return points_3d;
}

// Keypoints (pixel locations + orientation/scale) and their matching binary
// descriptors, bundled together since one is meaningless without the other.
struct OrbFeatures {
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
};

// Cap on how many keypoints ORB keeps (ranked by corner strength). KITTI
// frames are 1242x375; a few thousand keeps matching fast without starving
// later steps (PnP) of correspondences.
constexpr int kOrbMaxFeatures = 2000;

OrbFeatures detectOrbFeatures(const cv::Mat& image) {
    const cv::Ptr<cv::ORB> orb = cv::ORB::create(kOrbMaxFeatures);

    OrbFeatures features;
    orb->detectAndCompute(image, cv::noArray(), features.keypoints,
                           features.descriptors);
    return features;
}

// Matches ORB descriptors from one frame to the next. Each cv::DMatch pairs
// a query-frame keypoint with a train-frame keypoint via their descriptors'
// Hamming distance (lower = more similar); crossCheck rejects any pair that
// isn't each other's mutual best match, discarding a lot of ambiguous
// matches (e.g. repetitive textures) for free.
std::vector<cv::DMatch> matchFeatures(const OrbFeatures& features_query,
                                       const OrbFeatures& features_train) {
    constexpr bool kCrossCheck = true;
    const cv::BFMatcher matcher(cv::NORM_HAMMING, kCrossCheck);

    std::vector<cv::DMatch> matches;
    matcher.match(features_query.descriptors, features_train.descriptors,
                   matches);
    return matches;
}

void printMatchStats(const std::vector<cv::DMatch>& matches) {
    if (matches.empty()) {
        std::cout << "No matches found\n";
        return;
    }

    float min_distance = matches[0].distance;
    float max_distance = matches[0].distance;
    float total_distance = 0.0f;
    for (const cv::DMatch& match : matches) {
        min_distance = std::min(min_distance, match.distance);
        max_distance = std::max(max_distance, match.distance);
        total_distance += match.distance;
    }
    const float avg_distance = total_distance / static_cast<float>(matches.size());

    std::cout << "Matched " << matches.size() << " keypoints between frames\n";
    std::cout << "  Hamming distance: min=" << min_distance
              << " max=" << max_distance << " avg=" << avg_distance << "\n";
}

void printFirstValidPoint(const cv::Mat& points_3d) {
    for (int row = 0; row < points_3d.rows; ++row) {
        for (int col = 0; col < points_3d.cols; ++col) {
            const cv::Vec3f point = points_3d.at<cv::Vec3f>(row, col);
            // Valid point: in front of the camera and within a sane depth range.
            if (point[2] > 0 && point[2] < kMaxValidDepthMeters) {
                std::cout << "Valid pixel at (" << col << ", " << row << ")\n";
                std::cout << "  X = " << point[0] << " m\n";
                std::cout << "  Y = " << point[1] << " m\n";
                std::cout << "  Z = " << point[2] << " m\n";
                return;
            }
        }
    }
    std::cout << "No valid 3D points found - check image loading or Q matrix\n";
}

// Builds a viewable grayscale image from a point cloud's depth (Z) values:
// close = bright, far = dark, invalid/no-data pixels = black.
cv::Mat buildDepthImage(const cv::Mat& points_3d) {
    cv::Mat depth_image(points_3d.rows, points_3d.cols, CV_8UC1, cv::Scalar(0));

    for (int row = 0; row < points_3d.rows; ++row) {
        for (int col = 0; col < points_3d.cols; ++col) {
            const float depth = points_3d.at<cv::Vec3f>(row, col)[2];
            if (depth > 0 && depth < kMaxValidDepthMeters) {
                const float brightness = 255.0f * (1.0f - depth / kMaxValidDepthMeters);
                depth_image.at<uchar>(row, col) = static_cast<uchar>(brightness);
            }
        }
    }
    return depth_image;
}

bool isValidDepth(float z) {
    return z > 0 && z < kMaxValidDepthMeters;
}

struct PointCorrespondences {
    std::vector<cv::Point3f> object_points;
    std::vector<cv::Point2f> image_points;
};
// Keypoints on depth discontinuities get foreground depth bleeding into
// background pixels (SGBM edge fattening). Reject any whose neighborhood
// depth varies by more than this fraction of its own depth.
constexpr int kDepthCheckRadius = 2;  // 5x5 window
constexpr float kMaxRelativeDepthSpread = 0.1f;

bool isOnDepthEdge(const cv::Mat& points_3d, int row, int col, float center_z) {
    if (row < kDepthCheckRadius || row >= points_3d.rows - kDepthCheckRadius ||
        col < kDepthCheckRadius || col >= points_3d.cols - kDepthCheckRadius) {
        return true;  // can't check the window, so don't trust it
    }
    float min_z = center_z;
    float max_z = center_z;
    for (int dr = -kDepthCheckRadius; dr <= kDepthCheckRadius; ++dr) {
        for (int dc = -kDepthCheckRadius; dc <= kDepthCheckRadius; ++dc) {
            const float z = points_3d.at<cv::Vec3f>(row + dr, col + dc)[2];
            if (!isValidDepth(z)) {
                return true;  // invalid neighbor usually means an edge or occlusion
            }
            min_z = std::min(min_z, z);
            max_z = std::max(max_z, z);
        }
    }
    return (max_z - min_z) / center_z > kMaxRelativeDepthSpread;
}

PointCorrespondences buildCorrespondences(  const std::vector<cv::DMatch>& matches,
                                            const OrbFeatures& features_t0,
                                            const OrbFeatures& features_t1,
                                            const cv::Mat& points_3d) {
    PointCorrespondences corr;
    for(const cv::DMatch& match : matches) {
        const cv::Point2f keypoint_t0 = features_t0.keypoints[match.queryIdx].pt;

        const int col = cvRound(keypoint_t0.x);
        const int row = cvRound(keypoint_t0.y);

        if(col < 0 || col >= points_3d.cols || row < 0 || row >= points_3d.rows){
            continue;
        }

        const cv::Vec3f point = points_3d.at<cv::Vec3f>(row, col);
        if(!isValidDepth(point[2])) {
            continue;
        }

        if (isOnDepthEdge(points_3d, row, col, point[2])) {
            continue;
        }

        corr.object_points.push_back(cv::Point3f(point[0], point[1], point[2]));
        corr.image_points.push_back(features_t1.keypoints[match.trainIdx].pt);

    }

    return corr;
}

struct Pose {
    cv::Mat rotation; //3x3
    cv::Mat translation;  //3x1, in meters
};

// Standard pinhole intrinsic matrix: focal length on the diagonal, principal
// point (cx, cy) in the last column. solvePnP needs this to know how 3D
// points map to pixel coordinates.
cv::Mat buildIntrinsicMatrix(const StereoCalibration& calib){
    return (cv::Mat_<double>(3, 3) <<
        calib.focal_length, 0, calib.cx,
        0, calib.focal_length, calib.cy,
        0, 0, 1
    );
}

constexpr int kRansacIterations = 200;
constexpr float kRansacReprojectionErrorPx = 1.5f;
constexpr double kRansacConfidence = 0.999;

// Estimates the camera's rigid-body motion from the 3D<->2D correspondences.
// RANSAC matters here because your correspondences will contain outliers -
// mismatched ORB features, points on moving cars, depth noise near object
// edges - and a handful of bad correspondences can wreck a plain least-
// squares PnP solve.
std::optional<Pose> estimatePose(const PointCorrespondences& corr, const cv::Mat& camera_matrix){
    const cv::Mat dist_coeffs; //empty matrix becuase there is no distortion
                               //in KITTI images, since they are pre-rectified
    cv::Mat rvec;
    cv::Mat tvec;
    
    constexpr size_t kMinCorrespondences = 6;
    if(corr.object_points.size() < kMinCorrespondences) {
        return std::nullopt;
    }
    
    cv::Mat inliers;
    constexpr bool kUseExtrinsicGuess = false;
    const bool success = cv::solvePnPRansac(
        corr.object_points, corr.image_points, camera_matrix, dist_coeffs,
        rvec, tvec, kUseExtrinsicGuess, kRansacIterations,
        kRansacReprojectionErrorPx, kRansacConfidence, inliers);

    // std::cout << "  inliers: " << inliers.rows << " / "
            // << corr.object_points.size() << "\n";
    std::vector<float> inlier_depths;
    for (int k = 0; k < inliers.rows; ++k) {
        inlier_depths.push_back(corr.object_points[inliers.at<int>(k)].z);
    }
    float median_depth = 0.0f;
    if (!inlier_depths.empty()) {
        auto middle = inlier_depths.begin() + inlier_depths.size() / 2;
        std::nth_element(inlier_depths.begin(), middle, inlier_depths.end());
        median_depth = *middle;
    }
    std::cout << "  inliers: " << inliers.rows << " / " << corr.object_points.size()
            << "  median depth: " << median_depth << " m\n";
            
    if(!success) {
        return std::nullopt;
    }

    cv::Mat rotation;
    cv::Rodrigues(rvec, rotation);

    return Pose{rotation, tvec};

}

// Full 4x4 homogeneous transform [R t; 0 0 0 1] — the form matrix
// multiplication needs for composing consecutive poses.
cv::Mat toHomogeneous(const Pose& pose){
    cv::Mat T = cv::Mat::eye(4, 4, CV_64F);

    cv::Mat(pose.rotation).copyTo(T(cv::Rect(0,0,3,3)));
    cv::Mat(pose.translation).copyTo(T(cv::Rect(3,0,1,3)));

    return T;
}
// Inverts a rigid-body transform. Rotation matrices are orthonormal, so
// R^-1 = R^T (no expensive general inverse needed); translation inverts as
// -R^T * t.

Pose invertPose(const Pose& pose){
    Pose inv;
    cv::Mat R = cv::Mat(pose.rotation).t();          // R^T
    inv.rotation = R;
    inv.translation = -R * cv::Mat(pose.translation); //-R^T * t

    return inv;
}

// Runs the VO pipeline across a sequence of frames, composing each
// frame-to-frame pose into a running world-frame trajectory. trajectory[i]
// is the camera's world pose after processing frame i (trajectory[0] is
// identity, since frame 0 defines the world origin).
std::vector<cv::Mat> estimateTrajectory(const std::string& sequence_dir,
                                         const StereoCalibration& calib,
                                         const cv::Mat& camera_matrix,
                                         int num_frames) {
    std::vector<cv::Mat> trajectory;
    cv::Mat world_pose = cv::Mat::eye(4, 4, CV_64F);
    trajectory.push_back(world_pose.clone());

    cv::Mat left_prev = loadGrayscale(frameImagePath(sequence_dir, "image_0", 0));
    cv::Mat right_prev = loadGrayscale(frameImagePath(sequence_dir, "image_1", 0));

    cv::Mat disparity_prev = computeDisparity(left_prev, right_prev);
    cv::Mat points_3d_prev = disparityToPointCloud(disparity_prev, calib);
    OrbFeatures features_prev = detectOrbFeatures(left_prev);

    Pose last_pose{cv::Mat::eye(3, 3, CV_64F), cv::Mat::zeros(3, 1, CV_64F)};


    for (int i = 1; i < num_frames; ++i) {
        // 1. Load frame i's stereo pair.
        cv::Mat left_curr = loadGrayscale(frameImagePath(sequence_dir, "image_0", i));
        cv::Mat right_curr = loadGrayscale(frameImagePath(sequence_dir, "image_1", i));

        // 2. Detect ORB features in frame i.
        OrbFeatures features_curr = detectOrbFeatures(left_curr);

        // 3. Match frame i-1 against frame i.
        std::vector<cv::DMatch> matches = matchFeatures(features_prev, features_curr);

        // 4. Build correspondences using the PREVIOUS frame's 3D points, since
        //    match.queryIdx indexes into features_prev.
        PointCorrespondences corr = buildCorrespondences(matches, features_prev,
                                                          features_curr, points_3d_prev);

        // 5. Estimate this step's incremental pose.
        // Pose pose = estimatePose(corr, camera_matrix);
        const std::optional<Pose> estimated = estimatePose(corr, camera_matrix);
        if(!estimated) {
            std::cerr << "Frame " << i << ": pose estimation failed ("
              << corr.object_points.size()
              << " correspondences), reusing previous motion\n";
        }
        const Pose pose = estimated ? *estimated : last_pose;
        last_pose = pose;

        // 6. Compose it onto the running world pose.
        world_pose = world_pose * toHomogeneous(invertPose(pose));
        trajectory.push_back(world_pose.clone());

        const cv::Mat camera_position = world_pose(cv::Rect(3, 0, 1, 3));
        std::cout << "Frame " << i << " position: " << camera_position.t() << "\n";

        // 7. Shift state forward for the next iteration.
        cv::Mat disparity_curr = computeDisparity(left_curr, right_curr);
        cv::Mat points_3d_curr = disparityToPointCloud(disparity_curr, calib);

        left_prev = left_curr;
        right_prev = right_curr;
        points_3d_prev = points_3d_curr;
        features_prev = features_curr;
    }

    return trajectory;
}

// poses/00.txt: one line per frame, 12 space-separated values = a flattened
// row-major 3x4 [R|t] ground-truth pose, in the same frame-0-as-origin world
// convention estimateTrajectory() accumulates its own poses in.
std::vector<cv::Mat> loadGroundTruthPoses(const std::string& poses_path) {
    std::ifstream file(poses_path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open poses file at: " + poses_path);
    }

    std::vector<cv::Mat> poses;
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream stream(line);
        cv::Mat pose = cv::Mat::eye(4, 4, CV_64F);
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 4; ++col) {
                stream >> pose.at<double>(row, col);
            }
        }
        poses.push_back(pose);
    }
    return poses;
}

// Euclidean distance between two world poses' (x, y, z) camera positions.
double positionError(const cv::Mat& estimated_world_pose, const cv::Mat& ground_truth_pose) {
    const cv::Mat estimated_position = estimated_world_pose(cv::Rect(3, 0, 1, 3));
    const cv::Mat ground_truth_position = ground_truth_pose(cv::Rect(3, 0, 1, 3));
    return cv::norm(estimated_position, ground_truth_position);
}

// Compares the motion between two consecutive frames rather than absolute
// position, so early drift doesn't inflate every later frame's error.
double relativeTranslationError(const cv::Mat& est_prev, const cv::Mat& est_curr,
                                const cv::Mat& gt_prev, const cv::Mat& gt_curr) {
    const cv::Mat est_delta = est_prev.inv() * est_curr;
    const cv::Mat gt_delta = gt_prev.inv() * gt_curr;
    const cv::Mat error = gt_delta.inv() * est_delta;
    return cv::norm(error(cv::Rect(3, 0, 1, 3)));
}

// Distance traveled between two consecutive world poses.
double stepLength(const cv::Mat& pose_prev, const cv::Mat& pose_curr) {
    return cv::norm(pose_curr(cv::Rect(3, 0, 1, 3)) - pose_prev(cv::Rect(3, 0, 1, 3)));
}

void saveTrajectoryKitti(const std::vector<cv::Mat>& trajectory, const std::string& path) {
    std::ofstream out(path);
    out << std::setprecision(9);
    for (const cv::Mat& T : trajectory) {
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 4; ++col) {
                out << T.at<double>(row, col) << ((row == 2 && col == 3) ? "\n" : " ");
            }
        }
    }
}

}  // namespace

int main() {
    constexpr int kNumFrames = 200;
    const std::string sequence_dir = "/home/collin/datasets/kitti/sequences/00";

    const StereoCalibration calib = loadCalibration(sequence_dir + "/calib.txt");
    std::cout << "Parsed calibration: f=" << calib.focal_length
              << " cx=" << calib.cx << " cy=" << calib.cy
              << " cx_right=" << calib.cx_right << " tx=" << calib.tx << "\n";

    const cv::Mat camera_matrix = buildIntrinsicMatrix(calib);

    const std::vector<cv::Mat> trajectory =
        estimateTrajectory(sequence_dir, calib, camera_matrix, kNumFrames);

    std::cout << "Estimated trajectory over " << trajectory.size() << " frames\n";

    const std::string poses_path = "/home/collin/datasets/kitti/poses/00.txt";
    const std::vector<cv::Mat> ground_truth = loadGroundTruthPoses(poses_path);

    double total_error = 0.0;
    double max_error = 0.0;
    for (size_t i = 0; i < trajectory.size(); ++i) {
        const double error = positionError(trajectory[i], ground_truth[i]);
        std::cout << "Frame " << i << " position error: " << error << " m\n";
        total_error += error;
        max_error = std::max(max_error, error);
    }
    const double avg_error = total_error / static_cast<double>(trajectory.size());
    std::cout << "Average position error: " << avg_error
              << " m, max: " << max_error << " m\n";

    double total_relative_error = 0.0;
    for (size_t i = 1; i < trajectory.size(); ++i) {
        total_relative_error += relativeTranslationError(
            trajectory[i - 1], trajectory[i], ground_truth[i - 1], ground_truth[i]);
    }
    
    std::cout << "Average per-frame relative translation error: "
            << total_relative_error / static_cast<double>(trajectory.size() - 1)
            << " m\n";

    std::cout << "\nStep lengths (estimated vs ground truth):\n";
    for (size_t i = 1; i < trajectory.size(); ++i) {
        const double est_step = stepLength(trajectory[i - 1], trajectory[i]);
        const double gt_step = stepLength(ground_truth[i - 1], ground_truth[i]);
        std::cout << "Frame " << i << ": est=" << est_step << " m  gt=" << gt_step
                << " m  ratio=" << est_step / gt_step << "\n";
    }

    saveTrajectoryKitti(trajectory, "est_00.txt");


    return 0;
}
