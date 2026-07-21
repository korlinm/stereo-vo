#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

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

// StereoSGBM tuning parameters (see cv::StereoSGBM::create docs for meaning).
constexpr int kMinDisparity = 0;
constexpr int kNumDisparities = 96;   // must be divisible by 16
constexpr int kBlockSize = 11;        // odd, typically 5-11
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

}  // namespace

int main() {
    const std::string sequence_dir = "/home/collin/datasets/kitti/sequences/00";

    const StereoCalibration calib = loadCalibration(sequence_dir + "/calib.txt");
    std::cout << "Parsed calibration: f=" << calib.focal_length
              << " cx=" << calib.cx << " cy=" << calib.cy
              << " cx_right=" << calib.cx_right << " tx=" << calib.tx << "\n";

    const cv::Mat left = cv::imread(frameImagePath(sequence_dir, "image_0", 0),
                                     cv::IMREAD_GRAYSCALE);
    const cv::Mat right = cv::imread(frameImagePath(sequence_dir, "image_1", 0),
                                      cv::IMREAD_GRAYSCALE);

    if (left.empty() || right.empty()) {
        std::cerr << "ERROR: Could not load images\n";
        return -1;
    }

    const cv::Mat disparity = computeDisparity(left, right);
    const cv::Mat points_3d = disparityToPointCloud(disparity, calib);
    printFirstValidPoint(points_3d);

    const cv::Mat depth_image = buildDepthImage(points_3d);
    const std::string depth_image_path = "depth_frame0.png";
    cv::imwrite(depth_image_path, depth_image);
    std::cout << "Saved depth visualization to " << depth_image_path << "\n";

    // Detect ORB features independently in the left image of two consecutive
    // frames. Nothing is compared between them yet - that's the next step.
    const cv::Mat left_t1 = cv::imread(frameImagePath(sequence_dir, "image_0", 1),
                                        cv::IMREAD_GRAYSCALE);
    const OrbFeatures features_t0 = detectOrbFeatures(left);
    const OrbFeatures features_t1 = detectOrbFeatures(left_t1);
    std::cout << "Frame 0: detected " << features_t0.keypoints.size()
              << " ORB keypoints\n";
    std::cout << "Frame 1: detected " << features_t1.keypoints.size()
              << " ORB keypoints\n";

    const std::vector<cv::DMatch> matches = matchFeatures(features_t0, features_t1);
    printMatchStats(matches);

    // Next steps (see project task list):
    //   5. Build 3D (frame t) <-> 2D (frame t+1) correspondences
    //   6. solvePnPRansac -> incremental pose
    //   7. Compose incremental pose into running world pose

    return 0;
}
