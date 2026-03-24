#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>


int main() {
    //Load a stereo image pair

    cv::Mat left = cv::imread("0left.png", cv::IMREAD_GRAYSCALE);
    cv::Mat right = cv::imread("0right.png", cv::IMREAD_GRAYSCALE);

    if (left.empty() || right.empty())
    {
        std::cerr << "ERROR: Could not load images" << std::endl;
        return -1;
    }
        

    //for each frame pair (left_img, right_img)
    //1. compute disparity

    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(
        0,      // minDisparity
        96,     //numDisparities (must be divisible by 16)
        11,     // blockSize (odd number, typically 5-11)
        8  * 3 * 11 * 11, //P1 smoothness penalty
        32 * 3 * 11 * 11, //P2 smoothness penalty
        1,      // disp12MaxDiff
        4,      //preFilterCap
        10,     //uniquenessRatio
        100,    //speckeWindowSize
        32     //speckelRange
    );

    cv::Mat disparity_raw, disparity_float;
    sgbm->compute(left, right, disparity_raw);
    disparity_raw.convertTo(disparity_float, CV_32F, 1.0 / 16.0);
    

    double f    =  7.215377e+02;
    double cx   =  6.095593e+02;
    double cy   =  1.728540e+02;
    double Tx   = -3.875744e+02;  // P_rect_01[0][3] — already f*baseline in pixel units
    double cx_r =  6.095593e+02;  // cx of right camera (same here since rectified)

    cv::Mat Q = (cv::Mat_<double>(4, 4) <<
        1,  0,  0,       -cx,
        0,  1,  0,       -cy,
        0,  0,  0,        f,
        0,  0, -1.0/Tx,  (cx - cx_r) / Tx
    );
    // (cx - cx') / Tx — zero here since cx == cx_r
    //
    cv::Mat points3D;
    cv::reprojectImageTo3D(disparity_float, points3D, Q, true);

    //print sample 3d point to verify

    // int cxt = left.cols / 2;
    // int cyt = left.rows / 2;
    // cv::Vec3f pt = points3D.at<cv::Vec3f>(cyt, cxt);

    // std::cout << "Center pixel point:" << std::endl;
    // std::cout << "  X = " << pt[0] << " m" << std::endl;
    // std::cout << "  Y = " << pt[1] << " m" << std::endl;
    // std::cout << "  Z = " << pt[2] << " m" << std::endl;

    // if(pt[2] > 9999.0f) {
    //     std::cout << "  invalid disparity at this pixel" << std::endl;
    // }

    bool found = false;
    for (int row = 0; row < points3D.rows && !found; row++) {
        for (int col = 0; col < points3D.cols && !found; col++) {
            cv::Vec3f pt = points3D.at<cv::Vec3f>(row, col);
            // Valid point: Z is positive, finite, and under 100m
            if (pt[2] > 0 && pt[2] < 100.0f) {
                std::cout << "Valid pixel at (" << col << ", " << row << ")" << std::endl;
                std::cout << "  X = " << pt[0] << " m" << std::endl;
                std::cout << "  Y = " << pt[1] << " m" << std::endl;
                std::cout << "  Z = " << pt[2] << " m" << std::endl;
                found = true;
            }
        }
    }

    if (!found) {
        std::cout << "No valid 3D points found — check image loading or Q matrix" << std::endl;
    }


    //2. Detect + match features 
    //orb->detectAndCompute(left, keypoints, descriptors)
    //matcher->match(desc_prev, desc_curr, matches)

    //3. Build 3D-2D correspondences
    // for each match: prev 3d point -> curr2d point

    //4. solve pose
    //solvePnPRansac(pts3D,pts2Dk,K,dist,rvec,tvec)

    //5. Compose + print
    //T_world = T_world * deltaT
    //cout << T_world.translation() << endl
}


// int main(){
//     std::cout << "OpenCV version: " << CV_VERSION << std::endl;
//     cv::Mat img = cv::Mat::zeros(100,100, CV_8UC1);
//     std::cout << "Mat created: " << img.rows << "x" << img.cols << std::endl;
//     return 0;
// }