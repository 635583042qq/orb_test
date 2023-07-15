#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
#include <opencv2/core/eigen.hpp>
#include <iostream>
#include "orb_extractor1.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <chrono>
#include "vfc.h"
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Geometry>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include "opencv2/imgcodecs/legacy/constants_c.h"
#include <g2o/types/slam3d/types_slam3d.h>
#include "g2o/core/block_solver.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/core/robust_kernel_impl.h"
// #include "g2o/solvers/linear_solver_eigen.h"
// #include "g2o/solvers/linear_solver_dense.h"
// #include "g2o/types/types_six_dof_expmap.h"
// #include "g2o/types/types_seven_dof_expmap.h"
using namespace std;
using namespace cv;
using namespace XIAOC;

int nfeatures = 1000;
int nlevels = 1;
float fscaleFactor = 1.0;
float fIniThFAST = 40;
float fMinThFAST = 8;
// cv::Mat imagesrc = imread( "/home/lyh/orb_test/1.jpg");
// cv::Mat imagecur = imread( "/home/lyh/orb_test/2.jpg");
cv::Mat imagesrc;
cv::Mat imagecur;
bool firstflag = true;
// float Z = 0.39;
float Z = 0.46;
// float Z = 0.75;
float fx = 614.7144165039062;
float fy = 614.6425170898438;
float cx = 322.79010009765625;
float cy = 245.56085205078125;
cv::Mat K_Matrix = (Mat_<float>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
cv::Mat distMatrix = (Mat_<float>(5, 1) << 0, 0, 0, 0, 0);
Eigen::Matrix<float, 3, 3> R_result = Eigen::Matrix3f::Identity();
Eigen::Matrix<float, 3, 1> t_result;
Eigen::Matrix<double, 3, 3> R_resultBA = Eigen::Matrix3d::Identity();
Eigen::Matrix<double, 3, 1> t_resultBA;
const float thHuber2D = sqrt(5.99);
bool useoptimize = false;


Mat R = cv::Mat::eye(3, 3, CV_64F);
Mat t = cv::Mat::zeros(3, 1, CV_64F);


Eigen::Matrix4d bundleAdjustment(
    const vector<Point3f> points_3d,
    const vector<Point2f> points_2d,
    const Mat &K,
    Mat &R, Mat &t);

    cv::Point3f transformPoint(const cv::Point3f& point, const cv::Mat& rotation, const cv::Mat& translation) {
    // 将点坐标转换为齐次坐标
    cv::Mat pointHomogeneous = (cv::Mat_<float>(4, 1) << point.x, point.y, point.z, 1.0f);

    // 通过旋转矩阵和平移向量进行变换
    cv::Mat transformedPoint = rotation * pointHomogeneous + translation;

    // 将齐次坐标转换为三维坐标
    cv::Point3f transformedPoint3D(transformedPoint.at<float>(0), transformedPoint.at<float>(1), transformedPoint.at<float>(2));

    return transformedPoint3D;
}

void imageCallback(const sensor_msgs::ImageConstPtr &msg)
{
    if (firstflag)
    {
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        imagesrc = cv_ptr->image;
        firstflag = false;

        // cout<<"first"<<endl;
    }
    
    // 将ROS图像消息转换为OpenCV图像格式
    cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    imagecur = cv_ptr->image;
    Mat grayImgsrc, mask;
    cv::cvtColor(imagesrc, grayImgsrc, 6);
    // orb extractor initialize
    chrono::steady_clock::time_point time1 = chrono::steady_clock::now();
    ORBextractor *pORBextractor;
    pORBextractor = new ORBextractor(nfeatures, fscaleFactor, nlevels, fIniThFAST, fMinThFAST);
    Mat srcdesc;
    vector<KeyPoint> srckps;
    (*pORBextractor)(grayImgsrc, mask, srckps, srcdesc);
    Mat grayImgcur, maskcur;
    cvtColor(imagecur, grayImgcur, 6);
    Mat curdesc;
    vector<KeyPoint> curkps;

    (*pORBextractor)(grayImgcur, maskcur, curkps, curdesc);
    BFMatcher matcher_bf(NORM_HAMMING, true); // 使用汉明距离度量二进制描述子，允许交叉验证
    vector<DMatch> Matches_bf;
    matcher_bf.match(srcdesc, curdesc, Matches_bf);
    assert(Matches_bf.size() > 0);
    Mat BF_img;
    drawMatches(imagesrc, srckps, imagecur, curkps, Matches_bf, BF_img);
    resize(BF_img, BF_img, Size(2 * imagesrc.cols, imagesrc.rows));
    // cout << Matches_bf.size() << endl;
    putText(BF_img, "Brute Force Matches", Point(20, 20), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 255), 2);
    // imshow("Brute Force Matches", BF_img);
    // 数据格式预处理
    vector<Point2f> X;
    vector<Point2f> Y;
    X.clear();
    Y.clear();
    // 将Matches_bf里的匹配点对分别放到X,Y 向量里，约5行代码，参考OpenCV DMatch类
    for (int i = 0; i < Matches_bf.size(); i++)
    {
        int index1 = Matches_bf.at(i).queryIdx;
        int index2 = Matches_bf.at(i).trainIdx;
        X.push_back(srckps.at(index1).pt);
        Y.push_back(curkps.at(index2).pt);
    }
    // 调用VFC主函数
    // t = (double)getTickCount();
    VFC myvfc;
    myvfc.setData(X, Y);
    myvfc.optimize();
    vector<int> matchIdx = myvfc.obtainCorrectMatch();
    // t = 1000 * ((double)getTickCount() - t) / getTickFrequency();

    // 筛选正确的匹配
    std::vector<DMatch> Matches_VFC;
    for (unsigned int i = 0; i < matchIdx.size(); i++)
    {
        int idx = matchIdx[i];
        Matches_VFC.push_back(Matches_bf[idx]);
    }
    if(Matches_VFC.size()==0) {
        cout<<"no match!!!!!!!!!!!!"<<endl;
        cout<<srckps.size()<<endl;
        cout<<curkps.size()<<endl;
        }
    // 数据准备
    vector<cv::Point3f> vSrc_mappoints;
    vector<cv::Point2f> vCur;
    for (int i = 0; i < Matches_VFC.size(); ++i)
    {
        int index1 = Matches_VFC[i].queryIdx;
        int index2 = Matches_VFC[i].trainIdx;
        float x1 = srckps[index1].pt.x;
        float y1 = srckps[index1].pt.y;

        float z1 = Z;
        float x = z1 * (x1 - cx) / fx;
        float y = z1 * (y1 - cy) / fy;
        float z = z1;

        cv::Point3f v1(x, y, z);

        float x2 = curkps[index2].pt.x;
        float y2 = curkps[index2].pt.y;
        cv::Point2f v2(x2, y2);
        vCur.emplace_back(v2);
        // cout<<"v1"<<endl;
        // v1=transformPoint( v1, R, t) ;
        // v1.z=Z;

        vSrc_mappoints.push_back(v1);
        
        // cout << vSrc_mappoints[i] << endl;
    }

    // 位姿估计
    // cout<<"before pnp"<<endl;
    // Mat RR;
    // Mat RPNP,tPNP;
    // cv::solvePnP(vSrc_mappoints, vCur, K_Matrix, distMatrix, RPNP, tPNP,true);//Rwc twc 初始位姿使用上一帧位姿
    // cout<<"R="<<endl<<R<<endl;
    //  cout<<"t="<<endl<<t<<endl;
    //  cout<<"t="<<endl<<t.at<double> ( 0,0 )<<t.at<double> ( 1,0 ) <<t.at<double> ( 2,0 ) <<endl;

    // // Mat Rvec;
    // // Mat_<float> Tvec;

    // // R.convertTo(Rvec, CV_32F); // 旋转向量转换格式
    // // t.convertTo(Tvec, CV_32F); // 平移向量转换格式
    // // Mat_<float> rotMat(3, 3);
    // // Rodrigues(Rvec, rotMat);
    // Rodrigues(RPNP, RR);
    // cout<<"after pnp"<<endl;
    // R = RR*R;
    // t  += tPNP; 

    // cout<<R<<endl;
    // cout << t<<endl;




    //原来的
    Mat RR;
    Mat RPNP,tPNP;
    cv::solvePnP(vSrc_mappoints, vCur, K_Matrix, distMatrix, RPNP, tPNP);//Rwc twc
    Mat Rvec;
    Mat_<float> Tvec;
    RPNP.convertTo(Rvec, CV_32F); // 旋转向量转换格式
    tPNP.convertTo(Tvec, CV_32F); // 平移向量转换格式
    Mat_<float> rotMat(3, 3);
    Rodrigues(Rvec, rotMat);
    Rodrigues(RPNP, RR);//这个RR是给优化用的

    Eigen::Matrix<float, 3, 3> m1;
    Eigen::Matrix<float, 3, 1> m2;
    cv::cv2eigen(rotMat,m1);
    cv::cv2eigen(Tvec,m2);
    R_result = m1 * R_result;
    // R_result =   R_result* m1;
    t_result += m2;
    std::cout << "beforeBA R" << R_result << std::endl;
    std::cout << "beforeBA t" << t_result << std::endl;

    // 优化
    Eigen::Matrix4d TBA = bundleAdjustment(vSrc_mappoints, vCur, K_Matrix, RR, t);//
    Eigen::Matrix3d BAR = TBA.block<3, 3>(0, 0);
    Eigen::Vector3d BAt = TBA.block<3, 1>(0, 3);
    R_resultBA = BAR * R_resultBA;
    t_resultBA += BAt;
    std::cout << "R_result = " << R_resultBA << std::endl;
    std::cout << "t_result = " << t_resultBA << std::endl;





    // cout<<m1<<endl;
    // cout<<"caculat ok"<<endl;
    // 优化
    {
    // if (useoptimize)
    // {
    //     typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolverType;
    //     typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;

    //     std::unique_ptr<BlockSolverType::LinearSolverType> linearSolver(new LinearSolverType());
    //     std::unique_ptr<BlockSolverType> solver_ptr(new BlockSolverType(std::move(linearSolver)));
    //     g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(std::move(solver_ptr));

    //     g2o::SparseOptimizer optimizer;
    //     optimizer.setAlgorithm(solver);

    //     // cout<<"BAinit ok"<<endl;
    //     // 设置pose1的估计值
    //     g2o::VertexSE3Expmap *pose1 = new g2o::VertexSE3Expmap(); // camera pose
    //     Eigen::Matrix3d R_mat1 = Eigen::Matrix3d::Identity();
    //     pose1->setId(0);
    //     pose1->setEstimate(g2o::SE3Quat(R_mat1, Eigen::Vector3d(0, 0, 0)));
    //     pose1->setFixed(0);
    //     optimizer.addVertex(pose1);
    //     // 设置pose2的估计值
    //     g2o::VertexSE3Expmap *pose2 = new g2o::VertexSE3Expmap(); // camera pose
    //     Eigen::Matrix3d R_mat2 = m1.cast<double>();
    //     pose2->setId(1);
    //     pose2->setEstimate(g2o::SE3Quat(R_mat2.transpose(), R_mat2.transpose() * Eigen::Vector3d(-m2[0], -m2[1], -m2[2])));
    //     // pose2->setEstimate ( g2o::SE3Quat (R_mat2,Eigen::Vector3d ( m2[0], m2[1], m2[2]) ));
    //     optimizer.addVertex(pose2);
    //     //  cout<<"poseset ok"<<endl;
    //     // 设置point的估计值
    //     for (int i = 0; i < vSrc_mappoints.size(); i++)
    //     {
    //         g2o::VertexPointXYZ *point = new g2o::VertexPointXYZ();
    //         Eigen::Vector3d epoint(vSrc_mappoints[i].x, vSrc_mappoints[i].y, vSrc_mappoints[i].z);
    //         point->setEstimate(epoint);
    //         point->setId(2 + i);
    //         point->setFixed(true);
    //         point->setMarginalized(true);
    //         optimizer.addVertex(point);

    //         // 边
    //         Eigen::Vector2d measurement1;                               // 第一帧图像中特征点的观测值
    //         Eigen::Matrix2d information1 = Eigen::Matrix2d::Identity(); // 第一帧图像中特征点的信息矩阵
    //         int index1 = Matches_VFC[i].queryIdx;
    //         measurement1 << srckps[index1].pt.x, srckps[index1].pt.y;
    //         g2o::EdgeSE3ProjectXYZ *e1 = new g2o::EdgeSE3ProjectXYZ();
    //         e1->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(2 + i)));
    //         e1->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
    //         e1->setMeasurement(measurement1);
    //         e1->setInformation(Eigen::Matrix2d::Identity());

    //         Eigen::Vector2d measurement2;                               // 第一帧图像中特征点的观测值
    //         Eigen::Matrix2d information2 = Eigen::Matrix2d::Identity(); // 第一帧图像中特征点的信息矩阵
    //         int index2 = Matches_VFC[i].trainIdx;
    //         measurement2 << srckps[index2].pt.x, srckps[index2].pt.y;
    //         g2o::EdgeSE3ProjectXYZ *e2 = new g2o::EdgeSE3ProjectXYZ();
    //         e2->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(2 + i)));
    //         e2->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(1)));
    //         e2->setMeasurement(measurement2);
    //         e2->setInformation(Eigen::Matrix2d::Identity());

    //         g2o::RobustKernelHuber *rk1 = new g2o::RobustKernelHuber;
    //         g2o::RobustKernelHuber *rk2 = new g2o::RobustKernelHuber;
    //         e1->setRobustKernel(rk1);
    //         e2->setRobustKernel(rk2);
    //         // 这里的重投影误差，自由度为2，所以这里设置为卡方分布中自由度为2的阈值，如果重投影的误差大约大于1个像素的时候，就认为不太靠谱的点了，
    //         // 核函数是为了避免其误差的平方项出现数值上过大的增长
    //         rk1->setDelta(thHuber2D);
    //         rk2->setDelta(thHuber2D);
    //         e2->fx = fx;
    //         e2->fy = fy;
    //         e2->cx = cx;
    //         e2->cy = cy;
    //         e1->fx = fx;
    //         e1->fy = fy;
    //         e1->cx = cx;
    //         e1->cy = cy;
    //         optimizer.addEdge(e1);
    //         optimizer.addEdge(e2);
    //     }

    //     //  cout<<"mappoint ok"<<endl;
    //     // 执行优化
    //     optimizer.initializeOptimization();
    //     optimizer.optimize(20);
    //     //  cout<<"optimize ok"<<endl;
    //     g2o::VertexSE3Expmap *vSE3 = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(1));
    //     g2o::SE3Quat SE3quat = vSE3->estimate();

    //     Eigen::Matrix4d BAT = SE3quat.to_homogeneous_matrix();
    //     Eigen::Matrix3d BAR = BAT.block<3, 3>(0, 0);
    //     Eigen::Vector3d BAt = BAT.block<3, 1>(0, 3);
    //     R_resultBA = BAR * R_resultBA;
    //     t_resultBA = BAt + t_resultBA;
    //     std::cout << "t_result = " << BAt << std::endl;
    // }

    }
    // 更新

    // 打印测试
    //  std::cout << "beforeBA R" << R_result << std::endl;
    //  std::cout << "beforeBA t" << t_result << std::endl;
    //  std::cout << "R_result = " << R_resultBA << std::endl;
    //  std::cout << "t_result = " << t_resultBA << std::endl;
    //  std::cout << "r_resultbefore = " << m1 << std::endl;
    //  std::cout << "t_resultbefore = " << m2 << std::endl;
    //  std::cout << "rBA = " << RBA << std::endl;
    //  std::cout << "tBA = " << tBA << std::endl;

    imagesrc = imagecur;
}



int main(int argc, char **argv)
{
    ros::init(argc, argv, "image_subscriber");
    ros::NodeHandle nh;

    std::string image_topic = "/camera/color/image_raw"; // 替换为实际的图像话题名称
    ros::Subscriber sub = nh.subscribe(image_topic, 1000, imageCallback);

    ros::spin();

    return 0;
}

Eigen::Matrix4d bundleAdjustment(
    const vector<Point3f> points_3d,
    const vector<Point2f> points_2d,
    const Mat &K,
    Mat &R, Mat &t)
{
    // 初始化g2o
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolverType;
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;

    std::unique_ptr<BlockSolverType::LinearSolverType> linearSolver(new LinearSolverType());
    std::unique_ptr<BlockSolverType> solver_ptr(new BlockSolverType(std::move(linearSolver)));
    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(std::move(solver_ptr));

    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    // cout<<"BA ready"<<endl;

    // vertex
    g2o::VertexSE3Expmap *pose = new g2o::VertexSE3Expmap(); // camera pose
    Eigen::Matrix3d R_mat;
    R_mat << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2),
        R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2),
        R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2);

    pose->setId(0);

    pose->setEstimate(g2o::SE3Quat(
        R_mat,
        Eigen::Vector3d(t.at<double>(0, 0), t.at<double>(1, 0), t.at<double>(2, 0))));
    // pose->setEstimate ( g2o::SE3Quat (
    //             R_mat.transpose(),
    //         -R_mat.transpose()*Eigen::Vector3d ( t.at<double> ( 0,0 ), t.at<double> ( 1,0 ), t.at<double> ( 2,0 ) )
    //                     ) );
    optimizer.addVertex(pose);
    // cout<<"Vertex ready"<<endl;
    int index = 1;
    for (const Point3f p : points_3d) // landmarks
    {
        g2o::VertexPointXYZ *point = new g2o::VertexPointXYZ();
        point->setId(index++);
        point->setEstimate(Eigen::Vector3d(p.x, p.y, p.z));
        point->setMarginalized(true); // g2o 中必须设置 marg 参见第十讲内容
        optimizer.addVertex(point);
    }
    // cout<<"point ready"<<endl;
    // parameter: camera intrinsics
    g2o::CameraParameters *camera = new g2o::CameraParameters((fx, fy), Eigen::Vector2d(cx, cy), 0);
    // g2o::CameraParameters* camera = new g2o::CameraParameters (fx,fy,cx,cy,0);

    camera->setId(0);
    optimizer.addParameter(camera);

    // edges
    index = 1;
    for (const Point2f p : points_2d)
    {
        g2o::EdgeProjectXYZ2UV *edge = new g2o::EdgeProjectXYZ2UV();
        edge->setId(index);
        edge->setVertex(0, dynamic_cast<g2o::VertexPointXYZ *>(optimizer.vertex(index)));
        // edge->setVertex ( 1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)) );
        edge->setVertex(1, pose);
        edge->setMeasurement(Eigen::Vector2d(p.x, p.y));
        edge->setParameterId(0, 0);
        edge->setInformation(Eigen::Matrix2d::Identity());
        optimizer.addEdge(edge);
        index++;
    }
    //  cout<<"edge ready"<<endl;
    // chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    optimizer.setVerbose(false);
    optimizer.initializeOptimization();
    optimizer.optimize(20);
    // chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    // chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>> ( t2-t1 );
    // cout<<"optimization costs time: "<<time_used.count() <<" seconds."<<endl;

    // cout<<endl<<"after optimization:"<<endl;

    // g2o::VertexSE3Expmap* poseBA = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));

    Eigen::Matrix4d T = pose->estimate().to_homogeneous_matrix();
    // Eigen::Matrix4d T= poseBA->estimate().to_homogeneous_matrix()
    // cout<<"T="<<endl<<Eigen::Isometry3d ( pose->estimate() ).matrix() <<endl;
    return T;
}





