#include<ros/ros.h>
#include<iostream>
#include "orb_extractor1.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include<chrono>
#include "vfc.h"
using namespace cv;
using namespace std;
using namespace XIAOC;


int main(int argc,char **argv)
{
    ros::init(argc, argv, "main");
    // -----grid based orb extractor
    int nfeatures = 1000;
    int nlevels = 1;
    float fscaleFactor = 1.0;
    float fIniThFAST = 40;
    float fMinThFAST = 10;

    Mat imagesrc = imread( "/home/lyh/VSLAM_ws/src/picture/98.jpg");
    Mat imagecur = imread( "/home/lyh/VSLAM_ws/src/picture/99.jpg");

    Mat grayImgsrc,mask;
    cvtColor( imagesrc, grayImgsrc, 6 );

  
    // orb extractor initialize
    chrono::steady_clock::time_point time1 = chrono::steady_clock::now();
    ORBextractor *pORBextractor;
    pORBextractor = new ORBextractor( nfeatures, fscaleFactor, nlevels, fIniThFAST, fMinThFAST );
    cout << "ORBextractor initialize finished!" << endl;

    // orb extractor
    cout << "Extract orb descriptors..." << endl;
    Mat srcdesc;
    vector<KeyPoint> srckps;
    (*pORBextractor)( grayImgsrc, mask, srckps, srcdesc );


    chrono::steady_clock::time_point time2 = chrono::steady_clock::now();
    chrono::duration<double> time = chrono::duration_cast<chrono::duration<double>>(time2 - time1);
    cout<<"均匀化特征点计算用时："<<time.count()*1000<<"ms"<<endl;


    cout << "Extract orb descriptors finished!" << endl;
    cout << "The number of keypoints are = " << srckps.size() << endl;

    // draw keypoints in output image
    Mat outImgsrc;
    drawKeypoints( grayImgsrc, srckps, outImgsrc, cv::Scalar::all(-1), DrawMatchesFlags::DEFAULT );
    // imshow( "GridOrbKpsImg", outImgsrc );
    // waitKey( 0 );
    cout << "Finished! Congratulations!" << endl;


    // // ----original orb extractor for comparation
    // // orb initialization 
    // cout << "Using original orb extractor to extract orb descriptors for comparation." << endl;
    // Ptr<ORB> orb_ = ORB::create( 1000, 1.2f, 8, 19 );

    // // orb extract
    // vector<KeyPoint> orb_kps;
    // Mat orb_desc;
    // orb_->detectAndCompute( grayImgsrc, mask, orb_kps, orb_desc );

    // // draw keypoints in output image
    // Mat orbOutImg;
    // drawKeypoints( grayImgsrc, orb_kps, orbOutImg, cv::Scalar::all(-1), DrawMatchesFlags::DEFAULT );
    // imshow( "OrbKpsImg", orbOutImg );
    // waitKey(0);

    Mat grayImgcur,maskcur;
    cvtColor( imagecur, grayImgcur, 6 );

  
    // orb extractor initialize
    // chrono::steady_clock::time_point time1 = chrono::steady_clock::now();
    
    pORBextractor = new ORBextractor( nfeatures, fscaleFactor, nlevels, fIniThFAST, fMinThFAST );
    cout << "ORBextractor initialize finished!" << endl;

    // orb extractor
    cout << "Extract orb descriptors..." << endl;
    Mat curdesc;
    vector<KeyPoint> curkps;
    (*pORBextractor)( grayImgcur, maskcur, curkps, curdesc );


    // chrono::steady_clock::time_point time2 = chrono::steady_clock::now();
    // chrono::duration<double> time = chrono::duration_cast<chrono::duration<double>>(time2 - time1);
    // cout<<"均匀化特征点计算用时："<<time.count()*1000<<"ms"<<endl;


    // cout << "Extract orb descriptors finished!" << endl;
    // cout << "The number of keypoints are = " << srckps.size() << endl;

    // draw keypoints in output image
    Mat outImgcur;
    drawKeypoints( grayImgcur, curkps, outImgcur, cv::Scalar::all(-1), DrawMatchesFlags::DEFAULT );
    // imshow( "GridOrbKpsImg1", outImgcur );
    // waitKey( 0 );
    cout << "Finished! Congratulations!" << endl;
    //destroy all windows when you press any key
    //destroyAllWindows();

    BFMatcher matcher_bf(NORM_HAMMING, true); //使用汉明距离度量二进制描述子，允许交叉验证
    vector<DMatch> Matches_bf;
    matcher_bf.match(srcdesc, curdesc, Matches_bf);
    assert(Matches_bf.size() > 0);
    Mat BF_img;
    drawMatches(imagesrc, srckps, imagecur, curkps, Matches_bf, BF_img);
    resize(BF_img, BF_img, Size(2*imagesrc.cols, imagesrc.rows));
    // cout << Matches_bf.size() << endl;
    putText(BF_img, "Brute Force Matches", Point(20, 20), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 255), 2);
    // imshow("Brute Force Matches", BF_img);


    // 数据格式预处理
    vector<Point2f> X;
    vector<Point2f> Y;
    X.clear();
    Y.clear();

 	// -------------开始代码-------------
    // 将Matches_bf里的匹配点对分别放到X,Y 向量里，约5行代码，参考OpenCV DMatch类

    for(int i=0;i<Matches_bf.size();i++){
        int index1 = Matches_bf.at(i).queryIdx;
		int index2 = Matches_bf.at(i).trainIdx;
        X.push_back(srckps.at(index1).pt);
        Y.push_back(curkps.at(index2).pt);
    }


    // -------------结束代码-------------

    // 调用VFC主函数
    // t = (double)getTickCount();
    VFC myvfc;
    myvfc.setData(X, Y);
    myvfc.optimize();
    vector<int> matchIdx = myvfc.obtainCorrectMatch();
    // t = 1000 * ((double)getTickCount() - t) / getTickFrequency();

    // 筛选正确的匹配
    std::vector< DMatch > Matches_VFC;
    for (unsigned int i = 0; i < matchIdx.size(); i++) {
    int idx = matchIdx[i];
        Matches_VFC.push_back(Matches_bf[idx]);
    }

    // cout << "# Refine Matches (after VFC): " << Matches_VFC.size() << "/" << Matches_bf.size() << ", Times ="<<t<<" ms " << endl;

    // 绘制筛选结果
    Mat imageVFC;
    cv::drawMatches(imagesrc, srckps, imagecur, curkps, Matches_VFC, imageVFC);
    resize(imageVFC, imageVFC, Size(2*imagesrc.cols, imagesrc.rows));
    putText(imageVFC, "VFC Matches", Point(20, 20), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 255), 2);
    imshow("VFC Matches", imageVFC);
    cout << Matches_VFC.size() << endl;
    waitKey(0);




    //KNN匹配
    // cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);

    // // 设置 k 值
    // int k = 2; // 每个特征点最多匹配 k 个特征点

    // // 进行特征点匹配
    // std::vector<std::vector<cv::DMatch>> knnMatches;
    // matcher->knnMatch(srcdesc, curdesc, knnMatches, k);

    // // 可选：根据匹配程度进行筛选
    // float ratioThreshold = 0.6; // 设置筛选阈值
    // std::vector<cv::DMatch> filteredMatches;
    // for (const std::vector<cv::DMatch>& knnMatch : knnMatches) {
    //     if (knnMatch.size() >= 2) {
    //         float distanceRatio = knnMatch[0].distance / knnMatch[1].distance;
    //         if (distanceRatio < ratioThreshold) {
    //             filteredMatches.push_back(knnMatch[0]);
    //         }
    //     }
    // }

    // // 绘制匹配结果
    // cv::Mat matchImage;
    // cv::drawMatches(imagesrc, srckps, imagecur, curkps, filteredMatches, matchImage);
    //  cout << filteredMatches.size() << endl;
    // cv::imshow("Matches", matchImage);
    // cv::waitKey(0);


    return 0;
}