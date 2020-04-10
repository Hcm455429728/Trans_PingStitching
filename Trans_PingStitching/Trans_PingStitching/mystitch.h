#ifndef MYSTITCH_H
#define MYSTITCH_H

#include "highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"    
#include "opencv2/legacy/legacy.hpp"

using namespace cv;

//计算能量函数
Mat EnergyOfImg(Mat& source, Mat& goal, int start, int width);

//动态规划思想 找出最佳缝合线
vector<int> bestline(Mat& image);

//融合方法
void OptimizeSeam(Mat& img1, Mat& trans, Mat& dst, vector<int>& path, int start, int processWidth);

//缝合
Mat seaming(vector<Mat>& images, vector<int>& m_u0,vector<int>m_u1);


#endif