#ifndef MYSTITCH_H
#define MYSTITCH_H

#include "highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"    
#include "opencv2/legacy/legacy.hpp"

using namespace cv;

//������������
Mat EnergyOfImg(Mat& source, Mat& goal, int start, int width);

//��̬�滮˼�� �ҳ���ѷ����
vector<int> bestline(Mat& image);

//�ںϷ���
void OptimizeSeam(Mat& img1, Mat& trans, Mat& dst, vector<int>& path, int start, int processWidth);

//���
Mat seaming(vector<Mat>& images, vector<int>& m_u0,vector<int>m_u1);


#endif