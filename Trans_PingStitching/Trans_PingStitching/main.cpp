#pragma comment(lib, ".\\lib\\libeng.lib")
#pragma comment(lib, ".\\lib\\libmx.lib")
#pragma comment(lib, ".\\lib\\libmex.lib")
#pragma comment(lib, ".\\lib\\libmat.lib")
#pragma comment(lib, ".\\lib\\mclmcrrt.lib")
#pragma comment(lib, ".\\lib\\mclmcr.lib")
///////////////////静态加载Matlab附加依赖

#include "RewGlobal.h"
#include "mclcppclass.h"
#include "mclmcrrt.h"
///////////////////Matlab

#include <Eigen/Core>
#include <Eigen/SVD>  
#include <Eigen\Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include <opencv2/core/eigen.hpp>
///////////////////Eigen


#include "highgui/highgui.hpp"    
#include "opencv2/nonfree/nonfree.hpp"    
#include "opencv2/legacy/legacy.hpp"
#include <opencv2/stitching/detail/seam_finders.hpp>
#include <opencv2/stitching/stitcher.hpp>
//////////////////opencv



#include <iostream>  
#include <math.h>
#include <vector>

using namespace cv;
using namespace std;
using namespace cv::detail;
using namespace Eigen;



int main(int argc, char *argv[]){


	//读图，这个方式需要根据情况更改
	Mat image01 = imread("../core/1.jpg");    //右图
	Mat image02 = imread("../core/2.jpg");    //左图
	Mat image03 = imread("../core/3.jpg");    //右图
	Mat image04 = imread("../core/4.jpg");    //左图

	vector<Mat> image;

	image.push_back(image01);
	image.push_back(image02);
	image.push_back(image03);
	image.push_back(image04);

    const int im_n = image.size();                     //im_n

	vector<vector<int>> edge_list(im_n-1,vector<int>(2));   // edge_list  matlab 中以1起头
	for (int ei = 1; ei <= im_n - 1; ei++){
	
		edge_list[ei-1][0] = ei;
		edge_list[ei-1][1] = ei + 1;

	}

	vector<vector<double>> imsize(im_n,vector<double>(3,0));// imsize
	for (int i = 0; i < im_n; i++){
	
		imsize[i][0] = (double)image[i].rows;
		imsize[i][1] = (double)image[i].cols;
		imsize[i][2] = (double)image[i].channels();
	}

	////////////////////////////////////////////////////////////////////////feature detection and Ransac rm/////////////////////////////////////////
	int thr_hesi = 500;//海森矩阵阈值
	vector<vector<KeyPoint>>keyPoint;//关键点
	vector<Mat> imageDesc;//关键点的描述
	int ei = image.size() - 1;//matlab 中ei 四张图，就有3个组合（1和2、2和3、3和4），ei=3。


	//提取特征点   
	SiftFeatureDetector Detector(thr_hesi);//海森矩阵阈值
	cout << "Detector(" << thr_hesi << ") ing……" << endl;

	//对每幅图像 检测特征点
	keyPoint.resize(image.size());
	for (int i = 0; i < image.size(); i++)
	{
		Detector.detect(image.at(i), keyPoint[i]);
		//这里可以优化为多线程进行
	}
	cout << "----------" << endl;


	//特征点描述，为下边的特征点匹配做准备
	SiftFeatureDetector Descriptor;
	cout << "Descriptor ing……" << endl;

	//对每幅图像 描述特征点
	imageDesc.resize(image.size());
	for (int i = 0; i < image.size(); i++)
	{
		Descriptor.compute(image[i], keyPoint[i], imageDesc[i]);
		//这里可以优化为多线程进行
	}
	cout << "----------" << endl;


	////RANSAC👇
	//剔除不匹配的点
	FlannBasedMatcher matcher;
	cout << "good point of ransac" << endl;

	//准备变量
	vector<Point2f> imagePoints1, imagePoints2;//X_1,X_2
	using PointGroup = vector<vector<Point2f>>;
	PointGroup PointsGroup(2);//中间变量，图1和图2的点match的点放在一个PointGroup中，
	vector<PointGroup> X(image.size() - 1);//有三个PointGroup，放在X里面，这里表示的是 X{ei,i}
	//输出的图
	Mat img_match;

	for (int k = 0; k < ei; k++)
	{
		vector<DMatch> matchePoints;
		matcher.match(imageDesc[k], imageDesc[k + 1], matchePoints, Mat());
		cout << "total match points of image: " << k << "&" << k + 1 << ": " << matchePoints.size() << endl;
		//删除错误匹配的特征点
		vector<cv::DMatch> InlierMatches;//定义内点集合
		vector<cv::Point2f> p1, p2;//先把keypoint转换为Point格式

		for (int i = 0; i < matchePoints.size(); i++)
		{
			p1.push_back(keyPoint[k][matchePoints[i].queryIdx].pt);// pt是position
			p2.push_back(keyPoint[k + 1][matchePoints[i].trainIdx].pt);
		}
		//RANSAC FindFundamental剔除错误点
		vector<uchar> RANSACStatus;//用以标记每一个匹配点的状态，等于0则为外点，等于1则为内点。
		cv::findFundamentalMat(p1, p2, RANSACStatus, CV_FM_RANSAC);//p1 p2必须为float型
		for (int i = 0; i < matchePoints.size(); i++)
		{
			if (RANSACStatus[i] != 0)
			{
				InlierMatches.push_back(matchePoints[i]); //不等于0的是内点
			}
		}
		//画出特征点的图
		drawMatches(image[k + 1], keyPoint[k + 1], image[k], keyPoint[k], InlierMatches, img_match);

		for (int i = 0; i < InlierMatches.size(); i++)
		{
			PointsGroup[0].push_back(keyPoint[k][InlierMatches[i].queryIdx].pt);
			PointsGroup[1].push_back(keyPoint[k + 1][InlierMatches[i].trainIdx].pt);
		}
		X[k] = PointsGroup;//最后的结果
		PointsGroup[0].clear();
		PointsGroup[1].clear();
	}
	////RANSAC👆
	cout << "----------------------------" << endl;


	//将X转换为eigen库表示  X---》mg

	//using MatrixVec = vector<MatrixXd>;
	//using MatrixGroup = vector<MatrixVec>;//X
	//MatrixVec mv;//存一对图
	//MatrixGroup mg;//ei对图
	//MatrixXd mr;//动态创建

	//for (int i = 0; i < ei; i++)//ei行即ei对图
	//{
	//	for (int j = 0; j <= 1; j++)//每行两张图
	//	{
	//		int size_point = X[i][j].size();
	//		mr = Eigen::MatrixXd::Ones(3, size_point);//用1赋初值，第三行就不用管了
	//		for (int i = 0; i < size_point; i++)
	//		{
	//			mr(0, i) = X[0][0][i].x;
	//			mr(1, i) = X[0][0][i].y;
	//		}
	//		mv.push_back(mr);
	//	}
	//	mg.push_back(mv);
	//	mv.clear();//清理，下次再用
	//}

	//最后的结果就在mg里面

	/////////////////////////////////////



	///////////////////////////////////////////////////////////////////////////////Global  Transform Estimation//////////////////////////////////
	if (!RewGlobalInitialize())
		return 1;

	//Transform mg to mwArray format
	int Xrows = im_n - 1, Xcols =2;

	mwArray XData(Xrows, Xcols, mxCELL_CLASS);
	mwArray imData(im_n, 3, mxDOUBLE_CLASS);
	mwArray edgeData(im_n - 1, 2, mxDOUBLE_CLASS);

	for (int i = 1; i <= im_n; i++){
		
		for (int j = 1; j <= 3; j++){
		
			imData(i, j) = imsize[i - 1][j - 1];

		}
	}
	for (int i = 1; i <= im_n - 1; i++){
		
		edgeData(i, 1) = i;
		edgeData(i, 2) = i + 1;

	}
	for (int i = 1; i <= Xrows; i++){

		for (int j = 1; j <= Xcols; j++){

			int size_point = X[i - 1][j - 1].size();
			mwArray tmp(3, size_point, mxDOUBLE_CLASS);
			for (int s = 1; s <= size_point; s++){
				tmp(1, s) = X[i - 1][j - 1][s-1].x;
				tmp(2, s) = X[i - 1][j - 1][s-1].y;
				tmp(3, s) = 1.0;
			}
			XData(i, j) = tmp;
		}
	}

	cout << "-----------------mwArray Transformation success-------------------" << endl;

	mwArray mParas, mR_ref;//globalTransEsti的两个输出矩阵  mwArray格式的 paras和R_ref

	globalTransEsti(2, mParas, mR_ref, XData, imData, edgeData);

	int nParas = 4*im_n-3;

	MatrixXd paras(1, nParas);                            //paras

	for (int i = 0; i < nParas; i++){
		paras(0, i) = mParas(1, i + 1);
	}
		

	MatrixXd R_ref(3, 3);                                //R_ref

	for (int i = 0; i < 3; i++){
		for (int j = 0; j < 3; j++){
			R_ref(i, j) = mR_ref(i+1, j + 1);
		}
	}
	cout << R_ref << endl;
	std::vector<MatrixXd> M;                          //M

	MatrixXd D(im_n, 1);							  //D

	std::vector<vector<MatrixXd>> R_pair(im_n, std::vector<MatrixXd>(im_n));  //R_pair

	for (int i = 0; i < im_n; i++){

		int ki = paras(0, i);
		MatrixXd tmp(3, 3);
		tmp << ki, 0, imsize[i][1] / 2,
			0, ki, imsize[i][0] / 2,
			0, 0, 1;
		M.push_back(tmp);
		D(i, 0) = 0;
		R_pair[i][i] = MatrixXd::Identity(3, 3);

	}

	for (int i = 2; i <= im_n; i++){

		MatrixXd theta = paras.block<1, 3>(0, im_n + 3 * (i - 2));
		MatrixXd theta_m(3, 3);

		theta_m << 0, -theta(0, 2), theta(0, 1),
			theta(0, 2), 0, -theta(0, 0),
			-theta(0, 1), theta(0, 0), 0;

		R_pair[0][i - 1] = theta_m.exp();
		R_pair[i - 1][0] = R_pair[0][i - 1].transpose();

	}


	const int refi = 1;
	std::vector<MatrixXd> R(im_n);
	for (int i = 0; i < im_n; i++){

		R[i] = R_pair[refi - 1][i] * R_ref.transpose();
	}


	RewGlobalTerminate;
	////////////////////////////////////////


	return 0;
}
