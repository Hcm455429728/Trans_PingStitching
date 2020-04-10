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
#include <unordered_set>
#include <string>
#include <ctime>
#include "thread"

#include "transforms.h"
#include "mystitch.h"


using namespace cv;
using namespace std;
using namespace cv::detail;
using namespace Eigen;


#define pi 3.1415926


clock_t start_time;
clock_t end_time;

using PointGroup = vector<vector<Point2f>>;

/*生产1到n的行向量*/
MatrixXd OnetoN(int n);

/*生产m到n的行向量*/
MatrixXd MtoN(double m, double n);

/*cat函数实现，p内各个mat的行一致*/
MatrixXd cat(vector<MatrixXd>& p);

/*行数不变 matlab cat(1)*/
void cat2(MatrixXd& res, MatrixXd& p, MatrixXd& q);

/*列数不变 matlab cat(2)*/
void cat3(MatrixXd& res, MatrixXd& p, MatrixXd& q);

/*function meshgird declaration*/
void meshgrid(Eigen::MatrixXd &vecX, Eigen::MatrixXd &vecY, Eigen::MatrixXd &meshX, Eigen::MatrixXd &meshY);

/*function count std declaration*/
double countStd(MatrixXd& src);



//多线程计算和描述特征点
void detectPoiot_thread_func(Mat& pic, SurfFeatureDetector& Detector, vector<KeyPoint>* keyPoint1)
{
	Detector.detect(pic, *keyPoint1);
}
void computePoint_thread_func(Mat& pic, SurfDescriptorExtractor& Descriptor, vector<KeyPoint>& keyPoint1, Mat* imageDesc1)
{
	Descriptor.compute(pic, keyPoint1, *imageDesc1);
}
void match_ransac_thread_func(FlannBasedMatcher& matcher, int k, vector<Mat>& imageDesc, vector<vector<KeyPoint>>& keyPoint, PointGroup* X)
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
	//drawMatches(image[k + 1], keyPoint[k + 1], image[k], keyPoint[k], InlierMatches, img_match);

	PointGroup PointsGroup(2);//中间变量，图1和图2的点match的点放在一个PointGroup中，

	for (int i = 0; i < InlierMatches.size(); i++)
	{
		PointsGroup[0].push_back(keyPoint[k][InlierMatches[i].queryIdx].pt);
		PointsGroup[1].push_back(keyPoint[k + 1][InlierMatches[i].trainIdx].pt);
	}
	*X = PointsGroup;//最后的结果
}


//测试多幅缝合
void test_seaming()
{
	//注意：从最右图开始存，每张图的大小一致
	//matlab在677行输出im_our{i}
	Mat image1 = imread("../core1/4_im_our.jpg");    //最右图
	Mat image2 = imread("../core1/3_im_our.jpg");    //左图
	Mat image3 = imread("../core1/2_im_our.jpg");    //
	Mat image4 = imread("../core1/1_im_our.jpg");    //最左图

	//注意：从大往小存，735和1843即4_im_our在图中的坐标
	//matlab在301行输出 fprintf('num--%d--m_u0_(%d)\n',i,m_u0_(i));
	vector<int> m_u0{ 735, 488, 213, 1 };
	vector<int> m_u1{ 1843, 1619, 1339, 1089 };

	vector<Mat> images{ image1, image2, image3, image4 };

	Mat dst = seaming(images, m_u0, m_u1);
	imwrite("../core1/seaming.jpg", dst);
}


int main(int argc, char *argv[]){


	test_seaming();

	return 0;



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

	//全局参数 与原程序中相同
	int im_n = image.size();                     //im_n
	const double lambda = 0.001*(image[0].cols)*(image[0].rows);
	const double intv_mesh = 10;
	const double K_smooth = 5;

	vector<vector<int>> edge_list(im_n - 1, vector<int>(2));   // edge_list  matlab 中以1起头
	int edge_n = im_n - 1;


	for (int ei = 1; ei <= im_n - 1; ei++){

		edge_list[ei - 1][0] = ei;
		edge_list[ei - 1][1] = ei + 1;

	}

	vector<vector<double>> imsize(im_n, vector<double>(3, 0));// imsize
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
	vector<thread> m_thread(image.size());//创建线程


	//Detector
	start_time = clock();
	SurfFeatureDetector Detector(thr_hesi);//海森矩阵阈值
	cout << "Detector(" << thr_hesi << ") ing……" << endl;
	keyPoint.resize(image.size());
	for (int i = 0; i < image.size(); i++)
		m_thread[i] = thread(detectPoiot_thread_func, image.at(i), Detector, &(keyPoint[i]));//传地址
	for (int i = 0; i < image.size(); i++)
		m_thread[i].join();
	end_time = clock();
	cout << "The Detector time is: " << (double)(end_time - start_time) / CLOCKS_PER_SEC << "s" << endl;
	cout << "----------" << endl;



	//Descriptor
	start_time = clock();
	SurfFeatureDetector Descriptor;
	cout << "Descriptor ing……" << endl;
	imageDesc.resize(image.size());
	for (int i = 0; i < image.size(); i++)
		m_thread[i] = thread(computePoint_thread_func, image[i], Descriptor, keyPoint[i], &(imageDesc[i]));
	for (int i = 0; i < image.size(); i++)
		m_thread[i].join();
	end_time = clock();
	cout << "The Descriptor time is: " << (double)(end_time - start_time) / CLOCKS_PER_SEC << "s" << endl;
	cout << "----------" << endl;


	//match and ransac
	start_time = clock();
	FlannBasedMatcher matcher;
	cout << "good point of ransac" << endl;
	vector<PointGroup> X(image.size() - 1);//有三个PointGroup，放在X里面，这里表示的是 X{ei,i}
	for (int k = 0; k < ei; k++)
		m_thread[k] = thread(match_ransac_thread_func, matcher, k, imageDesc, keyPoint, &X[k]);
	for (int k = 0; k < ei; k++)
		m_thread[k].join();
	end_time = clock();
	cout << "The RANSAC time is: " << (double)(end_time - start_time) / CLOCKS_PER_SEC << "s" << endl;
	cout << "----------------------------" << endl;



	///////////////////////////////////////////////////////////////////////////////Global  Transform Estimation//////////////////////////////////
	if (!RewGlobalInitialize())
		return 1;

	//Transform mg to mwArray format
	int Xrows = im_n - 1, Xcols = 2;

	mwArray XData(Xrows, Xcols, mxCELL_CLASS);
	mwArray imData(im_n, 3, mxDOUBLE_CLASS);
	mwArray edgeData(im_n - 1, 2, mxDOUBLE_CLASS);
	vector<vector<MatrixXd>> X_(Xrows, vector<MatrixXd>(Xcols));
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
			MatrixXd tmpM(3, size_point);
			for (int s = 1; s <= size_point; s++){
				tmp(1, s) = X[i - 1][j - 1][s - 1].x;
				tmp(2, s) = X[i - 1][j - 1][s - 1].y;
				tmp(3, s) = 1.0;

				tmpM(0, s - 1) = X[i - 1][j - 1][s - 1].x;
				tmpM(1, s - 1) = X[i - 1][j - 1][s - 1].y;
				tmpM(2, s - 1) = 1.0;
			}
			XData(i, j) = tmp;
			X_[i - 1][j - 1] = tmpM;
		}
	}

	cout << "-----------------mwArray Transformation success-------------------" << endl;

	mwArray mParas, mR_ref;//globalTransEsti的两个输出矩阵  mwArray格式的 paras和R_ref

	globalTransEsti(2, mParas, mR_ref, XData, imData, edgeData);

	int nParas = 4 * im_n - 3;

	MatrixXd paras(1, nParas);                            //paras

	for (int i = 0; i < nParas; i++){
		paras(0, i) = mParas(1, i + 1);
	}


	MatrixXd R_ref(3, 3);                                //R_ref

	for (int i = 0; i < 3; i++){
		for (int j = 0; j < 3; j++){
			R_ref(i, j) = mR_ref(i + 1, j + 1);
		}
	}

	cout << R_ref << endl;
	std::vector<MatrixXd> M;                          //M

	std::vector<MatrixXd> D(im_n, MatrixXd::Zero(1, 1));							  //D

	std::vector<vector<MatrixXd>> R_pair(im_n, std::vector<MatrixXd>(im_n));  //R_pair

	for (int i = 0; i < im_n; i++){

		int ki = paras(0, i);
		MatrixXd tmp(3, 3);
		tmp << ki, 0, imsize[i][1] / 2,
			0, ki, imsize[i][0] / 2,
			0, 0, 1;
		M.push_back(tmp);
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

	for (int i = 2; i <= im_n - 1; i++){
		for (int j = i + 1; j <= im_n; j++){

			R_pair[i - 1][j - 1] = R_pair[0][j - 1] * R_pair[i - 1][0];
			R_pair[j - 1][i - 1] = R_pair[i - 1][j - 1].transpose();
		}
	}

	const int refi = 1;
	std::vector<MatrixXd> R(im_n);
	for (int i = 0; i < im_n; i++){

		R[i] = R_pair[refi - 1][i] * R_ref.transpose();
	}


	RewGlobalTerminate;
	cout << R_pair[1][2] << endl;
	cout << "-------------global transformation estimation success ------------" << endl;
	////////////////////////////////////////



	////////////////comput_mosaic_parameters////////////////////

	double fe = max(M[refi - 1](0, 0), M[refi - 1](1, 1));//下标从0开始的，都要减一

	vector<MatrixXd> ubox(im_n);
	vector<MatrixXd> vbox(im_n);
	vector<MatrixXd> ubox_(im_n);
	vector<MatrixXd> vbox_(im_n);
	MatrixXd ubox_all_;
	MatrixXd vbox_all_;
	MatrixXd part1, part2, part3, part4;//ubox\vbox都是由四部分组成

	vector<MatrixXd> ubox_vec;
	vector<MatrixXd> vbox_vec;//for matlab cat

	for (int i = 0; i < im_n; i++)
	{
		//ubox{i} = [1:imsize(i,2)  1:imsize(i,2)  ones(1,imsize(i,1))  imsize(i,2)*ones(1,imsize(i,1))] ;
		part1 = OnetoN(imsize[i][1]);//1：第i幅图片的列数
		part2 = part1;
		part3.setOnes(1, imsize[i][0]);//i幅图片行数个1
		part4.setConstant(1, imsize[i][0], imsize[i][1]);
		ubox[i].resize(1, 2 * (imsize[i][1] + imsize[i][0]));
		ubox[i] << part1, part2, part3, part4;

		//vbox{i} = [ones(1,imsize(i,2))  imsize(i,1)*ones(1,imsize(i,2))  1:imsize(i,1)        1:imsize(i,1) ];
		part1.setOnes(1, imsize[i][1]);//i幅图片的列数个1
		part2.setConstant(1, imsize[i][1], imsize[i][0]);
		part3 = OnetoN(imsize[i][0]);//1：第i幅图片的行数
		part4 = part3;
		vbox[i].resize(1, 2 * (imsize[i][1] + imsize[i][0]));
		vbox[i] << part1, part2, part3, part4;

		//[ubox_{i}, vbox_{i}] =  trans_persp2equi(ubox{i}, vbox{i}, R{i}', M{i}, D{i}, fe);
		MatrixXd temp = R[i].transpose();//转置要用中间变量，不支持自复制
		trans_persp2equi(ubox_[i], vbox_[i], ubox[i], vbox[i], temp, M[i], D[i], fe);


		ubox_vec.push_back(ubox_[i]);
		vbox_vec.push_back(vbox_[i]);//for matlab cat
	}

	ubox_all_ = cat(ubox_vec);
	vbox_all_ = cat(vbox_vec);/*ubox_all_ << ubox_[0], ubox_[1], ubox_[2], ubox_[3];  vbox_all_ << vbox_[0], vbox_[1], vbox_[2], vbox_[3];*/


	double u0 = ubox_all_.minCoeff();
	double u1 = ubox_all_.maxCoeff();
	MatrixXd ur = MtoN(u0, u1);
	double v0 = vbox_all_.minCoeff();
	double v1 = vbox_all_.maxCoeff();
	MatrixXd vr = MtoN(v0, v1);
	double mosaicw = ur.cols();
	double mosaich = vr.cols();

	MatrixXd m_u0_, m_u1_, m_v0_, m_v1_, imw_, imh_;
	m_u0_.setOnes(im_n, 1);
	m_u1_.setOnes(im_n, 1);
	m_v0_.setOnes(im_n, 1);
	m_v1_.setOnes(im_n, 1);
	imw_.setOnes(im_n, 1);
	imh_.setOnes(im_n, 1);

	for (int i = 0; i < im_n; i++)
	{
		double margin = 0.2 * min(imsize[0][0], imsize[0][1]);
		double u0_im_ = max(ubox_[i].minCoeff() - margin, u0);
		double u1_im_ = min(ubox_[i].maxCoeff() + margin, u1);
		double v0_im_ = max(vbox_[i].minCoeff() - margin, v0);
		double v1_im_ = min(vbox_[i].maxCoeff() + margin, v1);
		m_u0_(i, 0) = int(u0_im_ - u0 + 1) + 1;
		m_u1_(i, 0) = int(u1_im_ - u0 + 1);
		m_v0_(i, 0) = int(v0_im_ - v0 + 1) + 1;
		m_v1_(i, 0) = int(v1_im_ - v0 + 1);
		imw_(i, 0) = int(m_u1_(i) - m_u0_(i) + 1); //最后每张图的长宽
		imh_(i, 0) = int(m_v1_(i) - m_v0_(i) + 1);
	}
	cout << "-------------------comput_mosaic_parameters-------------------------------" << endl;

	///////////////////////////////////////




	////////////////////local mosaic///////////////////////////

	vector<vector<int>> Adj(im_n, vector<int>(im_n, 0));


	for (int ei = 1; ei <= edge_n; ei++){
		int i = edge_list[ei - 1][0];
		int j = edge_list[ei - 1][1];
		Adj[i - 1][j - 1] = ei;
		Adj[j - 1][i - 1] = ei;
	}

	int XLength = ur.cols(), YLength = vr.cols();

	MatrixXd u(YLength, XLength), v(YLength, XLength);

	meshgrid(ur, vr, u, v);

	for (int ki = 1; ki <= im_n; ki++){

		int i = floor((ki + refi - 2) % im_n) + 1;
		MatrixXd u_im = u.block(m_v0_(ki - 1, 0) - 1, m_u0_(ki - 1, 0) - 1, m_v1_(ki - 1, 0) - m_v0_(ki - 1, 0) + 1, m_u1_(ki - 1, 0) - m_u0_(ki - 1, 0) + 1);
		MatrixXd v_im = v.block(m_v0_(ki - 1, 0) - 1, m_u0_(ki - 1, 0) - 1, m_v1_(ki - 1, 0) - m_v0_(ki - 1, 0) + 1, m_u1_(ki - 1, 0) - m_u0_(ki - 1, 0) + 1);
		MatrixXd u_im_, v_im_;
		trans_equi2persp(u_im_, v_im_, u_im, v_im, R[i - 1], M[i - 1], D[i - 1], fe);///时间有点长

		bool need_deform = false;
		vector<double> sub_u0, sub_u1, sub_v0, sub_v1;
		MatrixXd Pi, Pi_;


		for (int kj = 1; kj <= ki - 1; kj++){

			int j = floor((kj + refi - 2) % im_n) + 1;
			if (Adj[i - 1][j - 1] > 0){

				need_deform = true;
				MatrixXd ubox_ji, vbox_ji;
				trans_persp2persp(ubox_ji, vbox_ji, ubox[j - 1], vbox[j - 1], R_pair[j - 1][i - 1], M[j - 1], D[j - 1], M[i - 1], D[i - 1]);
				sub_u0.push_back(max(1.0, ubox_ji.minCoeff()));
				sub_u1.push_back(min(imsize[i - 1][1], ubox_ji.maxCoeff()));
				sub_v0.push_back(max(1.0, vbox_ji.minCoeff()));
				sub_v1.push_back(min(imsize[i - 1][0], vbox_ji.maxCoeff()));

				int ei = Adj[i - 1][j - 1];
				MatrixXd Xi, Xj;
				if (i == edge_list[ei - 1][0] && j == edge_list[ei - 1][1]){
					Xi = X_[ei - 1][0];
					Xj = X_[ei - 1][1];
				}
				else
				{
					Xi = X_[ei - 1][1];
					Xj = X_[ei - 1][0];
				}
				MatrixXd xj_i, yj_i;
				MatrixXd Xirow = Xi.row(0);
				MatrixXd Xjrow = Xi.row(1);
				trans_persp2persp(xj_i, yj_i, Xirow, Xjrow, R_pair[j - 1][i - 1], M[j - 1], D[j - 1], M[i - 1], D[i - 1]);

				Pi = Xi.topRows(2);
				cat3(Pi_, xj_i, yj_i);


			}
		}

		if (need_deform){

			double sub_u0_ = *(std::max_element(std::begin(sub_u0), std::end(sub_u0)));
			double sub_u1_ = *(std::max_element(std::begin(sub_u1), std::end(sub_u1)));
			double sub_v0_ = *(std::max_element(std::begin(sub_v0), std::end(sub_v0)));
			double sub_v1_ = *(std::max_element(std::begin(sub_v1), std::end(sub_v1)));

			// merge the coincided points
			string piKey = "", pi_Key = "";
			unordered_set<string> ok_Pi, ok_Pi_;
			vector<int> ok_cols;
			for (int i = 0; i < Pi.cols(); i++){

				piKey = to_string(floor(Pi(0, i) + 0.5)) + "+" + to_string(floor(Pi(1, i) + 0.5));
				pi_Key = to_string(floor(Pi_(0, i) + 0.5)) + "+" + to_string(floor(Pi_(1, i) + 0.5));

				if (ok_Pi.find(piKey) == ok_Pi.end() && ok_Pi_.find(pi_Key) == ok_Pi_.end()){
					ok_cols.push_back(i);
					ok_Pi.insert(piKey);
					ok_Pi_.insert(pi_Key);
				}

			}

			MatrixXd Pi_nd(2, ok_cols.size()), Pi_nd_(2, ok_cols.size());
			for (int i = 0; i < ok_cols.size(); i++){
				Pi_nd.col(i) = Pi.col(ok_cols[i]);
				Pi_nd_.col(i) = Pi_.col(ok_cols[i]);
			}

			//form the linear system
			MatrixXd xj_ = Pi_nd_.row(0), yj_ = Pi_nd_.row(1);
			int n = xj_.cols();
			MatrixXd gxn = xj_ - Pi_nd.row(0), hyn = yj_ - Pi_nd.row(1);

			MatrixXd xx(n, n), yy(n, n);

			for (int i = 0; i < n; i++){

				xx.row(i) = xj_.row(0);
				yy.row(i) = yj_.row(0);

			}

			MatrixXd dist2 = (xx - xx.transpose()).array().square() + (yy - yy.transpose()).array().square();
			for (int i = 0; i < n; i++){

				dist2(i, i) = 1;
			}
			MatrixXd T = dist2.array().log();// 为了中转而设置的变量
			MatrixXd K = 0.5*(dist2.cwiseProduct(T));

			for (int i = 0; i < n; i++){
				K(i, i) = lambda * 8 * pi;
			}

			MatrixXd K_ = MatrixXd::Zero(n + 3, n + 3);
			MatrixXd G_ = MatrixXd::Zero(n + 3, 2);

			K_.topLeftCorner(n, n) = K;

			for (int i = 0; i < n; i++){

				K_(n, i) = xj_(0, i);
				K_(n + 1, i) = yj_(0, i);
				K_(n + 2, i) = 1.0;

				K_(i, n) = xj_(0, i);
				K_(i, n + 1) = yj_(0, i);
				K_(i, n + 2) = 1.0;

				G_(i, 0) = gxn(0, i);
				G_(i, 1) = hyn(0, i);

			}

			//solve the linear system
			MatrixXd W_ = K_.inverse()*G_;
			MatrixXd wx = W_.topLeftCorner(n, 1), wy = W_.topRightCorner(n, 1);
			MatrixXd a = W_.bottomLeftCorner(3, 1), b = W_.bottomRightCorner(3, 1);


			//remove outliers based on the distribution of weights

			vector<int> inlier;
			for (int kiter = 1; kiter < 10; kiter++){

				double wxStd = countStd(wx), wyStd = countStd(wy);
				for (int i = 0; i < wx.rows(); i++){

					if (abs(wx(i, 0)) < wxStd&&abs(wy(i, 0)) < wyStd)
						inlier.push_back(i);
				}

				int kn = inlier.size();

				if (kn < 0.0027*(K_.cols() - 3))
					break;

				MatrixXd tmpK(kn + 3, K_.cols());
				MatrixXd tmpG(kn + 3, 2);

				for (int i = 0; i < kn; i++){
					tmpK.row(i) = K_.row(inlier[i]);
					tmpG.row(i) = G_.row(inlier[i]);
				}
				for (int j = 0; j < 3; j++){
					tmpG.row(kn + j) = G_.row(n + j);
					tmpK.row(kn + j) = K_.row(n + j);
				}
				G_ = tmpG;

				K_ = MatrixXd(kn + 3, kn + 3);

				for (int s = 0; s < kn; s++){

					K_.col(s) = tmpK.col(inlier[s]);
				}

				for (int t = 0; t < 3; t++){
					K_.col(kn + t) = tmpK.col(kn + t);
				}

				W_ = K_.inverse()*G_;
				wx = W_.topLeftCorner(kn, 1), wy = W_.topRightCorner(kn, 1);
				a = W_.bottomLeftCorner(3, 1), b = W_.bottomRightCorner(3, 1);
				if (kiter < 9)
					inlier.clear();

			}

			int outSize = inlier.size();
			MatrixXd xj_tmp = xj_, yj_tmp = yj_, gxnTmp = gxn, hynTmp = hyn;
			xj_ = MatrixXd(1, outSize);
			yj_ = MatrixXd(1, outSize);
			gxn = MatrixXd(1, outSize);
			hyn = MatrixXd(1, outSize);

			for (int i = 0; i < outSize; i++){
				xj_(0, i) = xj_tmp(0, inlier[i]);
				yj_(0, i) = yj_tmp(0, inlier[i]);
				gxn(0, i) = gxnTmp(0, inlier[i]);
				hyn(0, i) = hynTmp(0, inlier[i]);
			}

			sub_u0_ = sub_u0_ + gxn.minCoeff();
			sub_u1_ = sub_u1_ + gxn.maxCoeff();
			sub_v0_ = sub_v0_ + hyn.minCoeff();
			sub_v1_ = sub_v1_ + hyn.maxCoeff();

			double eta_d0 = 0;
			double eta_d1 = K_smooth * max((gxn.cwiseAbs()).maxCoeff(), (hyn.cwiseAbs()).maxCoeff());





			MatrixXd u_mesh_((int)(ceil(imh_(i - 1, 0) / intv_mesh)), (int)(ceil(imw_(i - 1, 0) / intv_mesh)));
			MatrixXd v_mesh_((int)(ceil(imh_(i - 1, 0) / intv_mesh)), (int)(ceil(imw_(i - 1, 0) / intv_mesh)));
			MatrixXd gx_mesh_ = MatrixXd::Zero((int)(ceil(imh_(i - 1, 0) / intv_mesh)), (int)(ceil(imw_(i - 1, 0) / intv_mesh)));
			MatrixXd hy_mesh_ = MatrixXd::Zero((int)(ceil(imh_(i - 1, 0) / intv_mesh)), (int)(ceil(imw_(i - 1, 0) / intv_mesh)));
			MatrixXd rbf, tmpLog;
			for (int kf = 1; kf <= outSize; kf++){
				dist2 = (u_mesh_.array() + xj_(0, kf - 1)).array().square() + (v_mesh_.array() + yj_(0, kf - 1)).array().square();
				tmpLog = dist2.array().log();
				rbf = 0.5*(dist2.cwiseProduct(tmpLog));
				gx_mesh_ += wx(kf - 1, 0)*rbf;
				hy_mesh_ += wy(kf - 1, 0)*rbf;
			}
			gx_mesh_ = (gx_mesh_ + a(0, 0)*u_mesh_ + a(1, 0)*v_mesh_).array() + a(2, 0);
			hy_mesh_ = (hy_mesh_ + b(0, 0)*u_mesh_ + b(1, 0)*v_mesh_).array() + b(2, 0);
			Mat m_gx, m_hy;
			eigen2cv(gx_mesh_, m_gx);
			eigen2cv(hy_mesh_, m_hy);
			Mat m_gx_, m_hy_;
			cv::resize(m_gx, m_gx_, cv::Size(imw_(i - 1, 0), imh_(i - 1, 0)), 0, 0, cv::INTER_CUBIC);
			cv::resize(m_hy, m_hy_, cv::Size(imw_(i - 1, 0), imh_(i - 1, 0)), 0, 0, cv::INTER_CUBIC);
			int gxCols = m_gx_.cols, gxRows = m_gx_.rows;
			MatrixXd gx_im_(gxRows, gxCols), hy_im_(gxRows, gxCols);
			cv2eigen(m_gx_, gx_im_);
			cv2eigen(m_hy_, hy_im_);


			//smooth tansition to global transform
			MatrixXd u0_u_im_ = u_im_.array() - sub_u0_;
			u0_u_im_ *= -1;
			MatrixXd u1_u_im_ = u_im_.array() - sub_u1_;

			MatrixXd dist_horizontal = u0_u_im_.cwiseMax(u1_u_im_);

			MatrixXd v0_v_im_ = v_im_.array() - sub_v0_;
			v0_v_im_ *= -1;
			MatrixXd v1_v_im_ = v_im_.array() - sub_v1_;
			MatrixXd dist_vertical = v0_v_im_.cwiseMax(v1_v_im_);

			MatrixXd dist_sub = dist_horizontal.cwiseMax(dist_vertical);
			MatrixXd comZero = MatrixXd::Zero(dist_sub.rows(), dist_sub.cols());
			dist_sub = comZero.cwiseMax(dist_sub);
			MatrixXd eta = -1 * (dist_sub.array() - eta_d1) / (eta_d1 - eta_d0);
			MatrixXd comOne = MatrixXd::Ones(dist_sub.rows(), dist_sub.cols());

			eta = (dist_sub.array() < eta_d0).select(comOne, eta);
			eta = (dist_sub.array() > eta_d1).select(comZero, eta);

			gx_im_ = gx_im_.cwiseProduct(eta);
			hy_im_ = hy_im_.cwiseProduct(eta);
			u_im_ = u_im_ - gx_im_;
			v_im_ = v_im_ - hy_im_;



			//update the feature locations
			MatrixXd newXi, Xi;
			for (int kj = ki + 1; kj <= im_n; kj++){

				int j = floor((kj + refi - 2) % im_n) + 1;

				if (Adj[i - 1][j - 1] > 0){

					int ei = Adj[i - 1][j - 1];
					if (i == edge_list[ei - 1][0] && j == edge_list[ei - 1][1]){
						Xi = X_[ei - 1][0];
					}
					else
						Xi = X_[ei - 1][1];

					newXi = Xi;

					MatrixXd u_f, v_f;
					MatrixXd gx_f, hy_f;
					MatrixXd u0_u_f, u1_u_f;
					MatrixXd v0_v_f, v1_v_f;
					MatrixXd dist_horizontal_f, dist_vertical_f;
					MatrixXd dist_sub_f;
					MatrixXd eta_f;
					MatrixXd comZero_1, comOne_1;
					for (int kiter = 1; kiter <= 20; kiter++){
						u_f = newXi.row(0);
						v_f = newXi.row(1);
						gx_f = MatrixXd::Zero(1, newXi.cols());
						hy_f = MatrixXd::Zero(1, newXi.cols());

						for (int kf = 1; kf <= outSize; kf++){

							dist2 = (u_f.array() + xj_(0, kf - 1)).array().square() + (v_f.array() + yj_(0, kf - 1)).array().square();
							tmpLog = dist2.array().log();
							rbf = 0.5*(dist2.cwiseProduct(tmpLog));
							gx_f = gx_f + wx(kf - 1, 0)*rbf;
							hy_f = hy_f + wy(kf - 1, 0)*rbf;

						}
						gx_f = (gx_f + a(0, 0)*u_f + a(1, 0)*v_f).array() + a(2, 0);
						hy_f = (hy_f + b(0, 0)*u_f + b(1, 0)*v_f).array() + b(2, 0);

						u0_u_f = u_f.array() - sub_u0_;
						u0_u_f *= -1;
						u1_u_f = u_f.array() - sub_u1_;

						dist_horizontal_f = u0_u_f.cwiseMax(u1_u_f);

						v0_v_f = v_f.array() - sub_v0_;
						v0_v_f *= -1;
						v1_v_f = v_f.array() - sub_v1_;
						dist_vertical_f = v0_v_f.cwiseMax(v1_v_f);

						dist_sub_f = dist_horizontal_f.cwiseMax(dist_vertical_f);
						comZero_1 = MatrixXd::Zero(dist_sub_f.rows(), dist_sub_f.cols());
						dist_sub_f = comZero_1.cwiseMax(dist_sub_f);
						eta_f = -1 * (dist_sub_f.array() - eta_d1) / (eta_d1 - eta_d0);
						comOne_1 = MatrixXd::Ones(dist_sub_f.rows(), dist_sub_f.cols());

						eta_f = (dist_sub_f.array() < eta_d0).select(comOne_1, eta_f);
						eta_f = (dist_sub_f.array() > eta_d1).select(comZero_1, eta_f);

						gx_f = gx_f.cwiseProduct(eta_f);
						hy_f = hy_f.cwiseProduct(eta_f);


						newXi.row(0) = Xi.row(0) + gx_f;
						newXi.row(1) = Xi.row(1) + hy_f;

					}
					if (i == edge_list[ei - 1][0] && j == edge_list[ei - 1][1])
						X_[ei - 1][0] = newXi;
					else
						X_[ei - 1][1] = newXi;
				}

			}

		}//if(need_deform)

	}//for ki = 1:im_n



	return 0;
}




void  meshgrid(Eigen::MatrixXd &vecX, Eigen::MatrixXd &vecY, Eigen::MatrixXd &meshX, Eigen::MatrixXd &meshY)
{
	int vecXLength = vecX.cols();
	int vecYLength = vecY.cols();


	for (int i = 0; i < vecYLength; ++i)
	{
		meshX.row(i) = vecX.row(0);
	}

	for (int i = 0; i < vecXLength; ++i)
	{
		meshY.col(i) = vecY.transpose().col(0);

	}
	return;
}

double countStd(MatrixXd& src){

	double all = 0; //总数
	double sqAll = 0;
	int count = src.rows();
	for (int i = 0; i < count; i++){

		all += src(i, 0);

	}

	for (int i = 0; i < count; i++){
		sqAll += pow((src(i, 0) - all), 2);
	}

	return sqrt(sqAll);

}

/*生产1到n的行向量*/
MatrixXd OnetoN(int n)
{
	assert(n > 0);
	MatrixXd res(1, n);
	for (int i = 0; i < n; i++)
		res(0, i) = i + 1;
	return res;
}
/*生产m到n的行向量*/
MatrixXd MtoN(double m, double n)
{
	assert(n > m);
	int size = n - m + 1;
	MatrixXd res(1, size);
	for (int i = 0; i < size; i++, m++)
		res(0, i) = m;
	return res;
}
/*cat函数实现，p内各个mat的行一致*/
MatrixXd cat(vector<MatrixXd>& p)
{
	assert(p.size()>0);
	int col = 0;
	for (int i = 0; i < p.size(); i++)
		col += p[i].cols();//按列拼接，行数不变

	MatrixXd res(p[0].rows(), col);
	int num = 0;
	for (int row = 0; row < p[0].rows(); row++)
	{
		for (int i = 0; i < p.size(); i++)//每个mat
		{
			for (int j = 0; j < p[i].cols(); j++)//某个mat的列
			{
				res(row, num) = p[i](row, j);
				num++;
			}
		}
	}
	return res;
}

void cat2(MatrixXd& res, MatrixXd& p, MatrixXd& q)//行数不变 matlab cat(1)
{

	int col = 0;

	col += p.cols();
	col += q.cols();

	res = MatrixXd(p.rows(), col);

	for (int row = 0; row < p.rows(); row++)
	{
		int num = 0;
		for (int j = 0; j < p.cols(); j++)//某个mat的列
		{
			res(row, num) = p(row, j);
			num++;
		}
		for (int j = 0; j < q.cols(); j++){

			res(row, num) = q(row, j);
			num++;
		}

	}
	return;
}

void cat3(MatrixXd& res, MatrixXd& p, MatrixXd& q)//列数不变 matlab cat(2)
{

	int row = 0;

	row += p.rows();
	row += q.rows();

	res = MatrixXd(row, p.cols());


	int num = 0;
	for (int j = 0; j < p.rows(); j++)//某个mat的列
	{
		res.row(num) = p.row(j);
		num++;
	}
	for (int j = 0; j < q.rows(); j++){

		res.row(num) = q.row(j);
		num++;
	}


	return;
}
