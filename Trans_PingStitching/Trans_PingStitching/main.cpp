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



//csp
/*
para:任意大小的矩阵 u v M D
return:存放了xcam, ycam, zcam的容器
func:function [xcam, ycam, zcam] = trans_persp2cam(u, v, M, D)
*/
vector<MatrixXd> trans_persp2cam(MatrixXd& u, MatrixXd& v, MatrixXd& M, MatrixXd& D)
{
	double fx = M(0, 0), fy = M(1, 1);
	double cx = M(0, 2), cy = M(1, 2);
	double k1, k2, p1, p2, k3, k4, k5, k6;
	if (D.size() != 0)
	{
		k1 = D(0); k2 = D(1); p1 = D(2); p2 = D(3);
		k3 = D(4); k4 = D(5); k5 = D(6); k6 = D(7);
	}
	MatrixXd xcam, ycam, zcam;
	//x_d = (u - cx) / fx; “矩阵-常数”的操作 //R.array() -= s; ----> R = R - s
	xcam = (u.array() - cx) / fx;
	ycam = (v.array() - cy) / fy;

	////测试输出 没有问题
	//printMatrix(xcam);
	//cout << "------" << endl;
	//printMatrix(ycam);

	//if D!=0 跳过了
	zcam.setOnes(xcam.rows(), xcam.cols());
	vector<MatrixXd> res;
	res.push_back(xcam);
	res.push_back(ycam);
	res.push_back(zcam);
	return res;
}

/*
para:任意大小的矩阵 xcam ycam zcam fe
return:存放了u, v的容器
func:function [u, v] = trans_cam2equi(xcam, ycam, zcam, fe)
*/
vector<MatrixXd> trans_cam2equi(MatrixXd& xcam, MatrixXd& ycam, MatrixXd& zcam, double fe)
{
	MatrixXd rr = xcam.array().square() + ycam.array().square() + zcam.array().square();//matlab中.^2是每个元素求平方
	rr = rr.array().sqrt();// sqrt(rr),输出没有问题

	//theta = acos(zcam . / sqrt(xcam . ^ 2 + zcam . ^ 2)) .* (2 * (xcam > 0) - 1) ;
	MatrixXd thera;//temp1.*temp2;	
	MatrixXd temp1 = xcam.array().square() + zcam.array().square();
	temp1 = temp1.array().sqrt();//sqrt(xcam . ^ 2 + zcam . ^ 2)
	temp1 = zcam.array() / temp1.array();//zcam . / sqrt(xcam . ^ 2 + zcam . ^ 2)
	temp1 = temp1.array().acos(); //acos(zcam . / sqrt(xcam . ^ 2 + zcam . ^ 2))

	MatrixXd temp2(xcam.rows(), xcam.cols());
	for (int i = 0; i < xcam.rows(); i++)//(xcam > 0)
	{
		for (int j = 0; j < xcam.cols(); j++)
		{
			if (xcam(i, j)>0)
				temp2(i, j) = 2;//本来应该=1，直接等于2，后面不用再*2
			else
				temp2(i, j) = 0;
		}
	}
	temp2.array() -= 1;//(2 * (xcam > 0) - 1)
	thera = temp1.array()*temp2.array();//.* //输出正确



	//phi = asin(ycam ./ rr);
	MatrixXd phi = ycam.array() / rr.array();
	phi = phi.array().asin(); //输出正确

	vector<MatrixXd> res;
	res.push_back(fe*thera);
	res.push_back(fe*phi);
	return res;
}

/*
para:矩阵 x(1xN) y(1xN) R(3x3) M(3x3) D(一直都是0) double型fe
return:存放了u, v的容器
func:function [u, v] = trans_persp2equi(x, y, R, M, D, fe)
*/
vector<MatrixXd> trans_persp2equi(MatrixXd& x, MatrixXd& y, MatrixXd& R, MatrixXd& M, MatrixXd& D, double fe)
{
	vector<MatrixXd> xyz_cam = trans_persp2cam(x, y, M, D);
	MatrixXd xcam = xyz_cam[0];
	MatrixXd ycam = xyz_cam[1];
	MatrixXd zcam = xyz_cam[2];
	MatrixXd xr = R(0, 0) * xcam + R(0, 1) * ycam + R(0, 2) * zcam;
	MatrixXd yr = R(1, 0) * xcam + R(1, 1) * ycam + R(1, 2) * zcam;
	MatrixXd zr = R(2, 0) * xcam + R(2, 1) * ycam + R(2, 2) * zcam;

	//测试输出,没有问题

	vector<MatrixXd> uv = trans_cam2equi(xr, yr, zr, fe);
	return uv;
}

//computing mosaic parameters
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

/*
para:M(im_n,3,3),imsize(im_n,3) , im_n:图像张数
return:输出可以自行选择
*/
void comput_mosaic_parameters(vector<MatrixXd>& M, MatrixXd& imsize, vector<MatrixXd>& R, MatrixXd& D, int im_n)
{
	int refi = 1;//上面赋值为1；
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
		part1 = OnetoN(imsize(i, 1));//1：第i幅图片的列数
		part2 = part1;
		part3.setOnes(1, imsize(i, 0));//i幅图片行数个1
		part4.setConstant(1, imsize(i, 0), imsize(i, 1));
		ubox[i].resize(1, 2 * (imsize(i, 1) + imsize(i, 0)));
		ubox[i] << part1, part2, part3, part4;

		//vbox{i} = [ones(1,imsize(i,2))  imsize(i,1)*ones(1,imsize(i,2))  1:imsize(i,1)        1:imsize(i,1) ];
		part1.setOnes(1, imsize(i, 1));//i幅图片的列数个1
		part2.setConstant(1, imsize(i, 1), imsize(i, 0));
		part3 = OnetoN(imsize(i, 0));//1：第i幅图片的行数
		part4 = part3;
		vbox[i].resize(1, 2 * (imsize(i, 1) + imsize(i, 0)));
		vbox[i] << part1, part2, part3, part4;

		//[ubox_{i}, vbox_{i}] =  trans_persp2equi(ubox{i}, vbox{i}, R{i}', M{i}, D{i}, fe);
		MatrixXd temp = R[i];
		R[i] = temp.transpose();//转置要用中间变量，不支持自复制
		vector<MatrixXd> uv = trans_persp2equi(ubox[i], vbox[i], R[i], M[i], D, fe);
		ubox_[i] = uv[0];
		vbox_[i] = uv[1];

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
		double margin = 0.2 * min(imsize(0, 0), imsize(0, 1));
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
}

//测试代码
void test_comput_mosaic_parameters()
{
	vector<MatrixXd> M;
	MatrixXd m1(3, 3), m2(3, 3), m3(3, 3), m4(3, 3);
	m1 << 10743.3280958030, 0, 521,
		0, 10743.3280958030, 119.500000000000,
		0, 0, 1;
	m2 << 10715.4680850299, 0, 521,
		0, 10715.4680850299, 119.500000000000,
		0, 0, 1;
	m3 << 10786.0125733829, 0, 521,
		0, 10786.0125733829, 119.500000000000,
		0, 0, 1;
	m4 << 10650.4459525396, 0, 521,
		0, 10650.4459525396, 119.500000000000,
		0, 0, 1;
	M.push_back(m1);
	M.push_back(m2);
	M.push_back(m3);
	M.push_back(m4);


	MatrixXd imsize(4, 3);
	imsize << 239, 1042, 3,
		239, 1042, 3,
		239, 1042, 3,
		239, 1042, 3;

	vector<MatrixXd> R;
	MatrixXd r1(3, 3), r2(3, 3), r3(3, 3), r4(3, 3);

	r1 << 0.999254119147841, -0.0109239618370480, 0.0370387961987915,
		0.0109348400755766, 0.999940208696513, -9.11290041423332e-05,
		-0.0370355861111237, 0.000496074345788726, 0.999313824217222;
	r2 << 0.999914984813287, -0.000292620453986326, 0.0130360085575790,
		0.000292819287340547, 0.999999957039401, -1.33439494320460e-05,
		-0.0130360040928317, 1.71600097292891e-05, 0.999915027541254;
	r3 << 0.999890456865642, 0.00741340821192752, -0.0128107629633023,
		-0.00741307579663919, 0.999972520079759, 7.34341102594884e-05,
		0.0128109553215939, 2.15410908028565e-05, 0.999917936112624;
	r4 << 0.999298181098811, 0.00381266859973690, -0.0372640417930682,
		-0.00381416120190226, 0.999992725579099, 3.10356377494695e-05,
		0.0372638890473453, 0.000111117206080898, 0.999305453915886;
	R.push_back(r1);
	R.push_back(r2);
	R.push_back(r3);
	R.push_back(r4);

	MatrixXd D;
	D.setZero(4, 1);

	int im_n = 4;
	comput_mosaic_parameters(M, imsize, R, D, im_n);
}





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
