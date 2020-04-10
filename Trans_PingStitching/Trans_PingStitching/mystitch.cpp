#include "transforms.h"

#include "highgui/highgui.hpp"   
#include "opencv2/legacy/legacy.hpp"

#include <map>

using namespace std;
using namespace cv;

//计算能量函数
Mat EnergyOfImg(Mat& source, Mat& goal, int start, int width)//goal为仿射变换之后的图
{
	Mat left1(source(Rect(start, 0, width, source.rows)));
	Mat right1(goal(Rect(start, 0, width, source.rows)));
	Mat left, right;
	cvtColor(left1, left, CV_RGB2GRAY);//把图片转化为灰度图
	cvtColor(right1, right, CV_RGB2GRAY);

	//计算亮度差异
	Mat Ecolor(source.rows, width, CV_8UC1, Scalar::all(0));
	float ld, rd;//左右两边图像的距离差
	//计算距离差异
	Mat Edistance(source.rows, width, CV_8UC1, Scalar::all(0));
	int maxdistance = 2 * sqrt(pow(0 - source.rows / 2, 2) + pow(0 - source.cols / 2, 2));
	//计算结构差异
	Mat g_left_x, g_right_x, g_left_y, g_right_y;
	Mat Egeometry(source.rows, width, CV_8UC1, Scalar::all(0));

	Scharr(left, g_left_x, CV_8UC1, 1, 0);
	Scharr(left, g_left_y, CV_8UC1, 0, 1);
	Scharr(right, g_right_x, CV_8UC1, 1, 0);
	Scharr(right, g_right_y, CV_8UC1, 0, 1);

	int r = 500;//1025;//5cm半径400dpi对应像素点
	int y;
	float b, seita, l;
	//能量图
	float w1 = 0.2, w2 = 0.8, w3 = 0;//亮度差和结构差的权值
	Mat energyImg(source.rows, width, CV_8UC1, Scalar::all(0));
	int disval, colval, geoval, enerval;
	int num = 0;
	for (int i = 0; i < source.rows; i++)
	{
		for (int j = 0; j < width; j++)
		{
			y = abs(i - source.rows / 2);
			b = (float)y / r;
			seita = asin(b);
			l = seita*r;
			ld = sqrt(pow(l, 2) + pow(j + start - source.cols / 2, 2));
			rd = sqrt(pow(l, 2) + pow(j + start - source.cols / 2, 2));
			disval = 255 * float((ld + rd) / maxdistance);
			colval = abs(left.at<uchar>(i, j) - right.at<uchar>(i, j));
			geoval = abs(sqrt(pow(g_left_x.at<uchar>(i, j), 2) + pow(g_left_y.at<uchar>(i, j), 2)) -
				sqrt(pow(g_right_x.at<uchar>(i, j), 2) + pow(g_right_y.at<uchar>(i, j), 2)));
			enerval = (float)w1*colval + (float)w2*geoval + (float)w3*disval;
			Edistance.at<uchar>(i, j) = disval;
			Ecolor.at<uchar>(i, j) = colval;
			Egeometry.at<uchar>(i, j) = geoval;
			energyImg.at<uchar>(i, j) = enerval;
		}
	}
	//imwrite("C:/Users/csp96/Desktop/能量函数图111.jpg", energyImg);
	return energyImg;
}

//动态规划思想 找出最佳缝合线
vector<int> bestline(Mat& image)
{
	map<int, vector<int>> res;//<总能量，路径>
	int left, mid, right;
	int minval;
	for (int i = 1; i < image.cols - 1; i++)//列，去掉了首尾列
	{
		vector<int> path;
		int sumEnerge = image.at<uchar>(0, i);
		path.push_back(i);
		int index = i;
		for (int j = 1; j < image.rows; j++)//行
		{
			mid = image.at<uchar>(j, index);
			if (index > 0 && index<image.cols - 1)
			{
				left = image.at<uchar>(j, index - 1);
				right = image.at<uchar>(j, index + 1);
				/*	if (mid>50|left>50|right>50)
				continue;*///
				minval = min(left, mid);
				minval = min(minval, right);
			}
			else if (index == image.cols - 1)//碰到了右边界
			{
				left = image.at<uchar>(j, index - 1);
				minval = min(mid, left);
			}
			else//碰到了左边界
			{
				right = image.at<uchar>(j, index + 1);
				minval = min(mid, right);
			}

			sumEnerge += minval;
			//将最小值的横坐标放入path中
			if (minval == left&&index>0)
			{
				path.push_back(index - 1);
				index = index - 1;
			}
			else if (minval == right&&index<image.cols - 1)
			{
				path.push_back(index + 1);
				index = index + 1;
			}
			else
			{
				path.push_back(index);
			}
		}
		//当前路径的总能量和,放到path最后一位
		res.insert(pair<int, vector<int>>(sumEnerge, path));
	}
	//map根据key自动排序，map的第一个元素即为最佳缝合线

	//输出缝合线
	vector<int> path = res.begin()->second;
	for (int i = 0; i < image.rows; i++)
		image.at<uchar>(i, path[i]) = 255;
	//imwrite("C:/Users/csp96/Desktop/能量函数图.jpg", image);
	return res.begin()->second;
}

//融合方法
void OptimizeSeam(Mat& img1, Mat& trans, Mat& dst, vector<int>& path, int start, int processWidth)
{
	int rows = dst.rows;
	int cols = img1.cols; //注意，是列数*通道数
	double alpha = 1;//img1中像素的权重  
	double b;
	int mid;
	//👇消除灰度差 
	//左右重叠区域的灰度均值
	Mat img1_g, trans_g;
	cvtColor(img1, img1_g, CV_RGB2GRAY);
	cvtColor(trans, trans_g, CV_RGB2GRAY);
	float gray_left = 0;
	float gray_right = 0;
	for (int i = start; i < img1.cols; i++)
	{
		for (int j = 0; j < img1.rows; j++)
		{
			gray_left += img1_g.at<uchar>(j, i);
			gray_right += trans_g.at<uchar>(j, i);
		}
	}
	gray_left /= processWidth*rows;
	gray_right /= processWidth*rows;

	//修正拼接缝两边的亮度
	int k = gray_right > gray_left ? 1 : -1;
	float modval;//修正值
	int mod_width = 0;
	for (int i = 0; i < rows; i++)
	{
		uchar* p = img1.ptr<uchar>(i);  //获取第i行的首地址
		uchar* t = trans.ptr<uchar>(i);
		uchar* d = dst.ptr<uchar>(i);
		for (int j = start; j < cols; j++)
		{
			//最佳缝合线
			mid = path[i] + start;
			if (j <(mid - mod_width))
			{
				alpha = 1;
				b = (float)(mid - j) / (mid - start);
				modval = abs(gray_left - gray_right)*k*(float)(1 - b) / 2;
			}
			else if (j >(mid + mod_width))
			{
				alpha = 0;
				b = (float)(mid - j) / (mid - start);
				modval = -abs(gray_left - gray_right)*k*(float)(1 - b) / 2;

			}
			else
				alpha = 0.5;
			modval = 0;
			//alpha = (processWidth - (j - start)) / processWidth;
			d[j * 3] = p[j * 3] * alpha + t[j * 3] * (1 - alpha) + modval;
			d[j * 3 + 1] = p[j * 3 + 1] * alpha + t[j * 3 + 1] * (1 - alpha) + modval;
			d[j * 3 + 2] = p[j * 3 + 2] * alpha + t[j * 3 + 2] * (1 - alpha) + modval;
			////画出拼接缝
			if (j == mid){ d[j * 3] = 255; d[j * 3 + 1] = 0; d[j * 3 + 2] = 0; }
		}
	}
}


//缝合
Mat seaming(vector<Mat>& images, vector<int>& m_u0, vector<int>m_u1)
{
	//图像融合
	int height = images[0].rows;
	int width = images[0].cols;
	Mat dst(height, width, CV_8UC3);
	Mat right_img(height, width, CV_8UC3);
	Mat left_img(height, width, CV_8UC3);
	right_img = images[0];//初始化
	for (int i = 0; i < images.size() - 1; i++)//四张图 融合三次
	{

		int overla_start = m_u0[i];
		int overlap_width = m_u1[i + 1] - m_u0[i];//重叠区域的

		cout << "第" << (i + 1) << "次缝合" << endl;

		left_img = images[i + 1];

		dst.setTo(0);
		left_img.copyTo(dst(Rect(0, 0, width, height)));

		//最佳缝合线
		//1
		Mat energy = EnergyOfImg(left_img, right_img, overla_start, overlap_width);//能量函数修改
		//2
		vector<int> energy_path = bestline(energy);
		//3
		OptimizeSeam(left_img, right_img, dst, energy_path, overla_start, overlap_width);//缝合
		//保存
		string savepath = "C:\\Users\\Administrator\\Desktop\\csp\\Rew_Multiple_views\\images\\core\\";
		savepath += to_string(i + 1) + "and" + to_string(i + 2) + ".jpg";
		imwrite(savepath, dst);

		////更新右图
		right_img.setTo(0);
		dst.copyTo(right_img(Rect(0, 0, width, height)));
	}
	return dst;
}