
#include "transforms.h"

/*****p2e的三个函数*******/

/*function[xcam, ycam, zcam] = trans_persp2cam(u, v, M, D)*/
void trans_persp2cam(/*传出参数*/MatrixXd& xcam, MatrixXd& ycam, MatrixXd& zcam,
	/*传入参数*/MatrixXd& u, MatrixXd& v, MatrixXd& M, MatrixXd& D)
{
	double fx = M(0, 0), fy = M(1, 1);
	double cx = M(0, 2), cy = M(1, 2);
	double k1, k2, p1, p2, k3, k4, k5, k6;
	if (D(0) != 0)
	{
		k1 = D(0); k2 = D(1); p1 = D(2); p2 = D(3);
		k3 = D(4); k4 = D(5); k5 = D(6); k6 = D(7);
	}
	//x_d = (u - cx) / fx; “矩阵-常数”的操作 //R.array() -= s; ----> R = R - s
	xcam = (u.array() - cx) / fx;
	ycam = (v.array() - cy) / fy;

	//if D!=0 跳过了
	zcam.setOnes(xcam.rows(), xcam.cols());
}

/*
para:任意大小的矩阵 xcam ycam zcam fe
return:存放了u, v的容器
func:function [u, v] = trans_cam2equi(xcam, ycam, zcam, fe)
*/
void trans_cam2equi(/*传出参数*/MatrixXd& u, MatrixXd& v,
	/*传入参数*/MatrixXd& xcam, MatrixXd& ycam, MatrixXd& zcam, double fe)
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

	u = fe*thera;
	v = fe*phi;
}

/*
para:矩阵 x(1xN) y(1xN) R(3x3) M(3x3) D(一直都是0) double型fe
return:存放了u, v的容器
func:function [u, v] = trans_persp2equi(x, y, R, M, D, fe)
*/
void trans_persp2equi(/*传出参数*/MatrixXd& u, MatrixXd& v,
	/*传入参数*/MatrixXd& x, MatrixXd& y,
	MatrixXd& R, MatrixXd& M, MatrixXd& D, double fe)
{

	MatrixXd xcam, ycam, zcam;
	trans_persp2cam(xcam, ycam, zcam, x, y, M, D);
	MatrixXd xr = R(0, 0) * xcam + R(0, 1) * ycam + R(0, 2) * zcam;
	MatrixXd yr = R(1, 0) * xcam + R(1, 1) * ycam + R(1, 2) * zcam;
	MatrixXd zr = R(2, 0) * xcam + R(2, 1) * ycam + R(2, 2) * zcam;

	//测试输出,没有问题
	trans_cam2equi(u, v, xr, yr, zr, fe);
}


/*****e2p的三个函数*******/

/*function [xcam, ycam, zcam] = trans_equi2cam(u, v, fe)*/
void trans_equi2cam(/*传出参数*/MatrixXd& xcam, MatrixXd& ycam, MatrixXd& zcam,
	/*传入参数*/MatrixXd& u, MatrixXd& v, double fe)
{
	MatrixXd theta = u.array() / fe;//theta = u ./ fe ;
	MatrixXd phi = v.array() / fe;
	ycam = phi.array().sin();
	MatrixXd temp1 = phi.array().cos();
	MatrixXd temp2 = theta.array().cos();
	MatrixXd temp3 = theta.array().sin();
	zcam = temp1.array()*temp2.array();
	xcam = temp1.array()*temp3.array();
}

/*function [u, v] = trans_cam2persp(xcam, ycam, zcam, M, D)*/
void trans_cam2persp(/*传出参数*/MatrixXd& u, MatrixXd& v,
	/*传入参数*/MatrixXd& xcam, MatrixXd& ycam, MatrixXd& zcam,
	MatrixXd& M, MatrixXd& D)
{
	double fx = M(0, 0), fy = M(1, 1);
	double cx = M(0, 2), cy = M(1, 2);
	double k1, k2, p1, p2, k3, k4, k5, k6;
	if (D(0) != 0)//实际上跳过了
	{
		k1 = D(0); k2 = D(1); p1 = D(2); p2 = D(3);
		k3 = D(4); k4 = D(5); k5 = D(6); k6 = D(7);
	}
	MatrixXd x = xcam.array() / zcam.array();
	MatrixXd y = ycam.array() / zcam.array();

	//x(zcam<=0) = 100*cx; y(zcam <= 0) = 100 * cy;
	for (int i = 0; i < zcam.rows(); i++)//(xcam > 0)
	{
		for (int j = 0; j < zcam.cols(); j++)
		{
			if (zcam(i, j) <= 0)
			{
				x(i, j) = 100 * cx;
				y(i, j) = 100 * cy;
			}
		}
	}
	//if D ~= 0 不满足，直接跳过了
	u = fx*x.array() + cx;
	v = fy*y.array() + cy;
}

/*function [x, y] = trans_equi2persp(u, v, R, M, D, fe)*/
void trans_equi2persp(/*传出参数*/MatrixXd& x, MatrixXd& y,
	/*传入参数*/MatrixXd& u, MatrixXd& v,
	MatrixXd& R, MatrixXd& M, MatrixXd& D, double fe)
{
	MatrixXd xcam, ycam, zcam;
	trans_equi2cam(xcam, ycam, zcam, u, v, fe);

	MatrixXd xr = R(0, 0) * xcam + R(0, 1) * ycam + R(0, 2) * zcam;
	MatrixXd yr = R(1, 0) * xcam + R(1, 1) * ycam + R(1, 2) * zcam;
	MatrixXd zr = R(2, 0) * xcam + R(2, 1) * ycam + R(2, 2) * zcam;

	trans_cam2persp(x, y, xr, yr, zr, M, D);
}


/*****p2p的三个函数*******/
//p2c和c2p已经实现了
/*function [x, y] = trans_persp2persp(u, v, R, M1, D1, M2, D2)*/
void trans_persp2persp(/*传出参数*/MatrixXd& x, MatrixXd& y,
	/*传入参数*/MatrixXd& u, MatrixXd& v, MatrixXd& R,
	MatrixXd& M1, MatrixXd& D1, MatrixXd& M2, MatrixXd& D2)
{
	MatrixXd xcam, ycam, zcam;
	trans_persp2cam(xcam, ycam, zcam, u, v, M1, D1);

	MatrixXd xr = R(0, 0) * xcam + R(0, 1) * ycam + R(0, 2) * zcam;
	MatrixXd yr = R(1, 0) * xcam + R(1, 1) * ycam + R(1, 2) * zcam;
	MatrixXd zr = R(2, 0) * xcam + R(2, 1) * ycam + R(2, 2) * zcam;

	trans_cam2persp(x, y, xr, yr, zr, M2, D2);
}
