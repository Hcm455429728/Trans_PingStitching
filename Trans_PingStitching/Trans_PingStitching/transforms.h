#ifndef TRANSFORMS_H
#define TRANSFORMS_H

#include <Eigen/Dense>
using namespace Eigen;

/*****p2e的三个函数*******/
/*function[xcam, ycam, zcam] = trans_persp2cam(u, v, M, D)*/
void trans_persp2cam(/*传出参数*/MatrixXd& xcam, MatrixXd& ycam, MatrixXd& zcam,
					 /*传入参数*/MatrixXd& u,	 MatrixXd& v,	 MatrixXd& M,	 MatrixXd& D);

/*func:function [u, v] = trans_cam2equi(xcam, ycam, zcam, fe)*/
void trans_cam2equi(/*传出参数*/MatrixXd& u, MatrixXd& v,
					/*传入参数*/MatrixXd& xcam, MatrixXd& ycam, MatrixXd& zcam, double fe);

/*func:function [u, v] = trans_persp2equi(x, y, R, M, D, fe)*/
void trans_persp2equi(/*传出参数*/MatrixXd& u, MatrixXd& v,
					  /*传入参数*/MatrixXd& x, MatrixXd& y,
						          MatrixXd& R, MatrixXd& M, MatrixXd& D, double fe);



/*****e2p的三个函数*******/
/*function [xcam, ycam, zcam] = trans_equi2cam(u, v, fe)*/
void trans_equi2cam(/*传出参数*/MatrixXd& xcam, MatrixXd& ycam, MatrixXd& zcam,
					/*传入参数*/MatrixXd& u,	MatrixXd& v,	double fe);

/*function [u, v] = trans_cam2persp(xcam, ycam, zcam, M, D)*/
void trans_cam2persp(/*传出参数*/MatrixXd& u,	 MatrixXd& v,
					 /*传入参数*/MatrixXd& xcam, MatrixXd& ycam, MatrixXd& zcam,
								 MatrixXd& M,	 MatrixXd& D);

/*function [x, y] = trans_equi2persp(u, v, R, M, D, fe)*/
void trans_equi2persp(/*传出参数*/MatrixXd& x, MatrixXd& y,
					  /*传入参数*/MatrixXd& u, MatrixXd& v,
								  MatrixXd& R, MatrixXd& M, MatrixXd& D, double fe);


/*****p2p的三个函数*******/
//p2c和c2p已经实现了
/*function [x, y] = trans_persp2persp(u, v, R, M1, D1, M2, D2)*/
void trans_persp2persp(/*传出参数*/MatrixXd& x, MatrixXd& y,
					   /*传入参数*/MatrixXd& u, MatrixXd& v, MatrixXd& R,
								   MatrixXd& M1, MatrixXd& D1, MatrixXd& M2, MatrixXd& D2);


#endif // !TRANSFORMS





