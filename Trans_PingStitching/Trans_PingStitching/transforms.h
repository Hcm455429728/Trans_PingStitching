#ifndef TRANSFORMS_H
#define TRANSFORMS_H

#include <Eigen/Dense>
using namespace Eigen;

/*****p2e����������*******/
/*function[xcam, ycam, zcam] = trans_persp2cam(u, v, M, D)*/
void trans_persp2cam(/*��������*/MatrixXd& xcam, MatrixXd& ycam, MatrixXd& zcam,
					 /*�������*/MatrixXd& u,	 MatrixXd& v,	 MatrixXd& M,	 MatrixXd& D);

/*func:function [u, v] = trans_cam2equi(xcam, ycam, zcam, fe)*/
void trans_cam2equi(/*��������*/MatrixXd& u, MatrixXd& v,
					/*�������*/MatrixXd& xcam, MatrixXd& ycam, MatrixXd& zcam, double fe);

/*func:function [u, v] = trans_persp2equi(x, y, R, M, D, fe)*/
void trans_persp2equi(/*��������*/MatrixXd& u, MatrixXd& v,
					  /*�������*/MatrixXd& x, MatrixXd& y,
						          MatrixXd& R, MatrixXd& M, MatrixXd& D, double fe);



/*****e2p����������*******/
/*function [xcam, ycam, zcam] = trans_equi2cam(u, v, fe)*/
void trans_equi2cam(/*��������*/MatrixXd& xcam, MatrixXd& ycam, MatrixXd& zcam,
					/*�������*/MatrixXd& u,	MatrixXd& v,	double fe);

/*function [u, v] = trans_cam2persp(xcam, ycam, zcam, M, D)*/
void trans_cam2persp(/*��������*/MatrixXd& u,	 MatrixXd& v,
					 /*�������*/MatrixXd& xcam, MatrixXd& ycam, MatrixXd& zcam,
								 MatrixXd& M,	 MatrixXd& D);

/*function [x, y] = trans_equi2persp(u, v, R, M, D, fe)*/
void trans_equi2persp(/*��������*/MatrixXd& x, MatrixXd& y,
					  /*�������*/MatrixXd& u, MatrixXd& v,
								  MatrixXd& R, MatrixXd& M, MatrixXd& D, double fe);


/*****p2p����������*******/
//p2c��c2p�Ѿ�ʵ����
/*function [x, y] = trans_persp2persp(u, v, R, M1, D1, M2, D2)*/
void trans_persp2persp(/*��������*/MatrixXd& x, MatrixXd& y,
					   /*�������*/MatrixXd& u, MatrixXd& v, MatrixXd& R,
								   MatrixXd& M1, MatrixXd& D1, MatrixXd& M2, MatrixXd& D2);


#endif // !TRANSFORMS





