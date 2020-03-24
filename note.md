# 调用库的准备 #

* 加载Vlfeat库  跨平台图像处理库

* eigen C++下的矩阵操作库

* transform加载  投影类型？

  

# 计时器 #

* 读图+特征点检测    32s

* 特征点匹配   4min 50s

* Ransac   8s

* global transforms estimation 2s

* computing mosaic parameters 1s

  

# Rew_city #

* 文件名数组
* 图片数组
* (n-1)*2 重合边数组
* 图像大小数组



# Rew_Mosaic #

* 参数列表

  %   im : the input images
  %   edge_list: the overllapped edge list, each row represents an image pair to be aligned
  %   refi: the index of the reference image, 0 for automatic straithtening  //0表示自动扫描
  %   projection_type: the projection type of the mosaic,    //只有persp和equi两种
  %                    'persp' (default) for perspective projection 
  %                    and 'equi' for equirectangular projection.
  %   ransac_threshold： threshold of global ransac
  %                     0 for default value which is set to 0.1;
  %   imfolder： The folder containing the input images,
  %              and saving the resulted mosaic, if needed.

* 三个加权参数

  lambda = 0.001 * imsize(1,1)*imsize(1,2); //平衡拟合项和平滑项的加权参数

  intv_mesh = 10;  //计算变形函数的像素间隔

  K_smooth = 5; //在非重叠区域中的平滑过渡宽度被设置为最大偏差的K-smooth倍

* 特征点检测与匹配

  X_1/X_2  二维矩阵  最后一行全部是1  matlab补行用 ' ; ' 隔开

  ​				取值是：三维矩阵points的第i个二维矩阵，在这个二维矩阵中取第一行和第二行为X_1的前两行, 并				且取的这两行的列值与matches第1行所记得列值相同

* Ransac 参数  

  200最大估算次数   ransac_threshold置信概率  均是越大越准确    ok是计算所得X_1与X_2中合格的列

* X 矩阵  是四维矩阵  两列  每列中是该对图像中各自的筛选后的配准点

* global_paras.mat计算

  optimoptions 优化器  为lsqnonlin设置优化项   levenberg-marquardt  即LM算法利用该算法来实现最小二乘

  lsqnonlin  求解非线性最小二乘（非线性数据拟合）问题

  ​				  通过函数residual_all_robust拟合配准点矩阵X   返回单应性矩阵的最佳参数

  

  A.*B  *进行此运算时必须保证矩阵A和B的形状一样，即同为m*n矩阵。运算结果为对应位置的元素相乘组成同样形状（m*n）的矩阵
  
  A‘ 矩阵转置
  
  R_pair{i, j}  未利用参考图像修正前 图片i和j之间的单应性矩阵
  
  R  参考图像修正之后 图像与参考图像之间的单应性矩阵
  
  transtrans_persp2equi->trans_persp2cam->trans_cam2equi

* computing mosaic parameters   计算单应性投影后图像在融合时的位置

* 实际的变换使靠什么完成的？

  只在网格处进行变换

  interp2  双线性插值函数

* f  are  bias.    P 齐次投影坐标  I难道是对角阵？









​	



