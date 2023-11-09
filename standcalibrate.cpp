#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
#include <vector> 
#include <string> 
#include <io.h>


using namespace std;
using namespace cv;

#define  BLACKROWS   7 
#define  BLACKCOLS  10 
#define  BLOCKSIZE  25
#define  FACTOR     1.0f   //缩放比例

const char ESC_KEY = 27;

void getFiles(string path, vector<string>& files);
bool getAnglesformR(Mat rotation, double &angleX, double &angleY, double &angleZ);
Scalar randomColor(int64 seed);

void main()
{
	ofstream fout("../Correction_Output/caliberation_result.txt");   /* 保存标定结果的文件 */
    //######################################################################
	cout << "开始提取角点………………" << endl;
	//######################################################################
	int  image_count = 0;  /* 图像数量 */
	Size image_size;       /* 图像的尺寸 */
	Size board_size = Size(BLACKCOLS ,BLACKROWS);  /* 标定板上每行、列的角点数 */
	vector<Point2f>         image_points_buf;      /* 缓存每幅图像上检测到的角点 */
	vector<vector<Point2f>> image_points_seq;      /* 保存检测到的所有角点 */
	int count = -1;        //用于存储角点个数。
	string strParentPath("..\\Calib_img");
	vector<string> files;
	getFiles(strParentPath, files);
	int number = files.size();

	double rate;
	int  wndWidth, wndHeight;

	namedWindow("相机标定", CV_WINDOW_NORMAL);
	
	Mat imageSrc, imageZoom;
	for (int i = 0; i < number; i++)
	{
		image_count++;
		// 用于观察检验输出
		//cout << "图像编号" << image_count << endl << "文件路径:"<< files[i].c_str() << endl;
		imageSrc = imread(files[i],CV_LOAD_IMAGE_ANYCOLOR);
		resize(imageSrc, imageZoom, Size(), FACTOR, FACTOR, 1);
		if (image_count == 1)  //读入第一张图片时获取图像宽高信息
		{
			image_size.width = imageSrc.cols;
			image_size.height = imageSrc.rows;
			//cout << "图像尺寸[" << image_size.width << "," << image_size.height << "]" << endl;
			//照片的尺寸与图像窗口的比例
			rate = 800.0f / image_size.width;
			wndWidth  = image_size.width * rate;
			wndHeight = image_size.height * rate;
			resizeWindow("相机标定", wndWidth, wndHeight);
		}
		
		/* 提取角点 */
		if (0 == findChessboardCorners(imageZoom, board_size, image_points_buf, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FAST_CHECK | CV_CALIB_CB_NORMALIZE_IMAGE))
		{
			//cout << "无法找到角点!\n"; //找不到角点
		}
		else
		{
			cout << "图像编号" << image_count << endl << "文件路径:" << files[i].c_str() << endl;
			Mat imageGray;
			cvtColor(imageZoom, imageGray, CV_RGB2GRAY);
			/* 亚像素精确化 */
			//find4QuadCornerSubpix(imageGray, image_points_buf, Size(5, 5)); //对粗提取的角点进行精确化
			cornerSubPix(imageGray, image_points_buf, Size(11, 11),Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
			for (int t = 0; t < (int)image_points_buf.size(); t++)
			{
				image_points_buf[t].x /= FACTOR;
				image_points_buf[t].y /= FACTOR;
				if (t != 0)
				{
					line(imageSrc, image_points_buf[t], image_points_buf[t - 1], randomColor(cv::getTickCount()), 2);
				}
				//std::stringstream StrStm;
				//StrStm.str("");
				//StrStm << t;
				//putText(imageSrc, StrStm.str(), image_points_buf[t], 3, 3.0f, CV_RGB(255, 0, 0), 2, 8, false);
			}
			image_points_seq.push_back(image_points_buf);  //保存亚像素角点
			drawChessboardCorners(imageSrc, board_size, image_points_buf, false); //用于在图片中标记角点
			imshow("相机标定", imageSrc);//显示图片
			waitKey(300);//暂停0.5S	
			imageGray.release();
		}
		imageSrc.release();
	}

	int total = image_points_seq.size();
	if (total < 4)
	{
		cout << "样本数目不足!\n"; //找不到角点
		waitKey(500);//暂停0.5S		
		return;
	}

	//cout << "成功样本总数:" << total << endl;
	//int CornerNum = board_size.width * board_size.height;  //每张图片上总的角点数
	//for (int ii = 0; ii < total; ii++)
	//{
	//	if (0 == ii % CornerNum)     // 每幅图片的角点个数。此判断语句是为了输出 图片号，便于控制台观看 
	//	{
	//		int i = -1;
	//		i = ii / CornerNum;
	//		cout << "--> 第 " << i + 1 << "图片的数据 --> : " << endl;
	//	}

	//	if (0 == ii % 3)	// 此判断语句，格式化输出，便于控制台查看
	//	{
	//		cout << endl;
	//	}
	//	else
	//	{
	//		cout.width(10);
	//	}
	//	//输出所有的角点
	//	cout << "[" << image_points_seq[ii][0].x << "," << image_points_seq[ii][0].y << "]";
	//}
	cout << "角点提取完成！\n";
	//以下是摄像机标定
	cout << "开始标定………………";

	/*棋盘三维信息*/
	Size square_size = Size(BLOCKSIZE, BLOCKSIZE);          /* 实际测量得到的标定板上每个棋盘格的大小 */
	vector<vector<Point3f>> object_points;                  /* 保存标定板上角点的三维坐标 */

	Mat cameraMatrix = Mat::zeros(3, 3, CV_64F);      /*相机内参*/
	//if (CV_CALIB_FIX_ASPECT_RATIO & CV_CALIB_FIX_ASPECT_RATIO)
	//	cameraMatrix.at<double>(0, 0) = 1.0;
	Mat distCoeffs = Mat::zeros(4, 1, CV_64F);      /*畸变系数*/
	vector<int> point_counts;                                 // 每幅图像中角点的数量
	vector<Vec3d> rvecsMat;  /* 每幅图像的旋转向量 */
	vector<Vec3d> tvecsMat;  /* 每幅图像的平移向量 */
	int i, j, t;
	for (t = 0; t<total; t++)        /* 初始化标定板上角点的三维坐标 */
	{
		vector<Point3f> tempPointSet;
		for (i = 0; i<board_size.height; i++)
		{
			for (j = 0; j<board_size.width; j++)
			{
				Point3f realPoint;
				/* 假设标定板放在世界坐标系中z=0的平面上 */
				realPoint.x = i*square_size.width;
				realPoint.y = j*square_size.height;
				realPoint.z = 0;
				tempPointSet.push_back(realPoint);
			}
		}
		object_points.push_back(tempPointSet);
	}
	/* 初始化每幅图像中的角点数量，假定每幅图像中都可以看到完整的标定板 */
	for (i = 0; i<image_count; i++)
	{
		point_counts.push_back(board_size.width*board_size.height);
	}

#if 0
	/* 开始标定 */
	calibrateCamera(object_points, image_points_seq, image_size, cameraMatrix, distCoeffs, rvecsMat, tvecsMat, CV_CALIB_FIX_ASPECT_RATIO | CV_CALIB_FIX_K4 | CV_CALIB_FIX_K5);
	cout << "标定完成！\n";
#else
	int flags = 0;
	flags |= cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC;
	flags |= cv::fisheye::CALIB_CHECK_COND;
	flags |= cv::fisheye::CALIB_FIX_SKEW;
	fisheye::calibrate(object_points, image_points_seq, image_size, cameraMatrix, distCoeffs, rvecsMat, tvecsMat, flags, cv::TermCriteria(3, 20, 1e-6));
	cout << "定标完成！\n";
#endif

	//对标定结果进行评价
	cout << "开始评价标定结果………………\n";
	double total_err = 0.0; /* 所有图像的平均误差的总和 */
	double err = 0.0; /* 每幅图像的平均误差 */
	vector<Point2f> image_points2;    /* 保存重新计算得到的投影点 */
	cout << "\t每幅图像的标定误差：\n";
	fout << "每幅图像的标定误差：\n";
 
	for (i = 0; i<total; i++)
	{
		vector<Point3f>& tempPointSet = object_points[i];
		/* 通过得到的摄像机内外参数，对空间的三维点进行重新投影计算，得到新的投影点 */
		projectPoints(tempPointSet, rvecsMat[i], tvecsMat[i], cameraMatrix, distCoeffs, image_points2);
 
		/* 计算新的投影点和旧的投影点之间的误差*/
		vector<Point2f>& tempImagePoint = image_points_seq[i];
		Mat tempImagePointMat = Mat(1, tempImagePoint.size(), CV_32FC2);
		Mat image_points2Mat = Mat(1, image_points2.size(), CV_32FC2);
		for (int j = 0; j < tempImagePoint.size(); j++)
		{
			image_points2Mat.at<Vec2f>(0, j) = Vec2f(image_points2[j].x, image_points2[j].y);
			tempImagePointMat.at<Vec2f>(0, j) = Vec2f(tempImagePoint[j].x, tempImagePoint[j].y);
		}
		err = norm(image_points2Mat, tempImagePointMat, NORM_L2);
		total_err += err /= point_counts[i];
		std::cout << "第" << i + 1 << "幅图像的平均误差：" << err << "像素" << endl;
		fout << "第" << i + 1 << "幅图像的平均误差：" << err << "像素" << endl;
	}
	std::cout << "总体平均误差：" << total_err / image_count << "像素" << endl;
	fout << "总体平均误差：" << total_err / image_count << "像素" << endl << endl;
	std::cout << "评价完成！" << endl;
	point_counts.clear();

	//保存定标结果  
	Mat rotation_matrix = Mat(3, 3, CV_32FC1, Scalar::all(0)); /* 保存每幅图像的旋转矩阵 */
	std::cout << "开始保存定标结果………………" << endl;
	fout << "相机内参数矩阵：" << endl;
	fout << cameraMatrix << endl << endl;
	fout << "畸变系数：\n";
	fout << distCoeffs << endl << endl << endl;
	for (int i = 0; i<total; i++)
	{
		fout << "第" << i + 1 << "幅图像的旋转向量：" << endl;
		fout << rvecsMat[i] << endl;
		/* 将旋转向量转换为相对应的旋转矩阵 */
		Rodrigues(rvecsMat[i], rotation_matrix);
		fout << "第" << i + 1 << "幅图像的旋转矩阵：" << endl;
		fout << rotation_matrix << endl;
		fout << "第" << i + 1 << "幅图像的平移向量：" << endl;
		fout << tvecsMat[i] << endl << endl;
	}
	std::cout << "完成保存" << endl;
	fout << endl;

	imageZoom.release();

	/************************************************************************
	显示定标结果
	*************************************************************************/
	Mat mapx = Mat(image_size, CV_32FC1);
	Mat mapy = Mat(image_size, CV_32FC1);
	Mat R = Mat::eye(3, 3, CV_32F);
	cout << "保存矫正图像" << endl;
	for (int i = 0; i != total; i++)
	{
		cout << "Frame #" << i + 1 << "..." << endl;
		Mat newCameraMatrix = Mat(3, 3, CV_32FC1, Scalar::all(0));
		fisheye::initUndistortRectifyMap(cameraMatrix, distCoeffs, R, cameraMatrix, image_size, CV_32FC1, mapx, mapy);
		Mat temp = imread(files[i], CV_LOAD_IMAGE_ANYCOLOR);
		Mat t = temp.clone();
		cv::remap(temp, t, mapx, mapy, INTER_LINEAR);
		string imageFileName;
		std::stringstream StrStm;
		StrStm << i + 1;
		StrStm >> imageFileName;
		imageFileName += "_d.jpg";
		imwrite("../Correction_Output/img"+imageFileName, t);
	}
	std::cout << "保存结束" << endl;

	/************************************************************************
	测试一张图片
	*************************************************************************/
	if (0)
	{
		cout << "TestImage ..." << endl;
		Mat testImage = imread("../Calib_img/img1.bmp", 1);
		fisheye::initUndistortRectifyMap(cameraMatrix, distCoeffs, R, cameraMatrix, image_size, CV_32FC1, mapx, mapy);
		//fisheye::initUndistortRectifyMap(intrinsic_matrix,distortion_coeffs,R,
		//   getOptimalNewCameraMatrix(intrinsic_matrix, distortion_coeffs, image_size, 1, image_size, 0),image_size,CV_32FC1,mapx,mapy);
		Mat t = testImage.clone();
		cv::remap(testImage, t, mapx, mapy, INTER_LINEAR);
		imwrite("../Correction_Output/TestOutput.bmp", t);
		cout << "保存结束" << endl;
	}

	return;
}



Scalar randomColor(int64 seed)
{
	RNG rng(seed);
	int icolor = (unsigned)rng;
	return Scalar(icolor & 255, (icolor >> 8) & 255, (icolor >> 16) & 255);
}

/**
* @brief getAnglesformR
* @param rotation 输入旋转矩阵
* @return 返回角度
*/
bool getAnglesformR(Mat rotation, double &angleX, double &angleY, double &angleZ)
{
	//theta_{x} = atan2(r_{32}, r_{33})
	angleX = std::atan2(rotation.at<double>(2, 1), rotation.at<double>(2, 2));

	//      theta_{y} = atan2(-r_{31}, sqrt{r_{32}^2 + r_{33}^2})
	double tmp0 = rotation.at<double>(2, 0);
	double tmp1 = rotation.at<double>(2, 1) * rotation.at<double>(2, 1);
	double tmp2 = rotation.at<double>(2, 2) * rotation.at<double>(2, 2);
	angleY = std::atan2(-tmp0, sqrt(tmp1 + tmp2));

	//      theta_{z} = atan2(r_{21}, r_{11})
	angleZ = std::atan2(rotation.at<double>(1, 0), rotation.at<double>(0, 0));

	angleX *= (180 / CV_PI);
	angleY *= (180 / CV_PI);
	angleZ *= (180 / CV_PI);

	return true;
}

//遍历本地文件
void getFiles(string path, vector<string>& files)
{
	long   hFile = 0;
	struct _finddata_t fileinfo;
	string p;
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			if ((fileinfo.attrib &  _A_SUBDIR))
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					getFiles(p.assign(path).append("\\").append(fileinfo.name), files);
			}
			else
			{
				files.push_back(p.assign(path).append("\\").append(fileinfo.name));
			}
		} while (_findnext(hFile, &fileinfo) == 0);

		_findclose(hFile);
	}
}