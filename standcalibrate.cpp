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
#define  FACTOR     1.0f   //���ű���

const char ESC_KEY = 27;

void getFiles(string path, vector<string>& files);
bool getAnglesformR(Mat rotation, double &angleX, double &angleY, double &angleZ);
Scalar randomColor(int64 seed);

void main()
{
	ofstream fout("../Correction_Output/caliberation_result.txt");   /* ����궨������ļ� */
    //######################################################################
	cout << "��ʼ��ȡ�ǵ㡭����������" << endl;
	//######################################################################
	int  image_count = 0;  /* ͼ������ */
	Size image_size;       /* ͼ��ĳߴ� */
	Size board_size = Size(BLACKCOLS ,BLACKROWS);  /* �궨����ÿ�С��еĽǵ��� */
	vector<Point2f>         image_points_buf;      /* ����ÿ��ͼ���ϼ�⵽�Ľǵ� */
	vector<vector<Point2f>> image_points_seq;      /* �����⵽�����нǵ� */
	int count = -1;        //���ڴ洢�ǵ������
	string strParentPath("..\\Calib_img");
	vector<string> files;
	getFiles(strParentPath, files);
	int number = files.size();

	double rate;
	int  wndWidth, wndHeight;

	namedWindow("����궨", CV_WINDOW_NORMAL);
	
	Mat imageSrc, imageZoom;
	for (int i = 0; i < number; i++)
	{
		image_count++;
		// ���ڹ۲�������
		//cout << "ͼ����" << image_count << endl << "�ļ�·��:"<< files[i].c_str() << endl;
		imageSrc = imread(files[i],CV_LOAD_IMAGE_ANYCOLOR);
		resize(imageSrc, imageZoom, Size(), FACTOR, FACTOR, 1);
		if (image_count == 1)  //�����һ��ͼƬʱ��ȡͼ������Ϣ
		{
			image_size.width = imageSrc.cols;
			image_size.height = imageSrc.rows;
			//cout << "ͼ��ߴ�[" << image_size.width << "," << image_size.height << "]" << endl;
			//��Ƭ�ĳߴ���ͼ�񴰿ڵı���
			rate = 800.0f / image_size.width;
			wndWidth  = image_size.width * rate;
			wndHeight = image_size.height * rate;
			resizeWindow("����궨", wndWidth, wndHeight);
		}
		
		/* ��ȡ�ǵ� */
		if (0 == findChessboardCorners(imageZoom, board_size, image_points_buf, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FAST_CHECK | CV_CALIB_CB_NORMALIZE_IMAGE))
		{
			//cout << "�޷��ҵ��ǵ�!\n"; //�Ҳ����ǵ�
		}
		else
		{
			cout << "ͼ����" << image_count << endl << "�ļ�·��:" << files[i].c_str() << endl;
			Mat imageGray;
			cvtColor(imageZoom, imageGray, CV_RGB2GRAY);
			/* �����ؾ�ȷ�� */
			//find4QuadCornerSubpix(imageGray, image_points_buf, Size(5, 5)); //�Դ���ȡ�Ľǵ���о�ȷ��
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
			image_points_seq.push_back(image_points_buf);  //���������ؽǵ�
			drawChessboardCorners(imageSrc, board_size, image_points_buf, false); //������ͼƬ�б�ǽǵ�
			imshow("����궨", imageSrc);//��ʾͼƬ
			waitKey(300);//��ͣ0.5S	
			imageGray.release();
		}
		imageSrc.release();
	}

	int total = image_points_seq.size();
	if (total < 4)
	{
		cout << "������Ŀ����!\n"; //�Ҳ����ǵ�
		waitKey(500);//��ͣ0.5S		
		return;
	}

	//cout << "�ɹ���������:" << total << endl;
	//int CornerNum = board_size.width * board_size.height;  //ÿ��ͼƬ���ܵĽǵ���
	//for (int ii = 0; ii < total; ii++)
	//{
	//	if (0 == ii % CornerNum)     // ÿ��ͼƬ�Ľǵ���������ж������Ϊ����� ͼƬ�ţ����ڿ���̨�ۿ� 
	//	{
	//		int i = -1;
	//		i = ii / CornerNum;
	//		cout << "--> �� " << i + 1 << "ͼƬ������ --> : " << endl;
	//	}

	//	if (0 == ii % 3)	// ���ж���䣬��ʽ����������ڿ���̨�鿴
	//	{
	//		cout << endl;
	//	}
	//	else
	//	{
	//		cout.width(10);
	//	}
	//	//������еĽǵ�
	//	cout << "[" << image_points_seq[ii][0].x << "," << image_points_seq[ii][0].y << "]";
	//}
	cout << "�ǵ���ȡ��ɣ�\n";
	//������������궨
	cout << "��ʼ�궨������������";

	/*������ά��Ϣ*/
	Size square_size = Size(BLOCKSIZE, BLOCKSIZE);          /* ʵ�ʲ����õ��ı궨����ÿ�����̸�Ĵ�С */
	vector<vector<Point3f>> object_points;                  /* ����궨���Ͻǵ����ά���� */

	Mat cameraMatrix = Mat::zeros(3, 3, CV_64F);      /*����ڲ�*/
	//if (CV_CALIB_FIX_ASPECT_RATIO & CV_CALIB_FIX_ASPECT_RATIO)
	//	cameraMatrix.at<double>(0, 0) = 1.0;
	Mat distCoeffs = Mat::zeros(4, 1, CV_64F);      /*����ϵ��*/
	vector<int> point_counts;                                 // ÿ��ͼ���нǵ������
	vector<Vec3d> rvecsMat;  /* ÿ��ͼ�����ת���� */
	vector<Vec3d> tvecsMat;  /* ÿ��ͼ���ƽ������ */
	int i, j, t;
	for (t = 0; t<total; t++)        /* ��ʼ���궨���Ͻǵ����ά���� */
	{
		vector<Point3f> tempPointSet;
		for (i = 0; i<board_size.height; i++)
		{
			for (j = 0; j<board_size.width; j++)
			{
				Point3f realPoint;
				/* ����궨�������������ϵ��z=0��ƽ���� */
				realPoint.x = i*square_size.width;
				realPoint.y = j*square_size.height;
				realPoint.z = 0;
				tempPointSet.push_back(realPoint);
			}
		}
		object_points.push_back(tempPointSet);
	}
	/* ��ʼ��ÿ��ͼ���еĽǵ��������ٶ�ÿ��ͼ���ж����Կ��������ı궨�� */
	for (i = 0; i<image_count; i++)
	{
		point_counts.push_back(board_size.width*board_size.height);
	}

#if 0
	/* ��ʼ�궨 */
	calibrateCamera(object_points, image_points_seq, image_size, cameraMatrix, distCoeffs, rvecsMat, tvecsMat, CV_CALIB_FIX_ASPECT_RATIO | CV_CALIB_FIX_K4 | CV_CALIB_FIX_K5);
	cout << "�궨��ɣ�\n";
#else
	int flags = 0;
	flags |= cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC;
	flags |= cv::fisheye::CALIB_CHECK_COND;
	flags |= cv::fisheye::CALIB_FIX_SKEW;
	fisheye::calibrate(object_points, image_points_seq, image_size, cameraMatrix, distCoeffs, rvecsMat, tvecsMat, flags, cv::TermCriteria(3, 20, 1e-6));
	cout << "������ɣ�\n";
#endif

	//�Ա궨�����������
	cout << "��ʼ���۱궨���������������\n";
	double total_err = 0.0; /* ����ͼ���ƽ�������ܺ� */
	double err = 0.0; /* ÿ��ͼ���ƽ����� */
	vector<Point2f> image_points2;    /* �������¼���õ���ͶӰ�� */
	cout << "\tÿ��ͼ��ı궨��\n";
	fout << "ÿ��ͼ��ı궨��\n";
 
	for (i = 0; i<total; i++)
	{
		vector<Point3f>& tempPointSet = object_points[i];
		/* ͨ���õ������������������Կռ����ά���������ͶӰ���㣬�õ��µ�ͶӰ�� */
		projectPoints(tempPointSet, rvecsMat[i], tvecsMat[i], cameraMatrix, distCoeffs, image_points2);
 
		/* �����µ�ͶӰ��;ɵ�ͶӰ��֮������*/
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
		std::cout << "��" << i + 1 << "��ͼ���ƽ����" << err << "����" << endl;
		fout << "��" << i + 1 << "��ͼ���ƽ����" << err << "����" << endl;
	}
	std::cout << "����ƽ����" << total_err / image_count << "����" << endl;
	fout << "����ƽ����" << total_err / image_count << "����" << endl << endl;
	std::cout << "������ɣ�" << endl;
	point_counts.clear();

	//���涨����  
	Mat rotation_matrix = Mat(3, 3, CV_32FC1, Scalar::all(0)); /* ����ÿ��ͼ�����ת���� */
	std::cout << "��ʼ���涨����������������" << endl;
	fout << "����ڲ�������" << endl;
	fout << cameraMatrix << endl << endl;
	fout << "����ϵ����\n";
	fout << distCoeffs << endl << endl << endl;
	for (int i = 0; i<total; i++)
	{
		fout << "��" << i + 1 << "��ͼ�����ת������" << endl;
		fout << rvecsMat[i] << endl;
		/* ����ת����ת��Ϊ���Ӧ����ת���� */
		Rodrigues(rvecsMat[i], rotation_matrix);
		fout << "��" << i + 1 << "��ͼ�����ת����" << endl;
		fout << rotation_matrix << endl;
		fout << "��" << i + 1 << "��ͼ���ƽ��������" << endl;
		fout << tvecsMat[i] << endl << endl;
	}
	std::cout << "��ɱ���" << endl;
	fout << endl;

	imageZoom.release();

	/************************************************************************
	��ʾ������
	*************************************************************************/
	Mat mapx = Mat(image_size, CV_32FC1);
	Mat mapy = Mat(image_size, CV_32FC1);
	Mat R = Mat::eye(3, 3, CV_32F);
	cout << "�������ͼ��" << endl;
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
	std::cout << "�������" << endl;

	/************************************************************************
	����һ��ͼƬ
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
		cout << "�������" << endl;
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
* @param rotation ������ת����
* @return ���ؽǶ�
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

//���������ļ�
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