/****************************************************************/
/*******                   程序说明                       *******/
/*******     程序描述：重叠果实分割				          *******/
/*******     V S版 本：VS2012                             *******/
/*******     时    间：2016.8.1							  *******/
/****************************************************************/

//头文件
#include "iostream"  
#include "cv.h"  
#include "Math.h"
#include "vector" 
#include "time.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

const float pi = 3.14159;

//苹果位置结构体
struct circleXYR  
{
	Point2i xy;		//圆心坐标
	float R;		//半径
};

/****************************************************************/
/*******                最小二乘圆拟合函数                  *******/
/****************************************************************/
circleXYR LeastSquaresFitting(vector<Point2f> contour)
{
//	if (contour.size() < 3)
//		return;
	int i = 0;

	float x1sum = 0;
    double y1sum = 0;
    double x2sum = 0;
    double y2sum = 0;
    double x3sum = 0;
    double y3sum =0;
    double x1y1sum = 0;
    double x1y2sum = 0;
    double x2y1sum = 0;

	for (i = 0;i < contour.size();i++)
    {
        x1sum = x1sum + contour[i].x;
		y1sum = y1sum + contour[i].y;
		x2sum = x2sum + pow(contour[i].x,2);
		y2sum = y2sum + pow(contour[i].y,2);
		x3sum = x3sum + pow(contour[i].x,3);
		y3sum = y3sum + pow(contour[i].y,3);
		x1y1sum = x1y1sum + contour[i].x*contour[i].y;
		x1y2sum = x1y2sum + contour[i].x*pow(contour[i].y,2);;
		x2y1sum = x2y1sum + pow(contour[i].x,2)*contour[i].y;
    }

	float C,D,E,G,H,N;
	float a,b,c;
	N = contour.size();
	C = N*x2sum - pow(x1sum,2);
	D = N*x1y1sum - x1sum*y1sum;
	E = N*x3sum + N*x1y2sum - (x2sum + y2sum)*x1sum;
	G = N*y2sum - pow(y1sum,2);
	H = N*x2y1sum + N*y3sum - (x2sum + y2sum)*y1sum;
    a = (H*D-E*G)/(C*G-D*D);
    b = (H*C-E*D)/(D*D-G*C);
	c = -(a*x1sum + b*y1sum + x2sum + y2sum)/N;

	float A,B,R;
    A = a/(-2);
    B = b/(-2);
    R = sqrt(a*a+b*b-4*c)/2;

	circleXYR xyR;
	Point2f xy;

	xyR.xy.x = int(A);
	xyR.xy.y = int(B);
	xyR.R = R;


	return xyR;
}

/****************************************************************/
/*******                最大类间方差函数                  *******/
/****************************************************************/
int Otsu(Mat &src)
{
	float histogram[256] = {0};
	for (int i = 0; i < src.rows ; i++)
	{
		uchar *data = src.ptr<uchar>(i);
		for (int j = 0; j < src.cols; j++)
		{
			histogram[data[j]]++;
		}
	}
	int s = src.rows*src.cols; 

	float p[256]={0};
	for (int i = 0; i <= 255; i++)
		p[i] = histogram[i] / s;

	int min = 0;
	int max = 255;

	while (histogram[min] == 0)
	{
		min++;
	}
	while (histogram[max] == 0)
	{
		max--;
	}

//求平均灰度
	float u=0;
	for (int i = min; i <= max; i++)
		u = u + i * p[i];				//整幅图像的平均灰度  

	cout<<"平均灰度值："<<u<<endl;

//求最大类间方差
	int T;      
    float maxVariance = 0;    
    float w0 = 0, u0 = 0; 
	for (int i = min; i <= max; i++)
	{
		w0 = w0 + p[i];					//假设当前灰度i为阈值, 0~i 灰度的像素(假设像素值在此范围的像素叫做前景像素) 所占整幅图像的比例  
		u0 = u0 + i * p[i];				// 灰度i 之前的像素(0~i)的平均灰度值： 前景像素的平均灰度值

		float t = u*w0-u0;
		float variance = t*t/(w0*(1-w0));
		if(variance > maxVariance)   
        {			
			maxVariance = variance;    
            T = i;    
        }    
	}
	return T;
}

int main()
{
//读取图像并显示原图
	clock_t start, finish;
	start = clock();
	Mat img = imread("1.jpg");
	if (!img.data)
	{
		cout<<"读取图像失败"<<endl;
		return 0;
	}

//颜色通道分离
	vector<Mat> channels;
	Mat B,R,G;
	split(img,channels);
	B = channels.at(0);
	G = channels.at(1);
	R = channels.at(2);

//获得YUV空间中的V分量，V= 0.615*R-0.515*G-0.100*B 
	Mat grayImage;
	addWeighted(R,0.615,G,-0.515,0,grayImage);
	addWeighted(grayImage,1,B,-0.10,0,grayImage);

//图像预处理
	GaussianBlur(grayImage,grayImage,Size(7,7),0);	//高斯滤波

//用最大类间方差法对图像进行二值化
	Mat binaryImage;
	int T = Otsu(grayImage);
//	cout<<"图像分割阈值"<<T<<endl;
	threshold(grayImage,binaryImage,T,255,THRESH_BINARY);
	imshow("二值化图像",binaryImage);

//获取面积大于指定值的外部轮廓
	vector<vector<Point>> contours;			//存放所有轮廓
	vector<vector<Point>> area_contours;
	findContours(binaryImage,contours, CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
	int maxArea = 0;
	int maxNumber=0;
	for (int i=0;i<contours.size();i++)                                  //获取包含点数最多的轮廓索引
		if (contourArea(contours[i]) > img.cols*img.rows/80)
			area_contours.push_back(contours[i]);

//填充边界
	Mat fill_binaryImage(img.size(),CV_8U,Scalar(0));
	for (int i = 0; i < area_contours.size(); i++)
		drawContours(fill_binaryImage, area_contours, i, Scalar(255), CV_FILLED);
//	imshow("填充效果图",fill_binaryImage);

//创建彩色示意图像
	Mat imshowImgae1;
	Mat imshowImage2;
	Mat imshowImage3;
	Mat imshowImage4;
	Mat imshowImage5;
	Mat imshowImage6;
	Mat imshowImage7;
	Mat imshowImage8;
	Mat imshowImage9;
	channels.at(0) = fill_binaryImage;
	channels.at(1) = fill_binaryImage;
	channels.at(2) = fill_binaryImage;
	merge(channels,imshowImgae1);
	merge(channels,imshowImage2);
	merge(channels,imshowImage3);
	merge(channels,imshowImage4);
	merge(channels,imshowImage5);
	merge(channels,imshowImage6);
	merge(channels,imshowImage7);
	merge(channels,imshowImage8);
	merge(channels,imshowImage9);


	vector<Point> maxAreaContour;	//存放当前边界
//模糊线段生长参数
	float a,b;				//y=b*x+a
	int N,M;
	float dmax;
	float dd;
	int k;
	vector<int> sp;

//向量p1,p2旋转角参数
	Point2i p1,p2;
	float norm_p1,norm_p2;			//向量p1,p2的模
	int dot_p1p2;					//向量p1,p2的点乘
	int cross_p1p2;					//向量p1,p2的叉乘
	float temp_angle;				//夹角
	float angle;					//旋转角
	vector<Point2f> s_p_angle;		//存放所有的顶点和该顶点的旋转角
	Point2f temp_p_angle;			//临时存放一个顶点和该顶点的旋转角

//求凸点与异常凸点参数
	float s;
	vector<int> s_pitpoint;
	int pitpoint;

//分割边界参数
	vector<vector<Point2f>>  sum_segContour;				//由凹点及异常凸点分割的全部边界
	vector<Point2f> segContour;								//由凹点及异常凸点分割的安格边界

//筛选有效边界参数
	float maxLengthContour;									//存放最长边界
	float dsum;												//边界上的点到弦的距离和
	float avd;												//边界上的点到弦的距离的均值
	vector<vector<Point2f>> sum_Length_segContour;			//存放去除过短边界后的边界
	vector<vector<Point2f>> sum_curve_segContour;			//去除平直边界的边界存放容器

//最小二乘拟合回归参数
	float ce;												//合并系数
	float maxR;												//最大回归圆半径
	float min_circle_distance;								//最小圆心距
	float min_circle_distance_number;						//最小圆心距对应的边界编号
	float temp_circle_distance;								//临时存放最小圆心距
	int maxR_number;										//最大回归圆对应的边界编号
	circleXYR xyR;											//回归圆参数结构体
	circleXYR max_xyR;										//当前最大回归圆参数结构体
	circleXYR min_xyR;										//与最大回归圆圆心距最小的回归圆结构体
	vector<vector<Point2f>>	temp_contour;					//临时边界
	vector<Point2f> merge_contour;							//合并边界
	vector<Point2f> temp_merge_contour;						//临时合并边界
	vector<circleXYR> temp_sum_xyR;							//临时所有边界拟合结果
	vector<vector<Point2f>> sum_contour;					//最后所有合并结果

	for (int t = 0; t < area_contours.size(); t++)
	{
		maxAreaContour = area_contours[t];

	//模糊线段生长
		k = 0;
		while(k < maxAreaContour.size())
		{
			N=1;
			M=1;
			while (1)
			{
				//判断下一个点是否为首点
				if (k+N == maxAreaContour.size())
					M = -k;

				//计算当前模糊线段y=bx+a
				if (maxAreaContour[k+M].x - maxAreaContour[k].x == 0)
				{
					a = maxAreaContour[k].x;
				}
				else
				{
					b = (maxAreaContour[k+M].y - maxAreaContour[k].y)/(maxAreaContour[k+M].x-maxAreaContour[k].x);
					a = maxAreaContour[k].y - b*maxAreaContour[k].x;
				}

				//计算当前模糊生长的点到模糊线段的最大距离dmax
				dmax = 0;
				for (int i = k+1; i < k+N; i++)
				{
					if (maxAreaContour[k+M].x - maxAreaContour[k].x == 0)
						dd = abs(maxAreaContour[i].x - a);
					else
						dd = abs(b*maxAreaContour[i].x-maxAreaContour[i].y+a)/sqrt(1+pow(b,2));

					if (dd > dmax )
						dmax =abs(b*maxAreaContour[i].x-maxAreaContour[i].y+a)/sqrt(1+pow(b,2)); 
				}

				if (k+N == maxAreaContour.size())
					break;
				if (k+N >= maxAreaContour.size()-1)
					M=-k;
				else
					M=M+1;


				if ( dmax >= 3|| N > 20)				
					break;		
				N=N+1;
			}
			sp.push_back(k);
			k=k+N;
		}

	//绘制所有顶点
		for (int i = 0; i < sp.size(); i++)
			circle(imshowImgae1,maxAreaContour[sp[i]],1,Scalar(255,255,0),2);

	//求向量p1,p2的旋转角
		for (int i = 0; i < sp.size(); i++)
		{

			//判断该顶点是不是起始点，是，上一个点为终点
			if (i == 0)
				N = sp.size()-1;
			else
				N = i-1;

			//判断该顶点是不是终点，是，下一个点位起始点
			if (i == sp.size()-1)
				M = 0;
			else
				M=i+1;

			//求向量p1,p2
			p1.x = maxAreaContour[sp[N]].x - maxAreaContour[sp[i]].x;
			p1.y = maxAreaContour[sp[N]].y - maxAreaContour[sp[i]].y;
			p2.x = maxAreaContour[sp[M]].x - maxAreaContour[sp[i]].x;
			p2.y = maxAreaContour[sp[M]].y - maxAreaContour[sp[i]].y;

			//求p1,p2的夹角
			norm_p1 = sqrt(pow(p1.x,2)+pow(p1.y,2));				//向量p1的模
			norm_p2 = sqrt(pow(p2.x,2)+pow(p2.y,2));				//向量p2的模
			dot_p1p2 = p1.x*p2.x + p1.y*p2.y;						//向量p1,p2的点乘
			temp_angle = acos(dot_p1p2*1.0/(norm_p1*norm_p2));		//向量p1,p2的夹角

			//求向量p1,p2的叉乘
			cross_p1p2 = p1.x*p2.y - p2.x*p1.y;

			//求旋转角
			if (cross_p1p2 < 0)
				angle = 2*pi - temp_angle;
			else
				angle = temp_angle;

			temp_p_angle.x = sp[i];
			temp_p_angle.y = angle;
			s_p_angle.push_back(temp_p_angle);
		}

	//求异常凸点与凹点
		s=0;
		k = 0;
		for (int i = 0; i < s_p_angle.size(); i++)
			if (s_p_angle[i].y < pi)
			{
				s += s_p_angle[i].y ;
				k++;
			}
		for (int i = 0; i < s_p_angle.size(); i++)
		{
			if (s_p_angle[i].y > pi||s_p_angle[i].y < s/k)
			{
				pitpoint = s_p_angle[i].x;
				s_pitpoint.push_back(pitpoint);
			}
		}

	//绘制凹点与异常凸点
		for (int i = 0; i < s_pitpoint.size(); i++)
			circle(imshowImage2,maxAreaContour[s_pitpoint[i]],1,Scalar(255,255,0),2);


	//存放分割后的边界
		//除最后一条边界的所有边界
		for (int i = 0; i < s_pitpoint.size()-1; i++)
		{
			for (int j = s_pitpoint[i]; j < s_pitpoint[i+1]; j++)
				segContour.push_back(maxAreaContour[j]);
			sum_segContour.push_back(segContour);
			segContour.clear();
		}
		//最后一条边界
		for (int j = s_pitpoint[s_pitpoint.size()-1]; j < maxAreaContour.size(); j++)
			segContour.push_back(maxAreaContour[j]);
		for (int j = 0; j < s_pitpoint[0]; j++)
			segContour.push_back(maxAreaContour[j]);
		sum_segContour.push_back(segContour);
		segContour.clear();

	//求最长边界maxLengthContour
		maxLengthContour=0;
		for (int i = 0; i < sum_segContour.size(); i++)
			if (arcLength(sum_segContour[i],0) > maxLengthContour)
				maxLengthContour = arcLength(sum_segContour[i],0);

	//去除长度小于0.1最长边界的边界
		for (int i = 0; i < sum_segContour.size(); i++)
			if (arcLength(sum_segContour[i],0) > 0.2*maxLengthContour)
				sum_Length_segContour.push_back(sum_segContour[i]);

	//绘制去除过短边界后的边界图
		for (int i = 0; i < sum_Length_segContour.size(); i++)
			for (int j = 0; j < sum_Length_segContour[i].size(); j++)
				circle(imshowImage3,sum_Length_segContour[i][j],1,Scalar(255,255,0),2);

	//去除平直边界
		for (int i = 0; i < sum_Length_segContour.size(); i++)
		{
			//求弦的直线方程y=b*x+a
			if (sum_Length_segContour[i][sum_Length_segContour[i].size()-1].x - sum_Length_segContour[i][0].x == 0)
				a = sum_Length_segContour[i][0].x;
			else
			{
				b = (sum_Length_segContour[i][sum_Length_segContour[i].size()-1].y - sum_Length_segContour[i][0].y)/(sum_Length_segContour[i][sum_Length_segContour[i].size()-1].x - sum_Length_segContour[i][0].x);
				a = sum_Length_segContour[i][0].y - b*sum_Length_segContour[i][0].x;
			}

			//边界上的点到弦的距离和
			dsum = 0;
			for (int j = 0; j < sum_Length_segContour[i].size(); j++)
			{
				if (sum_Length_segContour[i][sum_Length_segContour[i].size()-1].x - sum_Length_segContour[i][0].x == 0)
					dsum += abs(sum_Length_segContour[i][j].x - a);
				else
					dsum += abs((b*sum_Length_segContour[i][j].x - sum_Length_segContour[i][j].y + a)/(sqrt(1 + pow(b,2))));
			}
			avd = dsum/sum_Length_segContour[i].size();
			if (avd >= 1.6)
				sum_curve_segContour.push_back(sum_Length_segContour[i]);
		}

	//绘制有效边界图
		for (int i = 0; i < sum_curve_segContour.size(); i++)
			for (int j = 0; j < sum_curve_segContour[i].size(); j++)
				circle(imshowImage4,sum_curve_segContour[i][j],1,Scalar(255,255,0),2);

	//对所有有效边界圆拟合
		for (int i = 0; i < sum_curve_segContour.size(); i++)
		{
			xyR = LeastSquaresFitting(sum_curve_segContour[i]);
			circle(imshowImage5,xyR.xy,xyR.R,Scalar(255,255,0),2);
		}

	//拟合圆合并
		while (1)
		{
			while (1)
			{
				temp_sum_xyR.clear();
				//对所有边界进行圆拟合
				for (int i = 0; i < sum_curve_segContour.size(); i++)
				{
					xyR = LeastSquaresFitting(sum_curve_segContour[i]);
					temp_sum_xyR.push_back(xyR);
				}

				//求最大拟合圆
				maxR = 0;
				for (int i = 0; i < temp_sum_xyR.size(); i++)
				{
					if (temp_sum_xyR[i].R > maxR)
					{
						maxR = temp_sum_xyR[i].R;
						maxR_number = i;
					}
				}
				max_xyR = temp_sum_xyR[maxR_number];
				merge_contour = sum_curve_segContour[maxR_number];

				//找出与最大拟合圆圆心距最小的拟合圆
				min_circle_distance = 100000;

				if (temp_sum_xyR.size() == 1)
					break;

				for (int i = 0; i < temp_sum_xyR.size(); i++)
				{
					if (i == maxR_number)
						continue;
					temp_circle_distance = sqrt(pow(temp_sum_xyR[i].xy.x-max_xyR.xy.x,2) + pow(temp_sum_xyR[i].xy.y-max_xyR.xy.y,2));
					if (temp_circle_distance < min_circle_distance)
					{
						min_circle_distance = temp_circle_distance;
						min_circle_distance_number = i;
					}
				}
				min_xyR = temp_sum_xyR[min_circle_distance_number];

				if (min_circle_distance < 1.1*(max_xyR.R - min_xyR.R))
					ce = 1;
				else if (max_xyR.R/min_xyR.R < 1.6)
					ce = 0.6*max_xyR.R/min_xyR.R;
				else if (max_xyR.R/min_xyR.R >= 1.6&&max_xyR.R/min_xyR.R < 2.6)
					ce = 0.96;
				else if (max_xyR.R/min_xyR.R >= 2.6&&max_xyR.R/min_xyR.R < 3.2)
					ce = -0.6*max_xyR.R/min_xyR.R + 2.52;
				else
					ce =0.6;

				//判断最小圆心距是否小于最大圆的半径,是，本次合并结束，否合并两条边界，继续循环。
				if (min_circle_distance > ce*max_xyR.R)
					break;

				//将两边界合并
				temp_merge_contour = merge_contour;
				for (int j = 0; j < sum_curve_segContour[min_circle_distance_number].size(); j++)
					temp_merge_contour.push_back(sum_curve_segContour[min_circle_distance_number][j]);
			
				//合并所有边界
				temp_contour.clear();
				for (int i = 0; i < sum_curve_segContour.size(); i++)
				{
					if (i == maxR_number||i == min_circle_distance_number)
						continue;
					temp_contour.push_back(sum_curve_segContour[i]);
				}
				temp_contour.push_back(temp_merge_contour);
				sum_curve_segContour = temp_contour;
			}

			//去除合并边界
			sum_contour.push_back(merge_contour);
			temp_contour.clear();
			for (int i = 0; i < sum_curve_segContour.size(); i++)
			{
				if (i == maxR_number)
					continue;
				temp_contour.push_back(sum_curve_segContour[i]);
			}

			sum_curve_segContour = temp_contour;
			if (sum_curve_segContour.empty())
				break;
		}

		//去除未合并的过小圆，并找出最大圆
		for (int i = 0; i < sum_contour.size(); i++)
		{
			xyR = LeastSquaresFitting(sum_contour[i]);
			if (xyR.R >= sqrt(img.rows*img.cols/(60*pi)))
			{
				circle(img,xyR.xy,xyR.R,Scalar(255,255,0),2);
				circle(imshowImage6,xyR.xy,xyR.R,Scalar(255,255,0),2);
			}
		}


		sp.clear();
		s_p_angle.clear();
		s_pitpoint.clear();
		segContour.clear();
		sum_Length_segContour.clear();
		sum_curve_segContour.clear();
		temp_contour.clear();
		merge_contour.clear();
		temp_merge_contour.clear();
		temp_sum_xyR.clear();
		sum_contour.clear();

	}
	finish = clock();
	cout << "时间：" << (float)(finish - start) / CLOCKS_PER_SEC << endl;
//	imshow("顶点图",imshowImgae1);
//	imshow("凹点及异常凸点图",imshowImage2);
//	imshow("去除过短边界后有效边界图",imshowImage3);
//	imshow("有效边界图",imshowImage4);
//	imshow("圆回归",imshowImage5);
//	imshow("回归圆合并图",imshowImage6);
	imshow("效果图",img);
	waitKey(0);
	cin.get();
	return 1;
}
