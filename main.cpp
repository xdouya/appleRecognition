/****************************************************************/
/*******                   ����˵��                       *******/
/*******     �����������ص���ʵ�ָ�				          *******/
/*******     V S�� ����VS2012                             *******/
/*******     ʱ    �䣺2016.8.1							  *******/
/****************************************************************/

//ͷ�ļ�
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

//ƻ��λ�ýṹ��
struct circleXYR  
{
	Point2i xy;		//Բ������
	float R;		//�뾶
};

/****************************************************************/
/*******                ��С����Բ��Ϻ���                  *******/
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
/*******                �����䷽���                  *******/
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

//��ƽ���Ҷ�
	float u=0;
	for (int i = min; i <= max; i++)
		u = u + i * p[i];				//����ͼ���ƽ���Ҷ�  

	cout<<"ƽ���Ҷ�ֵ��"<<u<<endl;

//�������䷽��
	int T;      
    float maxVariance = 0;    
    float w0 = 0, u0 = 0; 
	for (int i = min; i <= max; i++)
	{
		w0 = w0 + p[i];					//���赱ǰ�Ҷ�iΪ��ֵ, 0~i �Ҷȵ�����(��������ֵ�ڴ˷�Χ�����ؽ���ǰ������) ��ռ����ͼ��ı���  
		u0 = u0 + i * p[i];				// �Ҷ�i ֮ǰ������(0~i)��ƽ���Ҷ�ֵ�� ǰ�����ص�ƽ���Ҷ�ֵ

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
//��ȡͼ����ʾԭͼ
	clock_t start, finish;
	start = clock();
	Mat img = imread("1.jpg");
	if (!img.data)
	{
		cout<<"��ȡͼ��ʧ��"<<endl;
		return 0;
	}

//��ɫͨ������
	vector<Mat> channels;
	Mat B,R,G;
	split(img,channels);
	B = channels.at(0);
	G = channels.at(1);
	R = channels.at(2);

//���YUV�ռ��е�V������V= 0.615*R-0.515*G-0.100*B 
	Mat grayImage;
	addWeighted(R,0.615,G,-0.515,0,grayImage);
	addWeighted(grayImage,1,B,-0.10,0,grayImage);

//ͼ��Ԥ����
	GaussianBlur(grayImage,grayImage,Size(7,7),0);	//��˹�˲�

//�������䷽���ͼ����ж�ֵ��
	Mat binaryImage;
	int T = Otsu(grayImage);
//	cout<<"ͼ��ָ���ֵ"<<T<<endl;
	threshold(grayImage,binaryImage,T,255,THRESH_BINARY);
	imshow("��ֵ��ͼ��",binaryImage);

//��ȡ�������ָ��ֵ���ⲿ����
	vector<vector<Point>> contours;			//�����������
	vector<vector<Point>> area_contours;
	findContours(binaryImage,contours, CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
	int maxArea = 0;
	int maxNumber=0;
	for (int i=0;i<contours.size();i++)                                  //��ȡ��������������������
		if (contourArea(contours[i]) > img.cols*img.rows/80)
			area_contours.push_back(contours[i]);

//���߽�
	Mat fill_binaryImage(img.size(),CV_8U,Scalar(0));
	for (int i = 0; i < area_contours.size(); i++)
		drawContours(fill_binaryImage, area_contours, i, Scalar(255), CV_FILLED);
//	imshow("���Ч��ͼ",fill_binaryImage);

//������ɫʾ��ͼ��
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


	vector<Point> maxAreaContour;	//��ŵ�ǰ�߽�
//ģ���߶���������
	float a,b;				//y=b*x+a
	int N,M;
	float dmax;
	float dd;
	int k;
	vector<int> sp;

//����p1,p2��ת�ǲ���
	Point2i p1,p2;
	float norm_p1,norm_p2;			//����p1,p2��ģ
	int dot_p1p2;					//����p1,p2�ĵ��
	int cross_p1p2;					//����p1,p2�Ĳ��
	float temp_angle;				//�н�
	float angle;					//��ת��
	vector<Point2f> s_p_angle;		//������еĶ���͸ö������ת��
	Point2f temp_p_angle;			//��ʱ���һ������͸ö������ת��

//��͹�����쳣͹�����
	float s;
	vector<int> s_pitpoint;
	int pitpoint;

//�ָ�߽����
	vector<vector<Point2f>>  sum_segContour;				//�ɰ��㼰�쳣͹��ָ��ȫ���߽�
	vector<Point2f> segContour;								//�ɰ��㼰�쳣͹��ָ�İ���߽�

//ɸѡ��Ч�߽����
	float maxLengthContour;									//�����߽�
	float dsum;												//�߽��ϵĵ㵽�ҵľ����
	float avd;												//�߽��ϵĵ㵽�ҵľ���ľ�ֵ
	vector<vector<Point2f>> sum_Length_segContour;			//���ȥ�����̱߽��ı߽�
	vector<vector<Point2f>> sum_curve_segContour;			//ȥ��ƽֱ�߽�ı߽�������

//��С������ϻع����
	float ce;												//�ϲ�ϵ��
	float maxR;												//���ع�Բ�뾶
	float min_circle_distance;								//��СԲ�ľ�
	float min_circle_distance_number;						//��СԲ�ľ��Ӧ�ı߽���
	float temp_circle_distance;								//��ʱ�����СԲ�ľ�
	int maxR_number;										//���ع�Բ��Ӧ�ı߽���
	circleXYR xyR;											//�ع�Բ�����ṹ��
	circleXYR max_xyR;										//��ǰ���ع�Բ�����ṹ��
	circleXYR min_xyR;										//�����ع�ԲԲ�ľ���С�Ļع�Բ�ṹ��
	vector<vector<Point2f>>	temp_contour;					//��ʱ�߽�
	vector<Point2f> merge_contour;							//�ϲ��߽�
	vector<Point2f> temp_merge_contour;						//��ʱ�ϲ��߽�
	vector<circleXYR> temp_sum_xyR;							//��ʱ���б߽���Ͻ��
	vector<vector<Point2f>> sum_contour;					//������кϲ����

	for (int t = 0; t < area_contours.size(); t++)
	{
		maxAreaContour = area_contours[t];

	//ģ���߶�����
		k = 0;
		while(k < maxAreaContour.size())
		{
			N=1;
			M=1;
			while (1)
			{
				//�ж���һ�����Ƿ�Ϊ�׵�
				if (k+N == maxAreaContour.size())
					M = -k;

				//���㵱ǰģ���߶�y=bx+a
				if (maxAreaContour[k+M].x - maxAreaContour[k].x == 0)
				{
					a = maxAreaContour[k].x;
				}
				else
				{
					b = (maxAreaContour[k+M].y - maxAreaContour[k].y)/(maxAreaContour[k+M].x-maxAreaContour[k].x);
					a = maxAreaContour[k].y - b*maxAreaContour[k].x;
				}

				//���㵱ǰģ�������ĵ㵽ģ���߶ε�������dmax
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

	//�������ж���
		for (int i = 0; i < sp.size(); i++)
			circle(imshowImgae1,maxAreaContour[sp[i]],1,Scalar(255,255,0),2);

	//������p1,p2����ת��
		for (int i = 0; i < sp.size(); i++)
		{

			//�жϸö����ǲ�����ʼ�㣬�ǣ���һ����Ϊ�յ�
			if (i == 0)
				N = sp.size()-1;
			else
				N = i-1;

			//�жϸö����ǲ����յ㣬�ǣ���һ����λ��ʼ��
			if (i == sp.size()-1)
				M = 0;
			else
				M=i+1;

			//������p1,p2
			p1.x = maxAreaContour[sp[N]].x - maxAreaContour[sp[i]].x;
			p1.y = maxAreaContour[sp[N]].y - maxAreaContour[sp[i]].y;
			p2.x = maxAreaContour[sp[M]].x - maxAreaContour[sp[i]].x;
			p2.y = maxAreaContour[sp[M]].y - maxAreaContour[sp[i]].y;

			//��p1,p2�ļн�
			norm_p1 = sqrt(pow(p1.x,2)+pow(p1.y,2));				//����p1��ģ
			norm_p2 = sqrt(pow(p2.x,2)+pow(p2.y,2));				//����p2��ģ
			dot_p1p2 = p1.x*p2.x + p1.y*p2.y;						//����p1,p2�ĵ��
			temp_angle = acos(dot_p1p2*1.0/(norm_p1*norm_p2));		//����p1,p2�ļн�

			//������p1,p2�Ĳ��
			cross_p1p2 = p1.x*p2.y - p2.x*p1.y;

			//����ת��
			if (cross_p1p2 < 0)
				angle = 2*pi - temp_angle;
			else
				angle = temp_angle;

			temp_p_angle.x = sp[i];
			temp_p_angle.y = angle;
			s_p_angle.push_back(temp_p_angle);
		}

	//���쳣͹���밼��
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

	//���ư������쳣͹��
		for (int i = 0; i < s_pitpoint.size(); i++)
			circle(imshowImage2,maxAreaContour[s_pitpoint[i]],1,Scalar(255,255,0),2);


	//��ŷָ��ı߽�
		//�����һ���߽�����б߽�
		for (int i = 0; i < s_pitpoint.size()-1; i++)
		{
			for (int j = s_pitpoint[i]; j < s_pitpoint[i+1]; j++)
				segContour.push_back(maxAreaContour[j]);
			sum_segContour.push_back(segContour);
			segContour.clear();
		}
		//���һ���߽�
		for (int j = s_pitpoint[s_pitpoint.size()-1]; j < maxAreaContour.size(); j++)
			segContour.push_back(maxAreaContour[j]);
		for (int j = 0; j < s_pitpoint[0]; j++)
			segContour.push_back(maxAreaContour[j]);
		sum_segContour.push_back(segContour);
		segContour.clear();

	//����߽�maxLengthContour
		maxLengthContour=0;
		for (int i = 0; i < sum_segContour.size(); i++)
			if (arcLength(sum_segContour[i],0) > maxLengthContour)
				maxLengthContour = arcLength(sum_segContour[i],0);

	//ȥ������С��0.1��߽�ı߽�
		for (int i = 0; i < sum_segContour.size(); i++)
			if (arcLength(sum_segContour[i],0) > 0.2*maxLengthContour)
				sum_Length_segContour.push_back(sum_segContour[i]);

	//����ȥ�����̱߽��ı߽�ͼ
		for (int i = 0; i < sum_Length_segContour.size(); i++)
			for (int j = 0; j < sum_Length_segContour[i].size(); j++)
				circle(imshowImage3,sum_Length_segContour[i][j],1,Scalar(255,255,0),2);

	//ȥ��ƽֱ�߽�
		for (int i = 0; i < sum_Length_segContour.size(); i++)
		{
			//���ҵ�ֱ�߷���y=b*x+a
			if (sum_Length_segContour[i][sum_Length_segContour[i].size()-1].x - sum_Length_segContour[i][0].x == 0)
				a = sum_Length_segContour[i][0].x;
			else
			{
				b = (sum_Length_segContour[i][sum_Length_segContour[i].size()-1].y - sum_Length_segContour[i][0].y)/(sum_Length_segContour[i][sum_Length_segContour[i].size()-1].x - sum_Length_segContour[i][0].x);
				a = sum_Length_segContour[i][0].y - b*sum_Length_segContour[i][0].x;
			}

			//�߽��ϵĵ㵽�ҵľ����
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

	//������Ч�߽�ͼ
		for (int i = 0; i < sum_curve_segContour.size(); i++)
			for (int j = 0; j < sum_curve_segContour[i].size(); j++)
				circle(imshowImage4,sum_curve_segContour[i][j],1,Scalar(255,255,0),2);

	//��������Ч�߽�Բ���
		for (int i = 0; i < sum_curve_segContour.size(); i++)
		{
			xyR = LeastSquaresFitting(sum_curve_segContour[i]);
			circle(imshowImage5,xyR.xy,xyR.R,Scalar(255,255,0),2);
		}

	//���Բ�ϲ�
		while (1)
		{
			while (1)
			{
				temp_sum_xyR.clear();
				//�����б߽����Բ���
				for (int i = 0; i < sum_curve_segContour.size(); i++)
				{
					xyR = LeastSquaresFitting(sum_curve_segContour[i]);
					temp_sum_xyR.push_back(xyR);
				}

				//��������Բ
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

				//�ҳ���������ԲԲ�ľ���С�����Բ
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

				//�ж���СԲ�ľ��Ƿ�С�����Բ�İ뾶,�ǣ����κϲ���������ϲ������߽磬����ѭ����
				if (min_circle_distance > ce*max_xyR.R)
					break;

				//�����߽�ϲ�
				temp_merge_contour = merge_contour;
				for (int j = 0; j < sum_curve_segContour[min_circle_distance_number].size(); j++)
					temp_merge_contour.push_back(sum_curve_segContour[min_circle_distance_number][j]);
			
				//�ϲ����б߽�
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

			//ȥ���ϲ��߽�
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

		//ȥ��δ�ϲ��Ĺ�СԲ�����ҳ����Բ
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
	cout << "ʱ�䣺" << (float)(finish - start) / CLOCKS_PER_SEC << endl;
//	imshow("����ͼ",imshowImgae1);
//	imshow("���㼰�쳣͹��ͼ",imshowImage2);
//	imshow("ȥ�����̱߽����Ч�߽�ͼ",imshowImage3);
//	imshow("��Ч�߽�ͼ",imshowImage4);
//	imshow("Բ�ع�",imshowImage5);
//	imshow("�ع�Բ�ϲ�ͼ",imshowImage6);
	imshow("Ч��ͼ",img);
	waitKey(0);
	cin.get();
	return 1;
}
