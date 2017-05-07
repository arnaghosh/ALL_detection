#include <bits/stdc++.h>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
using namespace std;

cv::Mat filterOut(cv::Mat img,cv::Mat filter){
	cv::Mat res(img.rows,img.cols,CV_8UC3);
	cv::Mat filterResize(img.rows,img.cols,CV_32FC1);
	cv::resize(filter,filterResize,cv::Size(img.cols,img.rows));
	//cv::imshow("filRes",filterResize);
	//cv::waitKey(0);
	filterResize.convertTo(filterResize, CV_32F, 1.0 / 255, 0);
	cv::threshold(filterResize,filterResize,0.58,1,CV_THRESH_TOZERO);
	//cv::imshow("filRes",filterResize);
	//cv::waitKey(0);
	for(int i=0;i<img.rows;i++){
		for(int j=0;j<img.cols;j++){
			res.at<cv::Vec3b>(i,j)[0] = img.at<cv::Vec3b>(i,j)[0] * filterResize.at<float>(i,j); 
			res.at<cv::Vec3b>(i,j)[1] = img.at<cv::Vec3b>(i,j)[1] * filterResize.at<float>(i,j); 
			res.at<cv::Vec3b>(i,j)[2] = img.at<cv::Vec3b>(i,j)[2] * filterResize.at<float>(i,j); 
		}
	}
	return res;
}

int main(){
	cv::Mat img = cv::imread("ALL_IDB dataset/ALL_IDB1/img/Im001_1.jpg",1);
	cv::imshow("img",img);
	cv::Mat filter = cv::imread("test.jpg",0);
	cv::Mat res = filterOut(img,filter);
	cv::imshow("filter",filter);
	cv::waitKey(0);
	cv::imshow("res",res);
	cv::waitKey(0);
	cv::imwrite("testRes1_1.jpg",res);
	return 0;
}