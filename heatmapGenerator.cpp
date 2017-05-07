#include <bits/stdc++.h>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
using namespace std;

cv::Mat filterOut(cv::Mat img,cv::Mat filter){
	cv::Mat res(img.rows,img.cols,CV_8UC3);
	cv::Mat filterResize(img.rows,img.cols,CV_32FC1);
	//cv::adaptiveThreshold(filter,filter,255,CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY,5,0);
	double min, max;
	cv::minMaxIdx(filter,&min,&max);
	// cout<<min<<" "<<max<<endl;
	// filter.convertTo(filter,CV_8U, 255/max,0);
	// cv::threshold(filter,filter,0.54,255,CV_THRESH_OTSU);
	cv::resize(filter,filterResize,cv::Size(img.cols,img.rows));
	//cv::imshow("filRes",filterResize);
	//cv::waitKey(0);
	filterResize.convertTo(filterResize, CV_32F, 1.0 / 255, 0);
	cv::threshold(filterResize,filterResize,0.97,1,CV_THRESH_TOZERO);
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
	ifstream file("Filenames/whole_slide.txt");
	string imName;
	while(file.good()){
		getline(file,imName);
		if(imName.empty())continue;
		cv::Mat img = cv::imread(imName,1);
		// cv::imshow("img",img);
		string imName2 = imName;
		string xycName = imName;
		xycName.replace(xycName.begin()+25,xycName.begin()+28,"xyc");
		xycName.replace(xycName.end()-3,xycName.end(),"xyc");
		cout<<xycName<<endl;
		imName2.replace(imName2.begin(),imName2.begin()+29,"Deploy_Results/filter");
		// cout<<imName2<<endl;
		cv::Mat filter = cv::imread(imName2,0);
		cv::equalizeHist(filter,filter);
		cv::Mat res = filterOut(img,filter);
		// cv::imshow("filter",filter);
		// cv::waitKey(0);
		// cv::imshow("res",res);
		// cv::waitKey(0);
		string imName3 = imName2;
		imName3.replace(imName3.begin()+15,imName3.begin()+21,"mask");
		cout<<imName3<<endl;
		cv::imwrite(imName3,res);
	}
	return 0;
}