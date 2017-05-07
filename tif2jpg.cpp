#include <bits/stdc++.h>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
using namespace std;

int fixedSize = 257;
cv::Mat newImage(const cv::Mat src){
	cv::Mat res(3*src.rows,3*src.cols,CV_8UC3);
	cv::Mat t1,t2,t3;
	cv::flip(src,t1,1);
	// cv::imshow("new",t1);
	// cv::waitKey(0);
	cv::flip(src,t2,0);
	// cv::imshow("new",t2);
	// cv::waitKey(0);
	cv::flip(src,t3,-1);
	// cv::imshow("new",t3);
	// cv::waitKey(0);
	src.copyTo(res(cv::Rect(src.cols,src.rows,src.cols,src.rows)));
	t1.copyTo(res(cv::Rect(0,src.rows,src.cols,src.rows)));
	t1.copyTo(res(cv::Rect(2*src.cols,src.rows,src.cols,src.rows)));
	t2.copyTo(res(cv::Rect(src.cols,0,src.cols,src.rows)));
	t2.copyTo(res(cv::Rect(src.cols,2*src.rows,src.cols,src.rows)));
	t3.copyTo(res(cv::Rect(0,0,src.cols,src.rows)));
	t3.copyTo(res(cv::Rect(2*src.cols,0,src.cols,src.rows)));
	t3.copyTo(res(cv::Rect(0,2*src.rows,src.cols,src.rows)));
	t3.copyTo(res(cv::Rect(2*src.cols,2*src.rows,src.cols,src.rows)));
	// cv::imshow("res",res);
	// cv::waitKey(0);
	return res(cv::Rect(src.cols,src.rows,fixedSize,fixedSize));
}

int main(int argc, char** argv){
	ifstream infile("Filenames/f0.txt");
	string line;
	while (getline(infile, line))
	{
	    cv::Mat img= cv::imread(line,1);
	    /*cv::imshow("img",img);
	    cv::waitKey(0);*/
	    if(img.rows!=fixedSize || img.cols!=fixedSize){
	    	cout<<line<<endl;
	    	cout<<img.rows<<" "<<img.cols<<endl;
	    	img = newImage(img);
	    	cout<<img.rows<<" "<<img.cols<<endl;
	    }
	    string line2 = "patchDataset/" + line.substr(line.length()-11,line.length());
	    line2.replace(line2.end()-3,line2.end(),"jpg");
	    //cout<<line2<<endl;
	    cv::imwrite(line2,img);
	}

	return 0;
}