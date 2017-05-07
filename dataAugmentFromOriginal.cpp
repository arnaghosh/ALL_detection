#include <bits/stdc++.h>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
using namespace std;

cv::Mat rotateImage(const cv::Mat source, double angle,int border=20)
{
    cv::Mat bordered_source;
    int top,bottom,left,right;
    top=bottom=left=right=border;
    cv::copyMakeBorder( source, bordered_source, top, bottom, left, right, cv::BORDER_CONSTANT,cv::Scalar() );
    cv::Point2f src_center(bordered_source.cols/2.0F, bordered_source.rows/2.0F);
    cv::Mat rot_mat = getRotationMatrix2D(src_center, angle, 1.0);
    cv::Mat dst;
    cv::warpAffine(bordered_source, dst, rot_mat, bordered_source.size());  
    return dst;
}

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
	return res;
}

int main(){
	for(int f=0;f<2;f++){
		cout<<"Set "<<f<<" running..."<<endl;
		int count=1;
		int neg_count=0;
		string path="/home/siplab/ALL_detection/patchDataset/";
		string newPath = "/home/siplab/ALL_detection/Augmented_Dataset/";
		ifstream file1("Filenames/f"+to_string(f)+"_p.txt");
		string imName;
		while(file1.good()){
			getline(file1,imName);
			if(imName.empty())continue;
			neg_count++;
			string imNameTot = path+""+imName;
			cv::Mat im = cv::imread(imName,1);
			// cv::imshow("im",im);
			// cv::waitKey(0);
			cv::Mat newImg = newImage(im);
			for(int i=0;i<18;i++){
				cv::Mat im2 = rotateImage(newImg,20*i,0);
				im2 = im2(cv::Rect(im.cols,im.rows,im.cols,im.rows));
				// cv::imshow("im2",im2);
				// cv::waitKey(0);
				cv::imwrite(newPath+to_string(count)+"_"+to_string(f)+".jpg",im2);
				count++;
				cv::flip(im2,im2,1);
				// cv::imshow("im2",im2);
				// cv::waitKey(0);
				cv::imwrite(newPath+to_string(count)+"_"+to_string(f)+".jpg",im2);
				count++;
			}
		}
		cout<<"before= "<<neg_count<<" ,now= "<<(neg_count*36)<<endl;
	}
	return 0;
}