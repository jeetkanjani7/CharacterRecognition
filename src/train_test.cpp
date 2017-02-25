/*
 * train_test.cpp
 *
 *  Created on: 25-Feb-2017
 *      Author: jeetkanjani7
*/

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<stdio.h>
#include<iostream>
#include<opencv2/ml/ml.hpp>
#include<sstream>
using namespace std;
using namespace cv;

const int MIN_CONTOUR_AREA = 100;
const int RESIZED_IMAGE_WIDTH = 20;
const int RESIZED_IMAGE_HEIGHT = 30;


class contour_data
{
public:
	vector<Point> conpts;
	Rect bounding_rect;
	float floatarea;

	bool check_valid_contour()
	{
		if (floatarea < MIN_CONTOUR_AREA)
			return false;
	}

	static bool sortByBoundingRectXPosition(const contour_data& cdLeft, const contour_data& cdRight) {
			return(cdLeft.bounding_rect.x < cdRight.bounding_rect.x);
	}
};

int main()
{
	string output;
	std::vector<contour_data> allContoursWithData;
	std::vector<contour_data> validContoursWithData;
	Mat trainedimages,trainedclasses,testimage;

	FileStorage fsreadimages("images.xml",FileStorage::READ);
	if (fsreadimages.isOpened() == false) {
			std::cout << "error, unable to open training classifications file, exiting program\n\n";
			return(0);
	}
	fsreadimages["images"]>>trainedimages;

	fsreadimages.release();

	FileStorage fsreadclasses("classifications.xml",FileStorage::READ);
	if (fsreadclasses.isOpened() == false) {
				std::cout << "error, unable to open training classifications file, exiting program\n\n";
				return(0);
		}
	fsreadclasses["classifications"]>>trainedclasses;

	fsreadclasses.release();


	Mat newimage1,newimage0;
	trainedclasses.reshape(0, 1).convertTo(newimage1, CV_32FC1);
	trainedimages.reshape(0,1).convertTo(newimage0, CV_32FC1);

	cout<<newimage0.size();
	cout<<newimage1.size();
	cv::KNearest kNearest=cv::KNearest();

	try {
		kNearest.train(newimage0,newimage1);
	} catch (Exception e) {
		cout<<"exception in training";
	}

	testimage=imread("testimage.png",1);
	if(testimage.empty())
	{
		cout<<"test image not found";
		return 0;
	}

		Mat matGrayscale;
		Mat matBlurred;
		Mat matThresh;
		Mat matThreshCopy;

		cv::cvtColor(testimage, matGrayscale, CV_BGR2GRAY);
		cv::GaussianBlur(matGrayscale,matBlurred,cv::Size(5, 5),0);
		cv::adaptiveThreshold(matBlurred,matThresh,255,cv::ADAPTIVE_THRESH_GAUSSIAN_C,cv::THRESH_BINARY_INV,11,	2);

		matThreshCopy = matThresh.clone();

		std::vector<std::vector<cv::Point> > ptContours;
		std::vector<cv::Vec4i> v4iHierarchy;

		cv::findContours(matThreshCopy,	ptContours,v4iHierarchy,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);

		for (int i = 0; i < ptContours.size(); i++) {
				contour_data cd;
				cd.conpts = ptContours[i];
				cd.bounding_rect= cv::boundingRect(cd.conpts);
				cd.floatarea = cv::contourArea(cd.conpts);
				allContoursWithData.push_back(cd);
		}

		for (int i = 0; i < allContoursWithData.size(); i++) {
				if (allContoursWithData[i].check_valid_contour()) {
					validContoursWithData.push_back(allContoursWithData[i]);
				}
		}
		sort(validContoursWithData.begin(), validContoursWithData.end(), contour_data::sortByBoundingRectXPosition);


		for(int i=0;i<validContoursWithData.size();i++)
		{
			rectangle(testimage,validContoursWithData[i].bounding_rect,Scalar(0,255,0),2);
			Mat matroi = matThresh(validContoursWithData[i].bounding_rect);
			Mat matroiresized,matfloated;
			resize(matroi,matroiresized,Size(RESIZED_IMAGE_WIDTH,RESIZED_IMAGE_HEIGHT));
			matroiresized.convertTo(matfloated,CV_32FC1);
			float curcharacter=kNearest.find_nearest(matfloated.reshape(1,1),1);
			output=output+char(int(curcharacter));

		}

		imshow("test image",testimage);
		cout<<"\n\nThe number in the image is:" <<output;
		cvWaitKey(0);
		return 0;

}
