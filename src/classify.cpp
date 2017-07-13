/*
 * classify.cpp
 *
 *  Created on: 25-Feb-2017
 *      Author: jeetkanjani7
*/
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <iostream>
#include <opencv2/ml/ml.hpp>

using namespace std;
using namespace cv;


const int MIN_CONTOUR_AREA=100;
const int RESIZED_IMAGE_WIDTH = 20;
const int RESIZED_IMAGE_HEIGHT = 30;
int main()
{

	Mat training_orig,training_gray,training_blur,training_thresh,training_clone;
	Mat trained_images,trained_classes;
	vector<Vec4i> hierarchy;
	vector<vector<Point> > vecpts;
	char[] digits = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9' };
	int train_classes[10];

	training_orig=imread("training_numbers.png",1);

	if(training_orig.empty())
	{
		cout<<"Training image Not found";
		return 0;
	}
	//Pre-processing

	cvtColor(training_orig,training_gray,CV_BGR2GRAY);

	GaussianBlur(training_gray,training_blur,Size(5,5),0);

	adaptiveThreshold(training_blur,training_thresh,255,ADAPTIVE_THRESH_GAUSSIAN_C,THRESH_BINARY_INV,11,2);

	training_clone=training_thresh.clone();


	//Segmentation

	findContours(training_clone,vecpts,hierarchy,CV_RETR_EXTERNAL,CHAIN_APPROX_SIMPLE);


	for(int i=0;i<vecpts.size();i++)
	{
		if(contourArea(vecpts[i])>MIN_CONTOUR_AREA)
		{
			Rect boundingRect = cv::boundingRect(vecpts[i]);

			rectangle(training_orig, boundingRect, Scalar(0, 0, 255), 2);
			Mat matROI = training_thresh(boundingRect);
			Mat matROIResized;
			resize(matROI, matROIResized, Size(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT));
			imshow("matROI", matROI);
			imshow("matROIResized", matROIResized);
			imshow("matTrainingNumbers", training_orig);

			int intChar = waitKey(0);


			if(intChar == 27)
			{
				return(0);
			}
			cout<<intChar;
			for(int j=48;j<58;j++)
			{
				if(intChar==j)
				{
					trained_classes.push_back(intChar);
					Mat floated_image;
					matROIResized.convertTo(floated_image,CV_32FC1);
					Mat matImageReshaped = floated_image.reshape(1, 1);
					trained_images.push_back(matImageReshaped);
				}
			}


		}
	}
	cout<<"training Complete";

	Mat matClassificationFloats;
	trained_images.convertTo(matClassificationFloats, CV_32FC1);

	FileStorage fsClassifications("classifications.xml", FileStorage::WRITE);

	if (fsClassifications.isOpened() == false)
	{
		std::cout << "error, unable to open training classifications file, exiting program\n\n";
		return(0);
	}

	fsClassifications << "classifications" << matClassificationFloats;
	fsClassifications.release();

	// save training images to file ///////////////////////////////////////////////////////

	FileStorage fsTrainingImages("images.xml", FileStorage::WRITE);

	if (fsTrainingImages.isOpened() == false)
	{
		std::cout << "error, unable to open training images file, exiting program\n\n";
		return(0);
	}

		fsTrainingImages << "images" << trained_images;
		fsTrainingImages.release();

	return(0);


}



