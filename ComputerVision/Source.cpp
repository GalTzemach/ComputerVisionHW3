#include <iostream>
#include <opencv2\opencv.hpp>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
	String windowRes = "Result";
	const float MIN_THRESHOLD = 1.9f; /// Original = 3

	Mat img1_source, img2_source, img1, img2;

	// Loading the original images(color).
	img1_source = imread("1.jpg"); //first
	img2_source = imread("2.jpg"); //second

	if (img1_source.empty() || img2_source.empty())
	{
		cout << "Error while loading image/s \n" << endl;
		return -1;
	}

	// Converts to grayscale to achieve better results
	cvtColor(img1_source, img1, COLOR_BGR2GRAY);
	cvtColor(img2_source, img2, COLOR_BGR2GRAY);


	//-- Step 1: Detect the keypoints
	Ptr<FeatureDetector> detector = ORB::create();
	vector<KeyPoint> keypoints_img1, keypoints_img2;

	detector->detect(img1, keypoints_img1);
	detector->detect(img2, keypoints_img2);

	//-- Step 2: Calculate descriptors (feature vectors)
	Ptr<DescriptorExtractor> extractor = ORB::create();
	Mat descriptors_img1, descriptors_img2;

	extractor->compute(img1, keypoints_img1, descriptors_img1);
	extractor->compute(img2, keypoints_img2, descriptors_img2);

	//-- Step 3: Matching descriptor vectors
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce");
	vector<DMatch> allMatches;
	double max_dist = 0, min_dist = 10000; /// Original min_dist = 100
	matcher->match(descriptors_img1, descriptors_img2, allMatches);


	vector<DMatch> allMatches2;
	double max_dist2 = 0, min_dist2 = 10000; /// Original min_dist = 100
	matcher->match(descriptors_img2, descriptors_img1, allMatches2);


	//-- Quick calculation of max and min distances between keypoints
	double dist;

	for (int i = 0; i < descriptors_img1.rows; i++)
	{
		dist = allMatches[i].distance;
		if (dist < min_dist)
			min_dist = dist;
		if (dist > max_dist)
			max_dist = dist;
	}

	cout << "Max distance 1: " << max_dist << endl;
	cout << "Min distance 1: " << min_dist << endl;

	for (int i = 0; i < descriptors_img2.rows; i++)
	{
		dist = allMatches2[i].distance;
		if (dist < min_dist2)
			min_dist2 = dist;
		if (dist > max_dist2)
			max_dist2 = dist;
	}

	cout << "Max distance 2: " << max_dist2 << endl;
	cout << "Min distance 2: " << min_dist2 << endl;

	//-- Draw only "good" matches (i.e. whose distance is less than MIN_THRESHOLD*min_dist )
	vector< DMatch > filteredMatches;
	for (int i = 0; i < descriptors_img1.rows; i++)
	{
		if (allMatches[i].distance < MIN_THRESHOLD * min_dist)
		{
			filteredMatches.push_back(allMatches[i]);
		}
	}

	cout << "All matches 1: " << allMatches.size() << endl;
	cout << "Filtered matches 1: " << filteredMatches.size() << endl;

	vector< DMatch > filteredMatches2;
	for (int i = 0; i < descriptors_img2.rows; i++)
	{
		if (allMatches2[i].distance < MIN_THRESHOLD * min_dist2)
		{
			filteredMatches2.push_back(allMatches2[i]);
		}
	}

	cout << "All matches 2: " << allMatches2.size() << endl;
	cout << "Filtered matches 2: " << filteredMatches2.size() << endl;


	vector< DMatch > filteredMatchesFinal;
	for (int i = 0; i < filteredMatches.size(); i++)
	{
		for (int j = 0; j < filteredMatches2.size(); j++)
		{
			if (filteredMatches[i].distance == filteredMatches2[j].distance
				&& filteredMatches[i].queryIdx == filteredMatches2[j].trainIdx
				&& filteredMatches[i].trainIdx == filteredMatches2[j].queryIdx)
			{
				filteredMatchesFinal.push_back(filteredMatches[i]);
			}
		}
	}

	cout << "Filtered matches Final: " << filteredMatchesFinal.size() << endl;

	Mat img_result, img_result2;
	drawMatches(img1_source, keypoints_img1, img2_source, keypoints_img2, filteredMatchesFinal, img_result, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	
	img_result2 = img_result.clone();

	while (img_result2.cols > 1920 || img_result2.rows > 1080)
	{
		resize(img_result2, img_result2, Size(), 0.9, 0.9);
	}

	imwrite("res.jpg", img_result);
	imwrite("res2.jpg", img_result2);


	namedWindow(windowRes, CV_WINDOW_KEEPRATIO);
	//setWindowProperty(windowRes, CV_WND_PROP_ASPECTRATIO, CV_WINDOW_KEEPRATIO);
	setWindowProperty(windowRes, CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
	//resizeWindow(windowRes, img_result.cols, img_result.rows);
	imshow(windowRes, img_result);


	


	///// From here is optional
	////-- Localize the object
	//vector<Point2f> obj;
	//vector<Point2f> scene;

	//for (int i = 0; i < filteredMatches.size(); i++)
	//{
	//	//-- Get the keypoints from the good matches
	//	obj.push_back(keypoints_img1[filteredMatches[i].queryIdx].pt);
	//	scene.push_back(keypoints_img2[filteredMatches[i].trainIdx].pt);
	//}

	//Mat H = findHomography(obj, scene, CV_RANSAC);

	////-- Get the corners from the image_1 ( the object to be "detected" )
	//vector<Point2f> obj_corners(4);
	//obj_corners[0] = cvPoint(0, 0);
	//obj_corners[1] = cvPoint(img1.cols, 0);
	//obj_corners[2] = cvPoint(img1.cols, img1.rows);
	//obj_corners[3] = cvPoint(0, img1.rows);

	//vector<Point2f> scene_corners(4);
	//perspectiveTransform(obj_corners, scene_corners, H);

	////-- Draw lines between the corners (the mapped object in the scene - image_2 )
	//line(img_result, scene_corners[0] + Point2f(img1.cols, 0), scene_corners[1] + Point2f(img1.cols, 0), Scalar(0, 255, 0), 4);
	//line(img_result, scene_corners[1] + Point2f(img1.cols, 0), scene_corners[2] + Point2f(img1.cols, 0), Scalar(0, 255, 0), 4);
	//line(img_result, scene_corners[2] + Point2f(img1.cols, 0), scene_corners[3] + Point2f(img1.cols, 0), Scalar(0, 255, 0), 4);
	//line(img_result, scene_corners[3] + Point2f(img1.cols, 0), scene_corners[0] + Point2f(img1.cols, 0), Scalar(0, 255, 0), 4);

	////-- Show detected matches
	//namedWindow("Good Matches & Object detection 2", CV_WINDOW_KEEPRATIO);
	//imshow("Good Matches & Object detection 2", img_result);

	waitKey(0);
	return 0;

} // End main