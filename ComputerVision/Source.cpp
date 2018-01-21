#include <opencv2\opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;


int main(int argc, char** argv)
{
	String windowRes = "Result window";
	const float MIN_THRESHOLD = 10.0f; /// Original = 3
	clock_t startTime, endTime;
	startTime = clock();

	Mat img1_source, img2_source, img1, img2;

	// Loading the original images(color).
	img1_source = imread("first.jpg"); //first
	img2_source = imread("second.jpg"); //second

	if (img1_source.empty() || img2_source.empty())
	{
		cout << "Error while loading image/s \n" << endl;
		return -1;
	}

	// Converts to grayscale to achieve better results
	cvtColor(img1_source, img1, COLOR_BGR2GRAY);
	cvtColor(img2_source, img2, COLOR_BGR2GRAY);


	//-- Step 1: Detect the keypoints
	Ptr<FeatureDetector> detector = ORB::create(10000, 1.2, 8, 31, 0, 2, 0, 31, 20);
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

	// Find matches from img1 to img2
	vector<DMatch> allMatches1;
	double max_dist1 = 0, min_dist1 = 10000;
	matcher->match(descriptors_img1, descriptors_img2, allMatches1);

	// Find matches from img2 to img1
	vector<DMatch> allMatches2;
	double max_dist2 = 0, min_dist2 = 10000;
	matcher->match(descriptors_img2, descriptors_img1, allMatches2);


	//-- Quick calculation of max and min distances between keypoints
	double dist;

	// From img1 to img2
	for (int i = 0; i < allMatches1.size(); i++)
	{
		dist = allMatches1[i].distance;
		if (dist < min_dist1)
			min_dist1 = dist;
		if (dist > max_dist1)
			max_dist1 = dist;
	}

	// From img2 to img1
	for (int i = 0; i < descriptors_img2.rows; i++)
	{
		dist = allMatches2[i].distance;
		if (dist < min_dist2)
			min_dist2 = dist;
		if (dist > max_dist2)
			max_dist2 = dist;
	}

	//-- Draw only "good" matches (i.e. whose distance is less than MIN_THRESHOLD*min_dist )
	vector< DMatch > filteredMatches;

	// Filter allMatches1 by  MIN_THRESHOLD
	for (int i = 0; i < descriptors_img1.rows; i++)
	{
		if (allMatches1[i].distance < MIN_THRESHOLD * min_dist1)
		{
			filteredMatches.push_back(allMatches1[i]);
		}
	}

	cout << "Max distance 1: " << max_dist1 << endl;
	cout << "Min distance 1: " << min_dist1 << endl;
	cout << "All matches 1: " << allMatches1.size() << endl;
	cout << "Filtered matches 1: " << filteredMatches.size() << endl;

	vector< DMatch > filteredMatches2;

	// Filter allMatches2 by  MIN_THRESHOLD
	for (int i = 0; i < descriptors_img2.rows; i++)
	{
		if (allMatches2[i].distance < MIN_THRESHOLD * min_dist2)
		{
			filteredMatches2.push_back(allMatches2[i]);
		}
	}

	cout << "Max distance 2: " << max_dist2 << endl;
	cout << "Min distance 2: " << min_dist2 << endl;
	cout << "All matches 2: " << allMatches2.size() << endl;
	cout << "Filtered matches 2: " << filteredMatches2.size() << endl;

	// Filter again and takes only results that appear in both vector
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

	Mat img_result;
	drawMatches(img1_source, keypoints_img1, img2_source, keypoints_img2, filteredMatchesFinal, img_result, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	// Save the img_result as a file.
	imwrite("res.jpg", img_result);


	namedWindow(windowRes, CV_WINDOW_KEEPRATIO);
	//setWindowProperty(windowRes, CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
	//setWindowProperty(windowRes, CV_WND_PROP_ASPECTRATIO, CV_WINDOW_KEEPRATIO);
	//resizeWindow(windowRes, img_result.cols, img_result.rows);
	imshow(windowRes, img_result);

	endTime = clock();
	cout << "Total time = " << (double) (endTime - startTime) / CLOCKS_PER_SEC << " Sec" << endl;


	// #################################################################################
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