//#include <iostream>
//#include <opencv2/core.hpp>
//#include <opencv2/opencv.hpp>
//#include <opencv2/imgproc.hpp>
//
//#define LOG_TAG "opencv"
//
//#define DEFAULT_IDCARD_WIDTH  640
//#define DEFAULT_IDCARD_HEIGHT  320
//
//#define DEFAULT_IDNUMBER_WIDTH  240
//#define DEFAULT_IDNUMBER_HEIGHT  120
//
////#define  FIX_IDCARD_SIZE Size(DEFAULT_IDCARD_WIDTH,DEFAULT_IDCARD_HEIGHT)
//#define  FIX_IDNUMBER_SIZE  Size(DEFAULT_IDNUMBER_WIDTH,DEFAULT_IDNUMBER_HEIGHT)
////#define FIX_TEMPLATE_SIZE  Size(150, 26)
//
//#define TEMPLE_WIDTH 0.24
//#define TEMPLE_HEIGHT 0.07
//#define DEFAULT_CARD_WIDTH 640
//#define DEFAULT_CARD_HEIGHT 400
//#define  FIX_IDCARD_SIZE Size(DEFAULT_CARD_WIDTH,DEFAULT_CARD_HEIGHT)
//#define FIX_TEMPLATE_SIZE  Size(153, 28)
//
///**
//* 身份证识别
//* @param src_img 身份证的图片地址
//* @param dst_img 依据查找的图片的地址
//*/
//void icard(const char * src_img, const char * dst_img)
//{
//	//原始图
//	cv::Mat img_src = cv::imread(src_img);;
//	//模版
//	cv::Mat img_tpl = cv::imread(dst_img);
//	//灰度图 需要拿去模版匹配
//	cv::Mat img_gray;
//	//二值图 进行轮廓检测
//	cv::Mat img_threshold;
//	// Gaussian 图片
//	cv::Mat img_gaussian;
//	// Canny 图片
//	cv::Mat img_canny;
//	
//	//获得的身份证图
//	cv::Mat img_idCard;
//	//获得的身份证号码图
//	cv::Mat img_idNumber;
//	//Java_org_opencv_android_Utils_nBitmapToMat2(env, type, src, (jlong)&img_src, 0);
//	//Java_org_opencv_android_Utils_nBitmapToMat2(env, type, tpl, (jlong)&img_tpl, 0);
//
//	cvtColor(img_src, img_gray, cv::COLOR_BGRA2GRAY);
//	imwrite("../out/gray.png", img_gray);
//	threshold(img_gray, img_threshold, 195, 255, cv::THRESH_TRUNC);
//	imwrite("../out/threshold.png", img_threshold);
//
//	//高斯模糊让图片变的更加平滑
//	GaussianBlur(img_threshold, img_gaussian, cv::Size(3, 3), 0);
//	imwrite("../out/gaussian.png", img_gaussian);
//
//	//canny, 将边缘进行增强
//	Canny(img_gaussian, img_canny, 180, 255);
//	imwrite("../out/img_canny.png", img_canny);
//
//
//	std::vector<std::vector<cv::Point>> contours;
//	std::vector<cv::Vec4i> hierachy;
//	//轮廓检测 只检测外轮廓 并压缩水平方向，垂直方向，对角线方向的元素，只保留该方向的终点坐标，比如矩形就是存储四个点
//	// 第一个参数是原图，
//	// 第二个是获取轮廓的点的保存向量，是找到的各个轮廓
//	// 第三个是 层次结构，存放了轮廓同一等级的前后轮廓的索引，不同等级的父亲轮廓和孩子轮廓的索引
//	// 第四个参数 mode :CV_RETR_EXTERNAL找到的轮廓里面没有小轮廓（洞），CV_RETR_LIST找到的轮廓中可以包括小轮廓
//	// http://blog.csdn.net/corcplusplusorjava/article/details/20536251
//	findContours(img_canny, contours, hierachy, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
//	int width = img_src.cols >> 1;
//	int height = img_src.rows >> 1;
//
//	std::vector<cv::Rect> roiArea;
//	cv::Rect rectMin;
//	for (size_t i = 0; i < contours.size(); ++i)
//	{
//		std::vector<cv::Point> v = contours.at(i);
//		cv::Rect rect = boundingRect(v);
//		rectangle(img_threshold, rect, cv::Scalar(255, 255, 255));
//		if (rect.width >= width && rect.height >= height)
//		{
//			roiArea.push_back(rect);
//		}
//	}
//	if (roiArea.size() > 0)
//	{
//		rectMin = roiArea.at(0);
//		for (int i = 0; i < roiArea.size(); i++)
//		{
//			cv::Rect temp = roiArea.at(i);
//			if (temp.area() < rectMin.area())
//			{
//				rectMin = temp;
//			}
//		}
//		//        rectangle(img_threshold, rectMin, Scalar(255, 255, 255));
//	}
//	else
//	{
//		rectMin = cv::Rect(0, 0, img_gray.cols, img_gray.rows);
//	}
//	imwrite("../out/img_contours.png", img_threshold);
//
//	img_idCard = img_gray(rectMin);
//	imwrite("../out/img_idCard.png", img_idCard);
//
//	resize(img_idCard, img_idCard, cv::FIX_IDCARD_SIZE);
//	resize(img_tpl, img_tpl, cv::FIX_TEMPLATE_SIZE);
//	cvtColor(img_tpl, img_tpl, cv::COLOR_BGRA2GRAY);
//	int cols = img_idCard.cols - img_tpl.cols + 1;
//	int rows = img_idCard.rows - img_tpl.rows + 1;
//	//创建输出图像，输出图像的宽度 = 被查找图像的宽度 - 模版图像的宽度 + 1
//	//    Mat match(rows, cols, CV_32F);
//	//        TM_SQDIFF 平方差匹配法
//	//        TM_CCORR 相关匹配法
//	//        TM_CCOEFF 相关系数匹配法
//	//        TM_SQDIFF_NORMED
//	//        TM_CCORR_NORMED
//	//        TM_CCOEFF_NORMED
//	// 对于方法 SQDIFF 和 SQDIFF_NORMED, 越小的数值代表更高的匹配结果. 而对于其他方法, 数值越大匹配越好
//	cv::Mat match;
//	matchTemplate(img_idCard, img_tpl, match, cv::TM_CCORR_NORMED);
//	//归一化
//	cv::normalize(match, match, 0, 1, cv::NORM_MINMAX, -1);
//	cv::Point maxLoc;
//	minMaxLoc(match, 0, 0, 0, &maxLoc);
//	//计算 [身份证(模版):号码区域]
//	//号码区域:
//	//x: 身份证(模版)的X+宽
//	//y: 身份证(模版)Y
//	//w: 全图宽-(身份证(模版)X+身份证(模版)宽) - n(给个大概值)
//	//h: 身份证(模版)高
//	cv::Rect rect(maxLoc.x + img_tpl.cols + 10, maxLoc.y - 5,
//		img_idCard.cols - (maxLoc.x + img_tpl.cols) - 50,
//		img_tpl.rows + 15);
//	//拿二值的号码
//	//    resize(img_threshold, img_threshold, FIX_IDCARD_SIZE);
//	//    img_idNumber = img_threshold(rect);
//	img_idNumber = img_idCard(rect);
//	imwrite("../out/abc0.png", img_idNumber);
//	/*jobject result = createBitmap(env, img_idNumber, config);*/
//
//	img_src.release();
//	img_gray.release();
//	img_threshold.release();
//	img_idCard.release();
//	img_idNumber.release();
//	img_tpl.release();
//	match.release();
//}
//
//static void help()
//{
//	std::cout
//		<< "\n------------------------------------------------------------------\n"
//		<< " This program shows the serial out capabilities of cv::Mat\n"
//		<< "That is, cv::Mat M(...); cout << M;  Now works.\n"
//		<< "Output can be formatted to OpenCV, matlab, python, numpy, csv and \n"
//		<< "C styles Usage:\n"
//		<< "./cvout_sample\n"
//		<< "------------------------------------------------------------------\n\n"
//		<< std::endl;
//}
//
//void test_mat()
//{
//		//cv::CommandLineParser parser(argc, argv, "{help h||}");
//		//if (parser.has("help"))
//		//{
//		//	//help();
//		//	return ;
//		//}
//		cv::Mat I = cv::Mat::eye(4, 4, CV_64F);
//		I.at<double>(1, 1) = CV_PI;
//		std::cout << "I = \n" << I << ";" << std::endl;
//
//		cv::Mat r = cv::Mat(10, 3, CV_8UC3);
//		cv::randu(r, cv::Scalar::all(0), cv::Scalar::all(255));
//
//		std::cout << "r (default) = \n" << r << ";" << std::endl;
//		std::cout << "r (matlab) = \n" << format(r, cv::Formatter::FMT_MATLAB) << ";"  << std::endl;
//		std::cout << "r (python) = \n" << format(r, cv::Formatter::FMT_PYTHON) << ";" << std::endl;
//		std::cout << "r (numpy) = \n" << format(r, cv::Formatter::FMT_NUMPY) << ";"  << std::endl;
//		std::cout << "r (csv) = \n" << format(r, cv::Formatter::FMT_CSV) << ";" << std::endl;
//		std::cout << "r (c) = \n" << format(r, cv::Formatter::FMT_C) << ";" << std::endl;
//
//		cv::Point2f p(5, 1);
//		std::cout << "p = " << p << ";" << std::endl;
//
//		cv::Point3f p3f(2, 6, 7);
//		std::cout << "p3f = " << p3f << ";" << std::endl;
//
//		std::vector<float> v;
//		v.push_back(1);
//		v.push_back(2);
//		v.push_back(3);
//
//		std::cout << "shortvec = " << cv::Mat(v) << std::endl;
//
//		std::vector<cv::Point2f> points(20);
//		for (size_t i = 0; i < points.size(); ++i)
//		{
//			points[i] = cv::Point2f((float)(i * 5), (float)(i % 7));
//		}
//
//		std::cout << "points = " << points << ";" << std::endl;
//		return ;
//	
//}
//
//
//bool test_imagetomat(int  argc, char *argv[])
//{
//	//cv::CommandLineParser parser(argc, argv, "{@input | ../data/cards.png | input image}");
//	cv::Mat src = cv::imread("../data/cards.jpg"/*parser.get<cv::String>("@input")*/);
//	if (src.empty())
//	{
//		std::cout << "Could not open or find the image!\n" << std::endl;
//		std::cout << "Usage: " << argv[0] << " <Input image>" << std::endl;
//		return -1;
//	}
//
//	// Show source image
//	imshow("Source Image", src);
//	//for (int i = 0; i < src.rows; i++) {
//	//	for (int j = 0; j < src.cols; j++) {
//	//		if (src.at<cv::Vec3b>(i, j) == cv::Vec3b(255, 255, 255))
//	//		{
//	//			src.at<cv::Vec3b>(i, j)[0] = 0;
//	//			src.at<cv::Vec3b>(i, j)[1] = 0;
//	//			src.at<cv::Vec3b>(i, j)[2] = 0;
//	//		}
//	//	}
//	//}
//
//	//// Show output image
//	//imshow("Black Background Image", src);
//	////! [black_bg]
//
//	////! [sharp]
//	//// Create a kernel that we will use to sharpen our image
//	//cv::Mat kernel = (cv::Mat_<float>(3, 3) <<
//	//	1, 1, 1,
//	//	1, -8, 1,
//	//	1, 1, 1); // an approximation of second derivative, a quite strong kernel
//
//	//			  // do the laplacian filtering as it is
//	//			  // well, we need to convert everything in something more deeper then CV_8U
//	//			  // because the kernel has some negative values,
//	//			  // and we can expect in general to have a Laplacian image with negative values
//	//			  // BUT a 8bits unsigned int (the one we are working with) can contain values from 0 to 255
//	//			  // so the possible negative number will be truncated
//	//cv::Mat imgLaplacian;
//	//filter2D(src, imgLaplacian, CV_32F, kernel);
//	//cv::Mat sharp;
//	//src.convertTo(sharp, CV_32F);
//	//cv::Mat imgResult = sharp - imgLaplacian;
//
//	//// convert back to 8bits gray scale
//	//imgResult.convertTo(imgResult, CV_8UC3);
//	//imgLaplacian.convertTo(imgLaplacian, CV_8UC3);
//
//	//// imshow( "Laplace Filtered Image", imgLaplacian );
//	//imshow("New Sharped Image", imgResult);
//	////! [sharp]
//
//	////! [bin]
//	//// Create binary image from source image
//	//cv::Mat bw;
//	//cvtColor(imgResult, bw, cv::COLOR_BGR2GRAY);
//	//threshold(bw, bw, 40, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
//	//imshow("Binary Image", bw);
//	////! [bin]
//
//	////! [dist]
//	//// Perform the distance transform algorithm
//	//cv::Mat dist;
//	//distanceTransform(bw, dist, cv::DIST_L2, 3);
//
//	//// Normalize the distance image for range = {0.0, 1.0}
//	//// so we can visualize and threshold it
//	//normalize(dist, dist, 0, 1.0, cv::NORM_MINMAX);
//	//imshow("Distance Transform Image", dist);
//	////! [dist]
//
//	////! [peaks]
//	//// Threshold to obtain the peaks
//	//// This will be the markers for the foreground objects
//	//threshold(dist, dist, 0.4, 1.0, cv::THRESH_BINARY);
//
//	//// Dilate a bit the dist image
//	//cv::Mat kernel1 = cv::Mat::ones(3, 3, CV_8U);
//	//dilate(dist, dist, kernel1);
//	//imshow("Peaks", dist);
//	////! [peaks]
//
//	////! [seeds]
//	//// Create the CV_8U version of the distance image
//	//// It is needed for findContours()
//	//cv::Mat dist_8u;
//	//dist.convertTo(dist_8u, CV_8U);
//
//	//// Find total markers
//	//std::vector<std::vector<cv::Point> > contours;
//	//findContours(dist_8u, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
//
//	//// Create the marker image for the watershed algorithm
//	//cv::Mat markers = cv::Mat::zeros(dist.size(), CV_32S);
//
//	//// Draw the foreground markers
//	//for (size_t i = 0; i < contours.size(); i++)
//	//{
//	//	drawContours(markers, contours, static_cast<int>(i), cv::Scalar(static_cast<int>(i) + 1), -1);
//	//}
//
//	//// Draw the background marker
//	//cv::circle(markers, cv::Point(5, 5), 3, cv::Scalar(255), -1);
//	//imshow("Markers", markers * 10000);
//	////! [seeds]
//
//	////! [watershed]
//	//// Perform the watershed algorithm
//	//cv::watershed(imgResult, markers);
//
//	//cv::Mat mark;
//	//markers.convertTo(mark, CV_8U);
//	//cv::bitwise_not(mark, mark);
//	////    imshow("Markers_v2", mark); // uncomment this if you want to see how the mark
//	//// image looks like at that point
//
//	//// Generate random colors
//	//std::vector<cv::Vec3b> colors;
//	//for (size_t i = 0; i < contours.size(); i++)
//	//{
//	//	int b = cv::theRNG().uniform(0, 256);
//	//	int g = cv::theRNG().uniform(0, 256);
//	//	int r = cv::theRNG().uniform(0, 256);
//
//	//	colors.push_back(cv::Vec3b((uchar)b, (uchar)g, (uchar)r));
//	//}
//
//	//// Create the result image
//	//cv::Mat dst = cv::Mat::zeros(markers.size(), CV_8UC3);
//
//	//// Fill labeled objects with random colors
//	//for (int i = 0; i < markers.rows; i++)
//	//{
//	//	for (int j = 0; j < markers.cols; j++)
//	//	{
//	//		int index = markers.at<int>(i, j);
//	//		if (index > 0 && index <= static_cast<int>(contours.size()))
//	//		{
//	//			dst.at<cv::Vec3b>(i, j) = colors[index - 1];
//	//		}
//	//	}
//	//}
//
//	//// Visualize the final image
//	//imshow("Final Result", dst);
//	////! [watershed]
//
//	cv::waitKey();
//	return true;
//}
//
//int main(int argc,  char *argv[])
//{
//	
//	//test_mat();
//	//test_imagetomat(argc, argv);
//	icard("../data/id_card0.jpg", "../data/te.png");
//	//system("pause");
//	return EXIT_SUCCESS;
//}



#include <opencv2/opencv.hpp>
//#include <opencv2/contrib/contrib.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/core/ocl.hpp>
using namespace cv;

int main(int argc, char **argv)
{
	const int kWidth = 512, kHeight = 512;
	Vec3b red(0, 0, 255), green(0, 255, 0), blue(255, 0, 0);
	Mat image = Mat::zeros(kHeight, kWidth, CV_8UC3);
	// 训练样本标签数组
	int labels[150];
	for (int i = 0; i < 75; i++)
		labels[i] = 1;
	for (int i = 75; i < 150; i++)
		labels[i] = 2;
	Mat trainResponse(150, 1, CV_32SC1, labels);
	// 训练样本特征向量数组
	float trainDataArray[150][2];
	RNG rng;
	for (int i = 0; i < 75; i++)
	{
		trainDataArray[i][0] = 250 + static_cast<float>(rng.gaussian(30));
		trainDataArray[i][1] = 250 + static_cast<float>(rng.gaussian(30));
	}
	for (int i = 75; i < 150; i++)
	{
		trainDataArray[i][0] = 150 + static_cast<float>(rng.gaussian(30));
		trainDataArray[i][1] = 150 + static_cast<float>(rng.gaussian(30));
	}
	Mat trainData(150, 2, CV_32FC1, trainDataArray);
	float priors[2] = { 1, 1 };
	// AdaBoost训练参数，源代码分析中会解释，在这里我们设置弱分类器的个数
	// 为10
	CvBoostParams params(CvBoost::REAL, 10, 0.95, 5, false, priors);
	CvBoost boost;
	// 训练分类器
	boost.train(trainData, CV_ROW_SAMPLE, trainResponse, cv::Mat(), cv::Mat(),
		cv::Mat(), cv::Mat(), params);
	// 对图像内所有点进行预测，显示不同的颜色
	for (int i = 0; i < image.rows; i++)
	{
		for (int j = 0; j < image.cols; j++)
		{
			Mat sampleMat = (Mat_<float>(1, 2) << j, i);
			// 调用AdaBoost的预测函数
			float response = boost.predict(sampleMat);
			// 根据预测结果显示不同颜色
			if (response == 1)
				image.at<Vec3b>(i, j) = green;
			else
				image.at<Vec3b>(i, j) = blue;
		}
	}
	// 显示训练样本
	for (int i = 0; i < trainData.rows; i++)
	{
		const float* v = trainData.ptr<float>(i);
		Point pt = Point((int)v[0], (int)v[1]);
		if (labels[i] == 1)
			circle(image, pt, 5, Scalar::all(0), -1, 8);
		else
			circle(image, pt, 5, Scalar::all(255), -1, 8);
	}
	imshow("AdaBoost classifier demo", image);
	waitKey(0);
	return 0;
}