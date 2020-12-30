#include <numeric>
#include "matching2D.hpp"

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = true;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType = descriptorType.compare("DES_HOG") == 0 ? cv::NORM_L2 : cv::NORM_HAMMING;
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        if (descSource.type() != CV_32F)
          descSource.convertTo(descSource, CV_32F);
        if (descRef.type() != CV_32F)
          descRef.convertTo(descRef, CV_32F);        
      	matcher = cv::FlannBasedMatcher::create();       
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)

        std::vector<std::vector<cv::DMatch>> kmatches;
        matcher->knnMatch(descSource, descRef, kmatches, 2);
       	float disThres = 0.8;
      
        for (auto& kmatch : kmatches)
        {
          if(kmatch[0].distance < disThres * kmatch[1].distance)
              matches.push_back(kmatch[0]);
        }
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
double descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if (descriptorType.compare("BRIEF") == 0)
    {
      extractor = cv::xfeatures2d::BriefDescriptorExtractor::create(32, false);
    }
  	else if (descriptorType.compare("ORB") == 0)
    {
      int feat = 500;
      int edgeThres = 30;
      int fastThres = 20;
      auto scoretype = cv::ORB::FAST_SCORE;
      extractor = cv::ORB::create(feat, 1.2f, 8, edgeThres, 0, 2, scoretype, 31, fastThres);
    }
    else if (descriptorType.compare("FREAK") == 0)
    {
      extractor = cv::xfeatures2d::FREAK::create();
    }
  	else if (descriptorType.compare("AKAZE") == 0)
    {
      extractor = cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_KAZE,0,3,0.001f,4,4,cv::KAZE::DIFF_WEICKERT);
    }
  	else if (descriptorType.compare("SIFT") == 0)
    {
      extractor = cv::xfeatures2d::SIFT::create(0,3,0.04,10,1.6);
    }
  	
    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
    //cv::waitKey(0);
    return t;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
double detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector 
    for (auto& corner : corners)
  	{
    	cv::KeyPoint newKeyPoint;
    	newKeyPoint.pt = cv::Point2f(corner.x, corner.y);
    	newKeyPoint.size = blockSize;
    	keypoints.push_back(newKeyPoint);
 	 }  
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
    //cv::waitKey(0);
    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
    return t;
}
double detKeypointsHarris(vector<cv::KeyPoint>& keypoints, cv::Mat &img, bool bVis)
{
  int blockSize = 4;
  int apertureSize = 3;
  int minResponse = 100;
  double h = 0.04;
  
  cv::Mat dst, dst_norm, dst_norm_scaled;
  dst = cv::Mat::zeros(img.size(), CV_32FC1);
  
  double t = (double)cv::getTickCount();  
  cv::cornerHarris(img, dst, blockSize, apertureSize, h, cv::BORDER_DEFAULT);
  cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
  cv::convertScaleAbs(dst_norm, dst_norm_scaled);
  
  for (size_t j=0; j < dst_norm.rows; j++)
  {
    for (size_t i=0; i < dst_norm.cols; i++)
    {
      int response = (int)dst_norm.at<float>(j,i);
      if (response > minResponse)
      {
        cv::KeyPoint newPt;
        newPt.pt = cv::Point2f(i,j);
        newPt.size = 2 * apertureSize;
        newPt.response = response;
        
        bool bOverlap = false;
        for (auto it = keypoints.begin(); it != keypoints.end(); ++it)
        {
          double kptOverlap = cv::KeyPoint::overlap(newPt, *it);
          double maxOverlap = 0.00;
          if (kptOverlap > maxOverlap)
          {
            bOverlap = true;
            if (newPt.response > (*it).response)
            {
              *it = newPt;
              break;
            }
          }
        }
        if (!bOverlap)
          keypoints.push_back(newPt);
      }
    }
  }
  t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
  cout << "Harris detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
  
  if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Harris Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
  	return t;
}

double detKeypointsModern (std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis)
{
  cv::Ptr<cv::FeatureDetector> detector;
  if (detectorType.compare("FAST") == 0)
  {
    int thres = 30;
    bool nms = true;
    detector = cv::FastFeatureDetector::create(thres, nms, cv::FastFeatureDetector::TYPE_9_16);
  }
  else if (detectorType.compare("BRISK") == 0)
  {
    int thres = 30;
    int octnum = 3;
    detector = cv::BRISK::create(thres, octnum, 1.0f);
  }
  else if (detectorType.compare("ORB") == 0)
  {
    int feat = 500;
    int edgeThres = 30;
    int fastThres = 20;
    auto scoretype = cv::ORB::FAST_SCORE;
    detector = cv::ORB::create(feat, 1.2f, 8, edgeThres, 0, 2, scoretype, 31, fastThres);
  }
  else if (detectorType.compare("AKAZE") == 0)
  {
    detector = cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_KAZE,0,3,0.001f,4,4,cv::KAZE::DIFF_WEICKERT);
  }
  else if (detectorType.compare("SIFT") == 0)
  {
    detector = cv::xfeatures2d::SIFT::create(0,3,0.04,10,1.6);
  }
  
  double t = (double)cv::getTickCount();
  detector->detect(img, keypoints);
  t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
  cout << detectorType << "detection with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
  if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = detectorType + " Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
  return t;
}
  
  
  
  
    
  
    
  
    
  
            
            
          
        
      
    
  
  

