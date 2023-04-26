// DetectChars.h

#ifndef DETECT_CHARS_H
#define DETECT_CHARS_H

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/ml/ml.hpp>

#include "main.h"
#include "possible_char.h"
#include "possible_plate.h"
#include "preprocess.h"

// global constants ///////////////////////////////////////////////////////////////////////////////
// constants for checkIfPossibleChar, this checks one possible char only (does not compare to another char)
const int MIN_PIXEL_WIDTH = 2;
const int MIN_PIXEL_HEIGHT = 8;

const double MIN_ASPECT_RATIO = 0.25;
const double MAX_ASPECT_RATIO = 1.0;

const int MIN_PIXEL_AREA = 80;

// constants for comparing two chars
const double MIN_DIAG_SIZE_MULTIPLE_AWAY = 0.3;
const double MAX_DIAG_SIZE_MULTIPLE_AWAY = 5.0;

const double MAX_CHANGE_IN_AREA = 0.5;

const double MAX_CHANGE_IN_WIDTH = 0.8;
const double MAX_CHANGE_IN_HEIGHT = 0.2;

const double MAX_ANGLE_BETWEEN_CHARS = 12.0;

// other constants
const int MIN_NUMBER_OF_MATCHING_CHARS = 3;

const int RESIZED_CHAR_IMAGE_WIDTH = 20;
const int RESIZED_CHAR_IMAGE_HEIGHT = 30;

const int MIN_CONTOUR_AREA = 100;

// external global variables //////////////////////////////////////////////////////////////////////
extern const bool blnShowSteps;
extern cv::Ptr<cv::ml::KNearest>  kNearest;

// function prototypes ////////////////////////////////////////////////////////////////////////////

bool loadKNNDataAndTrainKNN();

std::vector<possible_plate> detectCharsInPlates(std::vector<possible_plate> &vectorOfPossiblePlates);

std::vector<possible_char> findPossibleCharsInPlate(cv::Mat &imgGrayscale, cv::Mat &imgThresh);

bool checkIfPossibleChar(possible_char &possibleChar);

std::vector<std::vector<possible_char> > findVectorOfVectorsOfMatchingChars(const std::vector<possible_char> &vectorOfPossibleChars);

std::vector<possible_char> findVectorOfMatchingChars(const possible_char &possibleChar, const std::vector<possible_char> &vectorOfChars);

double distanceBetweenChars(const possible_char &firstChar, const possible_char &secondChar);

double angleBetweenChars(const possible_char &firstChar, const possible_char &secondChar);

std::vector<possible_char> removeInnerOverlappingChars(std::vector<possible_char> &vectorOfMatchingChars);

std::string recognizeCharsInPlate(cv::Mat &imgThresh, std::vector<possible_char> &vectorOfMatchingChars);


#endif	// DETECT_CHARS_H

