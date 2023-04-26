// This header file contains declarations of functions used for detecting license plates in an image

#ifndef DETECT_PLATES_H
#define DETECT_PLATES_H

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

#include "main.h"
#include "possible_plate.h"
#include "possible_char.h"
#include "preprocess.h"
#include "detect_chars.h"

/// Constants used for calculating plate width and height.
const double PLATE_WIDTH_PADDING_FACTOR = 1.3;
const double PLATE_HEIGHT_PADDING_FACTOR = 1.5;

/// Function declarations.
std::vector<possible_plate> detectPlatesInScene(cv::Mat &imgOriginalScene);

std::vector<possible_char> findPossibleCharsInScene(cv::Mat &imgThresh);

possible_plate extractPlate(cv::Mat &imgOriginal, std::vector<possible_char> &vectorOfMatchingChars);

#endif