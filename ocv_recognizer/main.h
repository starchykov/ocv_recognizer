// main.h

/// Use a unique include guard to avoid naming conflicts.
#ifndef MY_MAIN
#define MY_MAIN

/// Include necessary libraries.
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

/// Include other necessary header files.
#include "detect_plates.h"
#include "possible_plate.h"
#include "detect_chars.h"

/// Include standard libraries.
#include<iostream>
/// TODO may have to modify this line if not using Windows.
/// #include<conio.h>

/// TODO un-comment or comment this line to show steps or not.
//#define SHOW_STEPS

/// Define global application constants for scalar colors.
const cv::Scalar SCALAR_BLACK = cv::Scalar(0.0, 0.0, 0.0);
const cv::Scalar SCALAR_WHITE = cv::Scalar(255.0, 255.0, 255.0);
const cv::Scalar SCALAR_YELLOW = cv::Scalar(0.0, 255.0, 255.0);
const cv::Scalar SCALAR_GREEN = cv::Scalar(0.0, 255.0, 0.0);
const cv::Scalar SCALAR_RED = cv::Scalar(0.0, 0.0, 255.0);

/// Declare function prototypes.
int main();

void drawRedRectangleAroundPlate(cv::Mat &imgOriginalScene, possible_plate &licPlate, int thickness);

void writeLicensePlateCharsOnImage(cv::Mat &imgOriginalScene, possible_plate &licPlate);

/// Use a unique include guard to avoid naming conflicts.

# endif

