// possible_plate.h

#ifndef POSSIBLE_PLATE_H
#define POSSIBLE_PLATE_H

#include <string>

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

///////////////////////////////////////////////////////////////////////////////////////////////////
class possible_plate {
public:
    // member variables ///////////////////////////////////////////////////////////////////////////
    cv::Mat imgPlate;
    cv::Mat imgGrayscale;
    cv::Mat imgThresh;

    cv::RotatedRect rrLocationOfPlateInScene;

    std::string strChars;

    ///////////////////////////////////////////////////////////////////////////////////////////////
    static bool sortDescendingByNumberOfChars(const possible_plate &ppLeft, const possible_plate &ppRight) {
        return(ppLeft.strChars.length() > ppRight.strChars.length());
    }

};


#endif		// end #ifndef POSSIBLE_PLATE_H

