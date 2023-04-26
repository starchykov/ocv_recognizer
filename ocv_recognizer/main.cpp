#include "main.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "tesseract/baseapi.h"


int main() {

    tesseract::TessBaseAPI tess;
    tess.Init(nullptr, "eng");

    /// Attempt KNN training.
    bool knnTrainingSuccessful = loadKNNDataAndTrainKNN();

    /// If KNN training was not successful show error message and exit the program.
    if (!knnTrainingSuccessful) {
        std::cout << "Error: KNN training was not successful" << std::endl;
        return 0;
    }

    /// Define source image name.
    std::string filename;

    std::cout << "Print image name filename: ";
    std::cin >> filename;

    /// Read image from file.
    cv::Mat imgOriginalScene = cv::imread("/Users/starchykov/Projects/Me/ocv_recognizer/ocv_recognizer/sources/" + filename);

    /// If unable to open image then show error message on command line and exit program.
    if (imgOriginalScene.empty()) {
        std::cout << "Error: image not read from file" << std::endl;
        return 0;
    }

//    /// Load the Haar cascade classifier for license plate detection.
//    cv::CascadeClassifier plate_cascade;
//    plate_cascade.load("/Users/starchykov/Projects/Me/ocv_recognizer/ocv_recognizer/haarcascade_plate_number.xml");
//
//    // Detect license plates in the image
//    std::vector<cv::Rect> plates;
//    plate_cascade.detectMultiScale(imgOriginalScene, plates, 1.1, 3, 0, cv::Size(100, 30));
//
//    // Process each license plate detected
//    for (int i = 0; i < plates.size(); i++) {
//        // Extract the license plate region from the image
//        cv::Mat plate_region = imgOriginalScene(plates[i]);
//
//        std::vector<possible_plate> plates = detectPlatesInScene(plate_region);
//        plates = detectCharsInPlates(plates);
//
//        cv::imshow("License Plates", plate_region);
//
//        // Convert the license plate region to grayscale
//        cv::Mat gray;
//        cv::cvtColor(plate_region, gray, cv::COLOR_BGR2GRAY);
//
//        // Convert the grayscale image to a Leptonica Pix object
//        Pix *pix = pixCreate(gray.cols, gray.rows, 8);
//        pixSetData(pix, reinterpret_cast<l_uint32 *>(gray.data));
//
//        // Set the image for Tesseract to recognize
//        tess.SetImage(pix);
//
//        // Perform OCR and get the recognized text
//        char *text = tess.GetUTF8Text();
//
//        // Output the recognized text
//        std::cout << "License plate " << i + 1 << ": " << text << std::endl;
//
//        // Clean up
//        delete[] text;
//
//        // Release memory allocated by gray
//        gray.release();
//    }
//
//    cv::waitKey(0);


    /// Detect plates.
    std::vector<possible_plate> plates = detectPlatesInScene(imgOriginalScene);

    /// Detect characters in plates.
    plates = detectCharsInPlates(plates);

    /// Show the original image.
    cv::imshow("imgOriginalScene", imgOriginalScene);

    /// If no plates were found, inform user and exit program.
    if (plates.empty()) {
        std::cout << "No license plates were detected" << std::endl;
        return 0;
    }

    /// Sort the vector of possible plates in descending order (most number of chars to least number of chars).
    std::sort(plates.begin(), plates.end(), possible_plate::sortDescendingByNumberOfChars);

    /// Get the plate with the most recognized characters (the first plate in sorted by string length descending order).
    possible_plate licPlate = plates.front();

    /// Show crop of plate and threshold of plate.
    cv::imshow("imgPlate", licPlate.imgPlate);
    cv::imshow("imgThresh", licPlate.imgThresh);

    /// If no characters were found in the plate show message and exit program.
    if (licPlate.strChars.empty()) {
        std::cout << "No characters were detected" << std::endl;
        return 0;
    }

    /// Draw red rectangle around plate.
    drawRedRectangleAroundPlate(imgOriginalScene, licPlate, 1);

    /// Write license plate text on the image.
    writeLicensePlateCharsOnImage(imgOriginalScene, licPlate);

    /// Show the image with the license plate and the red rectangle around it.
    cv::imshow("imgOriginalScene", imgOriginalScene);

    /// Write image out to file.
    cv::imwrite("imgOriginalScene.png", imgOriginalScene);

    /// Wait for a key press before closing the windows.
    cv::waitKey(0);

    cv::destroyAllWindows();

    cv::waitKey(0);

    std::string result;

    std::cout << "Do you want check another image? y/n: ";
    std::cin >> result;


    if (result == "y" || result == "yes") return main();
    else return 0;
}


/**
 * Draws a red rectangle around the given possible_plate in the input image.
 * @param imgOriginalScene Input image.
 * @param licPlate possible_plate object.
 * @param thickness Thickness of the lines in the rectangle.
 */
void drawRedRectangleAroundPlate(cv::Mat &imgOriginalScene, possible_plate &licPlate, int thickness = 2) {
    /// Get the 4 vertices of the rotated rectangle.
    cv::Point2f vertices[4];
    licPlate.rrLocationOfPlateInScene.points(vertices);

    /// Draw 4 red lines to form the rectangle.
    for (int i = 0; i < 4; i++) {
        cv::line(imgOriginalScene, vertices[i], vertices[(i + 1) % 4], SCALAR_RED, thickness);
    }
}


/**
 * This function writes the license plate characters on the original image.
 *
 * @param imgOriginalScene the input image
 * @param licPlate the license plate to write on the image
 */
void writeLicensePlateCharsOnImage(cv::Mat &imgOriginalScene, possible_plate &licPlate) {

    // Calculate font properties.
    const int intFontFace = cv::FONT_HERSHEY_SIMPLEX;                       /// Use plain jane font.
    const double dblFontScale = (double) licPlate.imgPlate.rows / 30.0;     /// Scale the font size based on the height of the plate area.
    const int intFontThickness = (int) std::round(dblFontScale * 1.5);      /// Set the font thickness based on the font scale.
    int intBaseline = 0;

    // Calculate the size of the text.
    cv::Size textSize = cv::getTextSize(
            licPlate.strChars,      /// The text to be written.
            intFontFace,            /// The font face.
            dblFontScale,           /// The font scale.
            intFontThickness,       /// The font thickness.
            &intBaseline            /// Pointer to baseline value.
    );

    /// Calculate the location where the text will be written.
    cv::Point ptCenterOfTextArea = cv::Point(
            (int) licPlate.rrLocationOfPlateInScene.center.x,                           /// The x-coordinate is the center of the license plate.
            (int) licPlate.rrLocationOfPlateInScene.center.y +                          /// The y-coordinate is the center of the license plate plus.
            ((licPlate.rrLocationOfPlateInScene.size.height / 2) + textSize.height)     /// Half of the height of the license plate plus the height of the text.

    );

    /// Write the text on the image using the defined font, size, and color.
    cv::putText(
            imgOriginalScene,       /// The input image.
            licPlate.strChars,      /// The text to be written.
            ptCenterOfTextArea,     /// The location where the text will be written.
            intFontFace,            /// The font face.
            dblFontScale,           /// The font scale.
            SCALAR_YELLOW,          /// The color of the text.
            intFontThickness        /// The thickness of the text.
    );
}


