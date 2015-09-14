/*
* Copyright (c) 2011. Philipp Wagner <bytefish[at]gmx[dot]de>.
* Released to public domain under terms of the BSD Simplified license.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above copyright
*     notice, this list of conditions and the following disclaimer in the
*     documentation and/or other materials provided with the distribution.
*   * Neither the name of the organization nor the names of its contributors
*     may be used to endorse or promote products derived from this software
*     without specific prior written permission.
*
*   See <http://www.opensource.org/licenses/bsd-license>
*/
#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/ml/ml.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <deque>
#include <stdlib.h>
#include <stdio.h>
#include <sstream>
#include <iostream>       // std::cout
#include <string>
#include <ctime>
//#include <chrono>



using namespace cv;
using namespace std;

uchar segmentation=8;
typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;
#define PI 3.1415926535897932
#define POWER(nBit) (1 << (nBit))

static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(CV_StsBadArg, error_message);
    }
    string line, path, classlabel;
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if(!path.empty() && !classlabel.empty()) {
            images.push_back(imread(path, 0));
            labels.push_back(atoi(classlabel.c_str()));
        }
    }
}
int UniformPattern59[256] =
{
    1,    2,   3,   4,   5,   0,   6,   7,   8,   0,   0,   0,   9,   0,  10,  11,
    12,   0,   0,   0,   0,   0,   0,   0,  13,   0,   0,   0,  14,   0,  15,  16,
    17,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    18,   0,   0,   0,   0,   0,   0,   0,  19,   0,   0,   0,  20,   0,  21,  22,
    23,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    24,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    25,   0,   0,   0,   0,   0,   0,   0,  26,   0,   0,   0,  27,   0,  28,  29,
    30,  31,   0,  32,   0,   0,   0,  33,   0,   0,   0,   0,   0,   0,   0,  34,
    0,    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  35,
    0,    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  36,
    37,  38,   0,  39,   0,   0,   0,  40,   0,   0,   0,   0,   0,   0,   0,  41,
    0,    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  42,
    43,  44,   0,  45,   0,   0,   0,  46,   0,   0,   0,   0,   0,   0,   0,  47,
    48,  49,   0,  50,   0,   0,   0,  51,  52,  53,   0,  54,  55,  56,  57,  58
};

float* get_histogram(Mat frame)
{
    ushort nbins = segmentation * segmentation * 59;

    unsigned short minrows = USHRT_MAX;
    unsigned short mincols = USHRT_MAX;

    if (frame.rows < minrows)	minrows = frame.rows;
    if (frame.cols < mincols)	mincols = frame.cols;

    unsigned short srows = (minrows-2) % segmentation;
    unsigned short scols = (mincols-2) % segmentation;
    minrows -= srows;
    mincols -= scols;


    unsigned short offsetY = (unsigned short)((frame.rows - minrows) / 2);
    unsigned short offsetX = (unsigned short)((frame.cols - mincols) / 2);

    //cout<<"offsetY"<<offsetY<<"  offsetX"<<offsetX<<"  mincols"<<mincols<<"  minrows"<<minrows<<endl;

    cv::Rect_<unsigned short> rect (offsetX, offsetY, mincols, minrows);
    frame = cv::Mat(frame, rect);
    //cout<<frame<<"\n";
    //nbins = segmentation * segmentation * 59;
    ushort rows = (ushort)frame.rows;
    ushort cols = (ushort)frame.cols;
    uint *xyaverage=new uint[rows * cols];
    float *histogram=new float[nbins];
    memset(xyaverage, 0, rows * cols * sizeof(int));
    memset(histogram, 0, nbins * sizeof(float));
    uint snumber = 0;
    //	cout<<rows<<cols<<frame.rows<<frame.cols<<"\n";
    for (ushort r = 0; r < rows; r++)
    {
        for (ushort c = 0; c < cols; c++)
        {
            xyaverage[r*cols+c] = frame.at<unsigned char>(r,c);
        }
    }
    //	lbp::XYBlock(xyaverage, segmentation, rows, cols, snumber, offset * nbins, histogram);
    const ushort rstep = (rows - 2) / segmentation;
    const ushort cstep = (cols - 2) / segmentation;
    //	cout<<"r c step"<<rstep<<" "<<cstep<<"\n";

    for (ushort r = rstep + 1; r <= (rows - 1); r += rstep)
    {
        for (ushort c = cstep + 1; c <= (cols - 1); c += cstep)
        {
            //lbp::Histogram(xyaverage, cols, r - rstep, r, c - cstep, c, snumber, offset, 0, histogram);
            uint *average=xyaverage;
            ushort rstart=r - rstep;
            ushort rend=r-1;
            ushort cstart=c - cstep;
            ushort cend=c-1;


            //cout<<cstart<<","<<cend<<"\n";
            for (ushort r0 = rstart; r0 < rend; r0++)
            {
                for (ushort c0 = cstart; c0 < cend; c0++)
                {
                    uchar CenterByte = (uchar)average[ r0 * cols + c0];
                    int BasicLBP = 0;

                    for (uchar p = 0; p < 8; p++)
                    {
                        int X = (int)(c0 + cos((2 * PI * p) / 8) + 0.5);
                        int Y = (int)(r0 - sin((2 * PI * p) / 8) + 0.5);

                        uchar CurrentByte = (uchar)average[Y * cols + X];

                        if (CurrentByte >= CenterByte)
                        {
                            BasicLBP += POWER(p);
                        }
                    }
                    histogram[UniformPattern59[BasicLBP] + snumber * 59]++;
                }
            }

            //lbp::Normalisation(snumber, offset, histogram);
            ushort start = snumber * 59;
            for (ushort i = start; i < (start + 59); i++)	histogram[i]/=rstep*cstep;
            snumber++;
            //cout<<"snumber: "<<snumber<<"\n";
        }
    }
    delete xyaverage;
    return histogram;
}

bool comp_func(Rect i,Rect j)
{
    return (i.width>j.width);
}


int main(int argc, const char *argv[]) {
    // Check for valid command line arguments, print usage
    // if no arguments were given.


    int choi;
    cout<<"What would you like to do today?"<<endl<<"[1] Add a new user\n"<<"[2] continue\n";
    cin>>choi;

    /*    if (argc != 4) {
    cout << "usage: " << argv[0] << " </path/to/haar_cascade> </path/to/csv.ext> </path/to/device id>" << endl;
    cout << "\t </path/to/haar_cascade> -- Path to the Haar Cascade for face detection." << endl;
    cout << "\t </path/to/csv.ext> -- Path to the CSV file with the face database." << endl;
    cout << "\t <device id> -- The webcam device id to grab frames from." << endl;
    exit(1);
    } */
    // Get the path to your CSV:
    //    string fn_haar = string(argv[1]);
    string fn_csv = "list.csv";
    //    int deviceId = atoi(argv[3]);
    // These vectors hold the images and corresponding labels:
    vector<Mat> images;
    vector<int> labels;
    CvSVM svm;
    Rect face_i;
    string box_text[2],name;
    int prediction;
    int pos_x[2],pos_y[2];
    bool var=false;
    Mat face, face_resized;
    vector<Rect> eyes1,eyes2,* eyes;
    vector<Rect> faces;
    deque<int> smile_dq (5,0);
    int smile_lab=0;
    // Read in the data (fails if no valid input filename is given, but you'll get an error message):
    svm.load("SVM_C_RBF_to_test");

    // Get the height from the first image. We'll need this
    // later in code to reshape the images to their original
    // size AND we need to reshape incoming faces to this size:

    // Create a FaceRecognizer and train it on the given images:

    // That's it for learning the Face Recognition model. You now
    // need to create the classifier for the task of Face Detection.
    // We are going to use the haar cascade you have specified in the
    // command line arguments:
    //
    CascadeClassifier haar_cascade;
    CascadeClassifier haar_cascade_profile;
    CascadeClassifier haar_cascade_smile;
    CascadeClassifier haar_cascade_eyes;
    haar_cascade.load("/home/varun/haarcascade_frontalface_default.xml");
    haar_cascade_profile.load("/home/varun/lbpcascade_profileface.xml");
    haar_cascade_smile.load("/home/varun/haarcascade_smile.xml");
    haar_cascade_eyes.load("/home/varun/haarcascade_eye_tree_eyeglasses.xml");
    // Get a handle to the Video device:
    VideoCapture cap(0);
    // Check if we can use this device at all:
    if(!cap.isOpened()) {
        cerr << "Capture Device ID " << 0 << "cannot be opened." << endl;
        return -1;
    }
    // Holds the current frame from the Video device:
    Mat frame;
    int fr_ctr=0;

//cout<<"mark -1"<<endl;
     if (choi==1)
    {
        int fl_cnt;
        string new_fl,file_nm,name_list="/home/varun/images/name_list.txt",label_file;
        ifstream myfile;
        ofstream myfile2;
        fstream myfile3;
        myfile.open("/home/varun/images/file_cnt");
        myfile>>fl_cnt;
        myfile.close();
        fl_cnt++;
        myfile2.open("/home/varun/images/file_cnt");
        myfile2<<fl_cnt;
        myfile2.close();
        new_fl=format("mkdir /home/varun/images/sub%d",fl_cnt);
        system(new_fl.c_str());


    int ctr_fr=0;
    while (ctr_fr<10)
    {
    cap >> frame;
    // Clone the current frame:
    Mat original = frame.clone();
    // Convert the current frame to grayscale:
    Mat gray;
    cvtColor(original, gray, CV_BGR2GRAY);
    haar_cascade.detectMultiScale(gray, faces);
    if (faces.size()!=0)
    ctr_fr++;
    else continue;
    sort(faces.begin(),faces.end(),comp_func);
    face_i = faces[0];
    face_i.x=face_i.x+0.15*face_i.width;
    face_i.width=0.7*face_i.width;
    face = gray(face_i);
    cv::resize(face, face_resized, Size(64,64), 1.0, 1.0, INTER_CUBIC);
    file_nm=format("/home/varun/images/sub%d/image-%02i.pgm",fl_cnt,ctr_fr);
    rectangle(original, face_i, CV_RGB(0, 255,0), 1);
    imshow("user_add", original);
    char key = (char) waitKey(20);
    imwrite(file_nm,face_resized);

    }

    destroyWindow("user_add");
    cout<<"Please enter the username: ";
    cin>>name;
    label_file=format("/home/varun/images/sub%d_label.txt",fl_cnt);
    myfile3.open(name_list.c_str(),std::fstream::app | std::fstream::out | std::fstream::in);
    myfile3<<"\n";
    myfile3<<name;
    //cout<<endl<<name<<endl;
    myfile3.close();
    myfile2.open(label_file.c_str());
    myfile2<<fl_cnt;
    myfile2.close();
    system("python mik_dir.py /home/varun/images");
    }
//cout<<"mark 0"<<endl;
    vector<string> list_names;
    ifstream myfile;
    string name_list="/home/varun/images/name_list.txt";
    myfile.open(name_list.c_str());
    string line_name;
    while(getline(myfile,line_name))
    {
        list_names.push_back(line_name);
    }
//cout<<"comes here?"<<endl;


//cout<<"mark 1"<<endl;
try {
        read_csv(fn_csv, images, labels);
    } catch (cv::Exception& e) {
        cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
        // nothing more we can do
        exit(1);
    }

//cout<<"or the real bit"<<endl;
    int im_width = images[0].cols;
    int im_height = images[0].rows;
    Ptr<FaceRecognizer> model = createFisherFaceRecognizer();
    model->train(images, labels);

    for(;;) {
        cap >> frame;
        // Clone the current frame:
        Mat original = frame.clone();
        // Convert the current frame to grayscale:
        Mat gray;
        Mat gray_inv;
        cvtColor(original, gray, CV_BGR2GRAY);
        flip(gray,gray_inv,1);
        //imshow("inverted", gray_inv);
        // Find the faces in the frame:


        if (var==false || fr_ctr%6==0)
        {


            faces.clear();
            haar_cascade.detectMultiScale(gray, faces);
            //std::cout<<"number of faces found: "<<faces.size()<<endl;
            if (faces.size()==0)
            {
                haar_cascade_profile.detectMultiScale(gray,faces);
                if (faces.size()!=0)
                    cout<<"profile face dir 1"<<endl;
                else
                {
                    haar_cascade_profile.detectMultiScale(gray_inv,faces);
                    if (faces.size()!=0)
                        cout<<"profile face dir 2"<<endl;


                }
            }
            if (faces.size()!=0) var=true;
            // At this point you have the position of the faces in
            // faces. Now we'll get the faces, make a prediction and
            // annotate it in the video. Cool or what?
            sort(faces.begin(),faces.end(),comp_func);
            //if(faces.size()>=2)
            //cout<<faces[0].width<<":"<<faces[1].width<<endl;

            int sz_fc=faces.size();
            for(int i = 0; i < min(sz_fc,2); i++) {
                // Process face by face:
                //face_i = faces[i];
                faces[i].x=faces[i].x+0.15*faces[i].width;
                faces[i].width=0.7*faces[i].width;
                face_i = faces[i];
                // Crop the face from the image. So simple with OpenCV C++:
                face = gray(face_i);
                //cout<<faces[i].width<<":"<<faces[i].height<<endl;
                // Resizing the face is necessary for Eigenfaces and Fisherfaces. You can easily
                // verify this, by reading through the face recognition tutorial coming with OpenCV.
                // Resizing IS NOT NEEDED for Local Binary Patterns Histograms, so preparing the
                // input data really depends on the algorithm used.
                //
                // I strongly encourage you to play around with the algorithms. See which work best
                // in your scenario, LBPH should always be a contender for robust face recognition.
                //
                // Since I am showing the Fisherfaces algorithm here, I also show how to resize the
                // face you have just found:

                cv::resize(face, face_resized, Size(im_width, im_height), 1.0, 1.0, INTER_CUBIC);
                // Now perform the prediction, see how easy that is:
                prediction = model->predict(face_resized);
                // And finally write all we've found out to the original image!
                // First of all draw a green rectangle around the detected face:


                // Create the text we will annotate the box with:

                //std::vector<Rect> smile;
                if(i==0)
                    eyes= &eyes1;
                else
                    eyes=&eyes2;

                haar_cascade_eyes.detectMultiScale(face, *eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(30, 30));

                ushort nbins = segmentation * segmentation * 59;
                float *histogram=new float[nbins];
                memset(histogram, 0, nbins * sizeof(float));
                histogram=get_histogram(face);
                cv::Mat test_mat = cv::Mat::zeros(1, nbins, CV_32FC1);
                for(uint k=0; k<nbins; k++)
                {
                    test_mat.at<float>(0,k)=histogram[k];
                }
                delete histogram;
                cv::Mat test_labels_mat= cv::Mat::zeros(1,1,CV_32FC1);
                svm.predict(test_mat,test_labels_mat);
                int lab=int(test_labels_mat.at<float>(0,0));
                //haar_cascade_smile.detectMultiScale(face, smile, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30,30));


                smile_dq.pop_front();
                smile_dq.push_back(lab);
                int total=0;
                for (unsigned l=0; l<smile_dq.size(); l++)
                    total += smile_dq.at(l);



                if(total>=3)
                {
                    //cout<<"smile detected "<<endl;
                    smile_lab=1;
                }
                else
                smile_lab=0;

//cout<<"mark 2"<<endl;
                box_text[i] = format("Prediction = %s, smile=%d", list_names[prediction-1].c_str(),smile_lab);
                // Calculate the position for annotated text (make sure we don't
                // put illegal values in there):
                pos_x[i] = std::max(face_i.tl().x - 10, 0);
                pos_y[i] = std::max(face_i.tl().y - 10, 0);
                // And now put it into the image:

            }
        }
        //cout<<"+ "<<faces.size()<<endl;
        int sz_fc=faces.size();
        for (int i=0; i< min(2,sz_fc); i++)
        {

            if (i==0)
                eyes=&eyes1;
            else
                eyes=&eyes2;


            int eye_sz=(*eyes).size();
            for( int j = 0; j < min(eye_sz,2); j++ )
            {
                Point center( faces[i].x + (*eyes)[j].x + (*eyes)[j].width*0.5, faces[i].y + (*eyes)[j].y + (*eyes)[j].height*0.5 );
                int radius = cvRound( ((*eyes)[j].width + (*eyes)[j].height)*0.25 );
                if(center.x>(faces[i].x+faces[i].width)||center.y>(faces[i].y+faces[i].height))
                    continue;
                circle( original, center, radius, Scalar( 255, 0, 0 ), 4, 8, 0 );
            }
            face_i = faces[i];
            rectangle(original, face_i, CV_RGB(0, 255,0), 1);
            putText(original, box_text[i], Point(pos_x[i], pos_y[i]), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
        }


        // Show the result:
        //cout<<"*"<<endl;
        imshow("face_recognizer", original);
        fr_ctr++;

        // And display it:
        char key = (char) waitKey(20);
        // Exit this loop on escape:
        if(key == 27)
            break;
    }
    return 0;
}
