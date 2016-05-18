#include <cv.h>
#include <highgui.h>
#include <time.h>

#define RED 2
#define GREEN 1
#define BLUE 0

using namespace cv;

int xGradient(Mat image, int x, int y)
{
    return image.at<uchar>(y-1, x-1) +
                2*image.at<uchar>(y, x-1) +
                 image.at<uchar>(y+1, x-1) -
                  image.at<uchar>(y-1, x+1) -
                   2*image.at<uchar>(y, x+1) -
                    image.at<uchar>(y+1, x+1);
}


int yGradient(Mat image, int x, int y)
{
    return image.at<uchar>(y-1, x-1) +
                2*image.at<uchar>(y-1, x) +
                 image.at<uchar>(y-1, x+1) -
                  image.at<uchar>(y+1, x-1) -
                   2*image.at<uchar>(y+1, x) -
                    image.at<uchar>(y+1, x+1);
}

Mat sobelFilter_Secuencial(Mat& src){
    int gx, gy, sum;
    Mat dst = src.clone();
    for(int y = 0; y < src.rows; y++)
            for(int x = 0; x < src.cols; x++)
                dst.at<uchar>(y,x) = 0.0;
    for(int y = 1; y < src.rows - 1; y++){
            for(int x = 1; x < src.cols - 1; x++){
                gx = xGradient(src, x, y);
                gy = yGradient(src, x, y);
                sum = abs(gx) + abs(gy);
                sum = sum > 255 ? 255:sum;
                sum = sum < 0 ? 0 : sum;
                dst.at<uchar>(y,x) = sum;
            }
        }
    return dst;

}



void img2gray(unsigned char *imageInput, int width, int height, unsigned char *imageOutput){
    for(int row=0; row < height; row++){
        for(int col=0; col < width; col++){
            imageOutput[row*width+col] = imageInput[(row*width+col)*3+RED]*0.299 + imageInput[(row*width+col)*3+GREEN]*0.587 \
                                     + imageInput[(row*width+col)*3+BLUE]*0.114;
        }
    }
}

int main(int argc, char **argv){

    char* imageName = argv[1];
    unsigned char *dataRawImage, *d_dataRawImage, *d_imageOutput, *h_imageOutput, *d_sobelOutput, *h_sobelOutput, *h_io;
    Mat image;
    image = imread(imageName, 1);
    clock_t start, end;
    double cpu_time_used;


    if(argc !=2 || !image.data){
        printf("No image Data \n");
        return -1;
    }

    Size s = image.size();

    int width = s.width;
    int height = s.height;
    int size = sizeof(unsigned char)*width*height*image.channels();
    int sizeGray = sizeof(unsigned char)*width*height;

    dataRawImage = (unsigned char*)malloc(size);
    
    dataRawImage = image.data;  
    h_imageOutput = (unsigned char *)malloc(sizeGray);
    img2gray(dataRawImage,width,height,h_imageOutput);
    Mat gray_image;
    gray_image.create(height,width,CV_8UC1);
    gray_image.data = h_imageOutput;

    start = clock();
    Mat prueba_secuencial = sobelFilter_Secuencial(gray_image);
    end = clock();
    cpu_time_used = ((double) (end - start)) /CLOCKS_PER_SEC;
    printf("%.10f\n",cpu_time_used);
    /*namedWindow(imageName, WINDOW_NORMAL);
    namedWindow("Gray Image Secuential", WINDOW_NORMAL);
    namedWindow("Sobel Image OpenCV Secuential", WINDOW_NORMAL);

    imshow(imageName,image);
    imshow("Gray Image Secuential", gray_image);
    imshow("Sobel Image OpenCV Secuential",prueba_secuencial);*/

    waitKey(0);
    return 0;
}