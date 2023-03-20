#ifndef ZR300READER_H
#define ZR300READER_H

#include <librealsense/rs.hpp>
#include "librealsense/rsutil.h"
#include <deque>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <fstream>
#include <memory>
#include <chrono>
#include <thread>
#include <queue>
#include <mutex>
#include <atomic>
#include <iomanip>
#include <string>
#include <condition_variable>

#include <okvis/VioParametersReader.hpp>
#include <okvis/VioInterface.hpp>
#include <boost/filesystem.hpp>

namespace okvis {

enum RSStreamType {ALL = 0, RGB_DEPTH, IRSTEREO};

template <class T>
class ThreadMutexObject
{
public:
    ThreadMutexObject()
    {}

    void assign(T newValue)
    {
        mutex.lock();

        object = lastCopy = newValue;

        mutex.unlock();
    }

    T getValue()
    {
        mutex.lock();

        lastCopy = object;

        mutex.unlock();

        return lastCopy;
    }

    void operator++(int)
    {
        mutex.lock();

        object++;

        mutex.unlock();
    }

private:
    T object;
    T lastCopy;
    std::mutex mutex;
};

template<class GET_DEPTH, class TRANSFER_PIXEL> void align_images(const rs_intrinsics & depth_intrin, const rs_extrinsics & depth_to_other, const rs_intrinsics & other_intrin, GET_DEPTH get_depth, TRANSFER_PIXEL transfer_pixel);

void align_depth_to_other(uint8_t * z_aligned_to_other, const uint16_t * z_pixels, float z_scale, const rs_intrinsics & z_intrin, const rs_extrinsics & z_to_other, const rs_intrinsics & other_intrin);

struct IMUInfo{
    IMUInfo(){
        latestIMUIndex.assign(-1);
    }
    void update(Eigen::Vector3d& a, Eigen::Vector3d& g, double ts){
        acc = a;
        gyr = g;
        timestamp = (uint64_t)(ts * 1e6);
        init = true;
        latestIMUIndex++;
    }
    uint64_t getSec(){
        return timestamp / 1000000000ULL;
    }
    uint64_t getNSec(){
        return timestamp - (getSec() * 1000000000ULL);
    }
    okvis::Time getOkvisTimestamp(){
        return okvis::Time(getSec(),getNSec());
    }
    Eigen::Vector3d gyr;
    Eigen::Vector3d acc;
    uint64_t timestamp;
    ThreadMutexObject<long int> latestIMUIndex;
    bool init = false;
};

struct RSFrameInfo{
    uint64_t timestamp = 0;
    int height = 0;
    int width = 0;
    bool init = false;
    double timeShift_from_cam_imu = 0; // in [ms]
    cv::Mat img;
    ThreadMutexObject<long int> frameIndex;
    RSFrameInfo(){
        frameIndex.assign(-1);
    }
    uint64_t getSec(){
        return timestamp / 1000000000ULL;
    }
    uint64_t getNSec(){
        return timestamp - (getSec() * 1000000000ULL);
    }
    okvis::Time getOkvisTimestamp(){
        return okvis::Time(getSec(),getNSec());
    }
};

struct RGBInfo:RSFrameInfo{

    void initialize(int w, int h){
        height = h;
        width = w;
        img.create(height, width, CV_8UC3);
    }
    void update(uchar * im, double ts)
    {
        memcpy(img.data, im, width * height * 3);
        timestamp = (uint64_t)((ts+timeShift_from_cam_imu) * 1e6);
        init = true;
    }
};

struct IRInfo:RSFrameInfo{

    void initialize(int w, int h){
        height = h;
        width = w;
        img.create(height, width, CV_8UC1);
    }
    void update(uchar * im, double ts)
    {
        memcpy(img.data, im, width * height * 1);
        timestamp = (uint64_t)((ts+timeShift_from_cam_imu) * 1e6);
        init = true;
    }
};

struct FishEyeInfo:RSFrameInfo{

    void initialize(int w, int h){
        height = h;
        width = w;
        img.create(height, width, CV_8UC1);
    }
    void update(uchar * im, double ts)
    {
        memcpy(img.data, im, width * height * 1);
        timestamp = (uint64_t)(ts * 1e6);
        init = true;
    }
};

struct DepthInfo:RSFrameInfo{

    void initialize(int w, int h){
        height = h;
        width = w;
        img.create(height, width, CV_16UC1);
    }
    void update(uint8_t * im, double ts)
    {
        memcpy(img.data, im, width * height * 2);
        img.convertTo(img, CV_16UC1, 5.0); // TUM format
        timestamp = (uint64_t)((ts + timeShift_from_cam_imu) * 1e6);
        init = true;
    }
};

class ZR300Reader
{
public:
    ZR300Reader();
    ZR300Reader(uint16_t w = 640,int16_t h = 480,int8_t f = 60,okvis::VioInterface* ve = NULL,
                RSStreamType st = RSStreamType::RGB_DEPTH,
                bool toRecData = false, std::string folder = "dataset");

    ZR300Reader(uint16_t,int16_t,int8_t,std::string);
    ~ZR300Reader();
    IMUInfo imu;
    RGBInfo rgb;
    DepthInfo depth;
    FishEyeInfo fisheye;
    IRInfo ir1;
    IRInfo ir2;

    void initRS();
    void initRecData();

    void stopRecData();
    void stopDevice();
    void enabledebug(){isDebug = true;}
    void printCamsInfo();
    void setTimeShift(double);
    void getColorIntrins(float& fx, float& fy, float& cx, float& cy);

    RSStreamType streamType = RSStreamType::RGB_DEPTH;
    bool readDataset(std::string,okvis::VioParameters);
private:

    okvis::VioInterface* okvis_estimator = NULL;

    std::deque<std::vector<rs::motion_data>> motionData;
    rs::context *ctx;
    rs::device  *dev;
    std::vector<uint16_t> supported_streams;
    std::mutex imu_mutex,rgb_mutex,fisheye_mutex,depth_mutex;

    uint16_t imgWidth;
    uint16_t imgHeight;
    uint8_t fps;

    rs::intrinsics depth_intrin;
    rs::extrinsics depth_to_color;
    rs::intrinsics color_intrin;
    float scale;

    bool recData;
    bool useOkvis;
    bool isDebug;

    bool isRecData;

    std::string output_path;
    std::ofstream *depthCsv;
    std::ofstream *rgbCsv;
    std::ofstream *ir2Csv;
    std::ofstream *ir1Csv;
    std::ofstream *fisheyeCsv;
    std::ofstream *imuCsv;
    std::ofstream *imuCsvCalibBegin;
    std::ofstream *imuCsvCalibEnd;

    void runImuCallback(rs::motion_data&);
    void runMCtimestampCallback(rs::timestamp_data&){}

    void runRGBCallback(rs::frame&);
    void runDepthCallback(rs::frame&);
    void runFishEyeCallback(rs::frame&);
    void runIRLeftCallback(rs::frame&);
    void runIRRightCallback(rs::frame&);

    // file helpers
    bool file_exists(const std::string &path);
    bool directory_exists(const std::string &path);
    void make_directory(const std::string &path);

};

}

#endif // ZR300READER_H
