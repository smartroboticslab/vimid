/*
 * SPDX-FileCopyrightText: 2023 Smart Robotics Lab, Imperial College London
 * SPDX-FileCopyrightText: 2023 Dimos Tzoumanikas
 * SPDX-FileCopyrightText: 2023 Binbin Xu
 * SPDX-FileCopyrightText: 2023 Yifei Ren
 * SPDX-License-Identifier: BSD-3-Clause
*/


#include <dataLoader.h>
#include <sys/stat.h>
#include <boost/filesystem.hpp>


/// \brief Data loader constructor without IMU
dataLoader::dataLoader(const std::string path){
    path_ = path;
    
    std::string rgbPath = path + "/cam0";
    std::string depthPath = path + "/depth0";
    initRGBDLoader(rgbPath, depthPath);
    checkNextRGBDFrame();

    std::string gtFile = path + "/vicon_data.csv";
    initGTLoader(gtFile);
    loadNextGTPose();

    std::string maskPath = path + "/mask_RCNN";
    initMaskLoader(maskPath);
}

/// \brief Data loader constructor with IMU
dataLoader::dataLoader(Configuration *c){
    path_ = c->okvis_dataset_path;

    // Depth offset
    // offset_ = 16766000;       // used for book gray
    // offset_ = 16707000;       // used for book rgb
    // offset_ = -59000;         // used for room
    // offset_ = -57088;         // used for outside chair rgb
    // offset_ = -48333568;      // used for outside chair gray
    offset_ = 0;
    // offset_ = -56832;         // used for outside chair 2 rgb

    std::string associateFile = path_ + "/associate.txt";
    rgbdAssociateCsv_ = new std::ifstream(associateFile);
    if (!rgbdAssociateCsv_->good()){
      std::cerr << "DataLoader Error: RGB-D associate txt file does not exists!" << std::endl;
      std::cout << "Start to use RGB-D alignment reader." << std::endl;
      std::string rgbPath = path_ + "/cam0";
      std::string depthPath = path_ + "/depth0";
      initRGBDLoader(rgbPath, depthPath);
      checkNextRGBDFrame();
      useAssociate_ = false;
    } else {
      index_ = -1;
      useAssociate_ = true;
    }

    std::string imuConfigFile = c->okvis_config_file;
    initImuLoader(imuConfigFile);

    // std::string gtFile = path_ + "/cam0_gt.visim";
    std::string gtFile = path_ + "/vicon_data.csv";
    initGTLoader(gtFile);
    loadNextGTPose();

    std::string maskPath = path_ + "/mask_RCNN";
    initMaskLoader(maskPath);
}

/// \brief Data loader destructor
dataLoader::~dataLoader(){
    if (useAssociate_){
        rgbdAssociateCsv_->close();
    } else {
        rgbdCsv_->close();
    }
    gtCsv_->close();
}

/// \brief Initialize RGB data loader
bool dataLoader::initRGBDLoader(const std::string rgbPath, const std::string depthPath){
    if (!directory_exists(rgbPath)){
        std::cerr << "DataLoader ERROR: RGB path does not exists!" << std::endl;
        return false;
    }
    if (!directory_exists(depthPath)){
        std::cerr << "DataLoader ERROR: Depth path does not exists!" << std::endl;
        return false;
    }

    std::string csvName = rgbPath + "/data.csv";
    rgbdCsv_ = new std::ifstream(csvName);
    if (!rgbdCsv_->good()){
        std::cerr << "DataLoader Error: RGB-D csv file does not exists!" << std::endl;
        return false;
    }

    return true;
}

/// \brief Initialize IMU data loader
bool dataLoader::initImuLoader(std::string imuConfigFile){
    vio_parameters_reader_ = okvis::VioParametersReader(imuConfigFile);
    vio_parameters_reader_.getParameters(parameters_);

    // Open the IMU file
    imu_file_ = std::ifstream(path_ + "/imu0/data.csv");
    if (!imu_file_.good()){
        std::cout << "no imu file found at " << path_+"/imu0/data.csv" << std::endl;
        return false;
    }
    start_ = okvis::Time(0.0);
    okvis::Time t_imu = start_;
    int number_of_lines = 0;
    std::getline(imu_file_, line_); // set reading position to the second line
    imuMeasurementsPtr_.reset(new okvis::threadsafe::ThreadSafeQueue<okvis::ImuMeasurement>);
    while (std::getline(imu_file_, line_)){
        ++number_of_lines;

        std::stringstream stream(line_);
        std::string s;
        std::getline(stream, s, ',');
        std::string nanoseconds = s.substr(s.size() - 9, 9);
        std::string seconds = s.substr(0, s.size() - 9);

        Eigen::Vector3d gyr;
        for (int j = 0; j < 3; ++j) {
            std::getline(stream, s, ',');
            gyr[j] = std::stof(s);
        }

        Eigen::Vector3d acc;
        for (int j = 0; j < 3; ++j) {
            std::getline(stream, s, ',');
            acc[j] = std::stof(s);
        }
        t_imu = okvis::Time(std::stoi(seconds), std::stoi(nanoseconds));

        // Create an imu measurement
        okvis::ImuMeasurement imu_measurement;
        imu_measurement.measurement.accelerometers = acc;
        imu_measurement.measurement.gyroscopes = gyr;
        imu_measurement.timeStamp = t_imu;

        // Add to the queue
        imuMeasurementsPtr_->PushBlockingIfFull(imu_measurement, 120000);
    }
        
    std::cout << "No. IMU measurements: " << number_of_lines << std::endl;
    if (number_of_lines <= 0) {
        std::cout << "no imu messages present in " << path_+"/imu0/data.csv" << std::endl;
        return false;
    }

    std::vector<okvis::Time> times;
    okvis::Time latest(0);
    int num_camera_images = 0;
    std::string folder(path_ + "/cam0/data");

    // Obtain all image names
    std::ifstream img_name_file(path_ + "/cam0/data.csv");
    std::string line;
    std::getline(img_name_file, line);
    while (std::getline(img_name_file, line)) {
        std::stringstream stream(line);
        std::string s;
        std::getline(stream, s, ',');
        std::getline(stream, s, ',');
        image_names_.push_back(s);
        num_camera_images++;
    }
    img_name_file.close();

    if (num_camera_images == 0) {
        std::cout << "no images at " << folder;
        return false;
    }
    std::cout << "No. cam0 images: " << num_camera_images << std::endl;

    cam_iterators_ = image_names_.begin();
    counter_ = 0;
    start_ = okvis::Time(0.0);

    return true;
}

/// \brief Initialize ground truth pose loader
bool dataLoader::initGTLoader(const std::string gtFile){
    gtCsv_ = new std::ifstream(gtFile);
    if (!gtCsv_->good()){
        std::cerr << "DataLoader Error: Ground Truth file does not exists!" << std::endl;
        return false;
    }
    return true;
}

/// \brief Initialize mask data loader
bool dataLoader::initMaskLoader(const std::string maskPath){
    return true;
}

/// \brief Check if the next RGB & Depth frame are valid
bool dataLoader::checkNextRGBDFrame(){
    std::string line;
    std::string _sub;

    // If do not use the associate file
    if (!useAssociate_){
        getline(*rgbdCsv_, line);

        // check if the EOF is reached
        if (!rgbdCsv_->good()){
          std::cout << "DataLoader Ending ..." << std::endl;
          return false;
        }
        std::stringstream temp(line);
        std::string _sub;

        getline(temp, _sub, ',');
        if (_sub == "#timestamp [ns]"){
            index_ = -1;    // set the first frame index
        } else {
            index_++;
            std::cout << "DataLoader Processing: Loading " << _sub << ".png ..." << std::endl;
            std::cout << "Current index: " << index_ << std::endl;
            rgbFramePath_ = path_ + "/cam0/data/" + _sub + ".png";    // current RGB frame path

            long int li = std::stol(_sub) + offset_;
            std::stringstream ss;
            ss << li;
            ss >> _sub;

            depthFramePath_ = path_ + "/depth0/data/" + _sub + ".png"; // current Depth frame path
            time_ = std::stol(_sub);
        }
    }
    // If use the associate file
    else {
        std::string line;
        getline(*rgbdAssociateCsv_, line);

        // check if the EOF is reached
        if (!rgbdAssociateCsv_->good()){
          std::cout << "DataLoader Ending ..." << std::endl;
          return false;
        }

        std::stringstream temp(line);
        getline(temp, _sub, ' ');
        getline(temp, _sub, ' ');

        index_++;
        std::cout << "DataLoader Processing: Loading RGB " << _sub << std::endl;
        std::cout << "Current index: " << index_ << std::endl;
        rgbFramePath_ = path_ + "/cam0/data/" + _sub;    // current RGB frame path

        std::stringstream ttmep(_sub);
        getline(ttmep, _sub, ',');
        time_ = std::stol(_sub);
        std::cout << time_ << std::endl;

        getline(temp, _sub, ' ');
        getline(temp, _sub);
        std::cout << "DataLoader Processing: Loading Depth " << _sub << std::endl;
        depthFramePath_ = path_ + "/depth0/data/" + _sub; // current Depth frame path
    }
    return true;
}

/// \brief Load next RGB frame
cv::Mat dataLoader::loadNextRGBFrame(){
    cv::Mat rgbFrame = cv::imread(rgbFramePath_);
    return rgbFrame;
}

/// \brief Load next Depth frame
cv::Mat dataLoader::loadNextDepthFrame(){
    cv::Mat depthFrame = cv::imread(depthFramePath_, cv::IMREAD_GRAYSCALE | CV_LOAD_IMAGE_ANYDEPTH);
    depthFrame.convertTo(depthFrame, CV_16UC1, 1.0f);
    return depthFrame;
}

/// \brief Load IMU measurements before next frame
bool dataLoader::loadImuMeasurements(){
    // check if at the end
    if (cam_iterators_ == image_names_.end()){
        std::cout << std::endl << "Finished. Press any key to exit." << std::endl << std::flush;
        return true;
    }

    // obtain image timestamps
    assert(cam_iterators_->size() > 13);
    std::string nanoseconds = cam_iterators_->substr(cam_iterators_->size() - 13, 9);
    std::string seconds = cam_iterators_->substr(0, cam_iterators_->size() - 13);
    viTimeStamp_ = okvis::Time(std::stoi(seconds), std::stoi(nanoseconds));
    std::cout << "Current Time: " << viTimeStamp_ << std::endl;
    if (start_ == okvis::Time(0.0)) {
        start_ = viTimeStamp_;
    }

    // Record the new viTimeStamp
    prevTimeStamp_ = currTimeStamp_;
    currTimeStamp_ = viTimeStamp_;

    okvis::Duration imu_overlap_dur(0.02);
    okvis::Time imu_overlap_time(0, 20000000);
    okvis::Time imuBeginTime(0, 0);
    if (prevTimeStamp_ >= imu_overlap_time) {
        imuBeginTime = prevTimeStamp_ - imu_overlap_dur;
    }
    okvis::Time imuEndTime;
    if(processFirst_) {
        // on the first run through, IMU measurements are only used to estimate gravity vector
        // (therefore, only want measurements up to the current time)
        imuEndTime = currTimeStamp_;
    }
    else {
        imuEndTime = currTimeStamp_ + imu_overlap_dur;
    }

    // Add all new IMU measurements to the deque
    okvis::ImuMeasurement data;
    bool waitForImu = true;
    while(waitForImu) {
        if(imuMeasurementsPtr_->PopNonBlocking(&data)) {
            imuMeasurements_.push_back(data);
            if(data.timeStamp > imuEndTime) {
                waitForImu = false;
            }
        }
    }

    // Create a transformation deque (imuData) of times between previous viTimeStamp and current viTimeStamp
    okvis::ImuMeasurementDeque::iterator first_msg = imuMeasurements_.begin();
    okvis::ImuMeasurementDeque::iterator last_msg = imuMeasurements_.end();
    --last_msg; // last_msg actually refers to the past-the-end element of the deque... we want the actual end

    // Move the first_msg iterator to the first message within the time window
    if(!processFirst_) { // want to include the very first measurement
        while(first_msg->timeStamp <= imuBeginTime && first_msg != imuMeasurements_.end()) {
            ++first_msg;
        }
    }

    // Move the last_msg iterator to the last message within the time window
    while(last_msg->timeStamp >= imuEndTime && last_msg != imuMeasurements_.begin() && last_msg > first_msg) {
        --last_msg;
    }

    if(last_msg != imuMeasurements_.end()) {
        ++last_msg;
    }

    imuData_ = okvis::ImuMeasurementDeque(first_msg, last_msg);

    cam_iterators_++;
    counter_++;

    if(processFirst_){
        processFirst_ = false;
    }

    return false;
}

/// \brief // Delete from main deque all imu measurements before the previous viTimeStamp with 0.02s buffer
void dataLoader::deleteImuMeasurements(){
    static const okvis::Time overlap(0, 20000000);
    static const okvis::Time zero(0, 0);
    okvis::Duration eraseUntil;
    if (prevTimeStamp_ > overlap) {
        eraseUntil = prevTimeStamp_ - overlap;
    }
    auto eraseEnd = imuMeasurements_.begin();
    while (eraseEnd->timeStamp - zero < eraseUntil) {
        ++eraseEnd;
    }
    if (eraseEnd != imuMeasurements_.begin()) {
        --eraseEnd; // go back to the previous position so < prevTimeStamp - overlap
    }
    imuMeasurements_.erase(imuMeasurements_.begin(), eraseEnd);
}

/// \brief Load next Ground Truth pose
Matrix4 dataLoader::loadNextGTPose(){
    Matrix4 pose;
    std::string line;
    getline(*gtCsv_, line);

    // check if is the first line
    if (isFirst_){
        isFirst_ = false;
        return pose;
    }

    // check if the EOF is reached
    if (!gtCsv_->good()){
        std::cout << "DataLoader Ending ..." << std::endl;
        return pose;
    }

    std::stringstream temp(line);
    std::string _sub;
    std::vector<double> _subArray;
    while (getline(temp, _sub, ',')){
        _subArray.push_back(std::stod(_sub));
    }

    // save translation and rotation
    // std::cout << "v: " << _subArray[1] << " " << _subArray[2] << " " << _subArray[3] << std::endl;
    // std::cout << "q: " << _subArray[4] << " " << _subArray[5] << " " << _subArray[6] << " " << _subArray[7] << std::endl;
    Eigen::Quaterniond q(_subArray[4], _subArray[5], _subArray[6], _subArray[7]);
    Eigen::Matrix3d C = q.toRotationMatrix();
    for (int i = 0; i < 3; i++){
        pose.data[i].x = C(i, 0);
        pose.data[i].y = C(i, 1);
        pose.data[i].z = C(i, 2);
        pose.data[i].w = _subArray[i+1];
    }
    pose.data[3].x = 0.0;
    pose.data[3].y = 0.0;
    pose.data[3].z = 0.0;
    pose.data[3].w = 1.0;
    // float4 quat = make_float4(_subArray[4], _subArray[5], _subArray[6], _subArray[7]);
    // float3 trans = make_float3(_subArray[1], _subArray[2], _subArray[3]);
    // pose = toMatrix4(quat, trans);
    return pose;
}

/// \brief Check if the file exists
bool dataLoader::file_exists(const std::string &path){
    std::ifstream file(path);
    return file.good();
}

/// \brief Check if the directory exists
bool dataLoader::directory_exists(const std::string &path){
#if defined(_WIN32)
    DWORD attributes = GetFileAttributes(path.c_str());
    return (attributes != INVALID_FILE_ATTRIBUTES &&
            (attributes & FILE_ATTRIBUTE_DIRECTORY));
#else
    struct stat sb;
    return stat(path.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode);
#endif
}
