/*
 * SPDX-FileCopyrightText: 2023 Smart Robotics Lab, Imperial College London
 * SPDX-FileCopyrightText: 2023 Dimos Tzoumanikas
 * SPDX-License-Identifier: BSD-3-Clause
*/

#include <dataRecorder.h>
#include <sys/stat.h>


/// \brief Data recorder constructor
dataRecorder::dataRecorder(const std::string path, const bool use_imu)
    :output_path(path){
    if (isRecData){
        initRecData(use_imu);
    }
}

//// \brief Data recorder destructor
dataRecorder::~dataRecorder(){
    stopRecData();
}

/// \brief Initialize data recorder
void dataRecorder::initRecData(bool use_imu){
    // check if the directory exists
    if (!directory_exists(output_path)){
        make_directory(output_path);
    }

    if (use_imu){
        std::string camCsvName = output_path + "VI-MID";
        camCsv = new std::ofstream(camCsvName.c_str(), std::ofstream::out);
        *camCsv << "#timestamp [ns], position.x [m], position.y [m], position.z [m], quaternion.x [], quaternion.y [], quaternion.z [], quaternion.w []\n";
    } else {
        std::string camCsvName = output_path + "MID-Fusion";
        camCsv = new std::ofstream(camCsvName.c_str(), std::ofstream::out);
        *camCsv << "#timestamp [ns], position.x [m], position.y [m], position.z [m], quaternion.x [], quaternion.y [], quaternion.z [], quaternion.w []\n";
    }

    // std::string insCsvName = output_path + "instance.txt";
    // insCsv = new std::ofstream(insCsvName.c_str(), std::ofstream::out);
    // *insCsv << "#timestamp [ns], position.x [m], position.y [m], position.z [m], quaternion.w [], quaternion.x [], quaternion.y [], quaternion.z []\n";

    std::this_thread::sleep_for(std::chrono::milliseconds(300));
}

/// \brief Stop recording data and close CSV file
void dataRecorder::stopRecData(){
    isRecData = false;
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    camCsv->close();
    insCsv->close();

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

/// \brief Record the pose of the camera under OKVIS
void dataRecorder::recCamPose(const okvis::Time &timestamp, const okvis::kinematics::Transformation &T_WC){
    Eigen::Vector3d r = T_WC.r();
    Eigen::Matrix3d C = T_WC.C();
    Eigen::Quaterniond q(C);
    q.normalized();
    // *camCsv << timestamp << "," << r[0] << "," << r[1] << "," << r[2] << "," << q.w() << "," << q.x() << "," << q.y() << "," << q.z() << std::endl;
    *camCsv << timestamp << " " << r[0] << " " << r[1] << " " << r[2] << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
}

/// \brief Record the pose of the camera under using long int
void dataRecorder::recCamPose(const long int &timestamp, const okvis::kinematics::Transformation &T_WC){
  Eigen::Vector3d r = T_WC.r();
  Eigen::Matrix3d C = T_WC.C();
  Eigen::Quaterniond q(C);
  q.normalized();
  *camCsv << timestamp << " " << r[0] << " " << r[1] << " " << r[2] << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
}


/// \brief Check if the file exists
bool dataRecorder::file_exists(const std::string &path) {
    std::ifstream file(path);
    return file.good();
}

/// \brief Check if the directory exists
bool dataRecorder::directory_exists(const std::string &path) {
#if defined(_WIN32)
    DWORD attributes = GetFileAttributes(path.c_str());
    return (attributes != INVALID_FILE_ATTRIBUTES &&
            (attributes & FILE_ATTRIBUTE_DIRECTORY));
#else
    struct stat sb;
    return stat(path.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode);
#endif
}

/// \brief Make a new directory
void dataRecorder::make_directory(const std::string &path) {
#if defined(_WIN32)
    _mkdir(path.c_str());
#else
    mkdir(path.c_str(), 0755);
#endif
}