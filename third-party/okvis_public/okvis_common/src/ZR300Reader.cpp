#include <okvis/ZR300Reader.h>

#if defined(_WIN32)
#include "Windows.h"
#include <direct.h>
#else
#include <sys/stat.h>
#endif

namespace okvis {

template<class GET_DEPTH, class TRANSFER_PIXEL> void align_images(const rs_intrinsics & depth_intrin, const rs_extrinsics & depth_to_other, const rs_intrinsics & other_intrin, GET_DEPTH get_depth, TRANSFER_PIXEL transfer_pixel)
{
    // Iterate over the pixels of the depth image
    for(int depth_y = 0; depth_y < depth_intrin.height; ++depth_y)
    {
        int depth_pixel_index = depth_y * depth_intrin.width;
        for(int depth_x = 0; depth_x < depth_intrin.width; ++depth_x, ++depth_pixel_index)
        {
            // Skip over depth pixels with the value of zero, we have no depth data so we will not write anything into our aligned images
            if(float depth = get_depth(depth_pixel_index))
            {
                // Map the top-left corner of the depth pixel onto the other image
                float depth_pixel[2] = {depth_x-0.5f, depth_y-0.5f}, depth_point[3], other_point[3], other_pixel[2];
                rs_deproject_pixel_to_point(depth_point, &depth_intrin, depth_pixel, depth);
                rs_transform_point_to_point(other_point, &depth_to_other, depth_point);
                rs_project_point_to_pixel(other_pixel, &other_intrin, other_point);
                const int other_x0 = static_cast<int>(other_pixel[0] + 0.5f);
                const int other_y0 = static_cast<int>(other_pixel[1] + 0.5f);

                // Map the bottom-right corner of the depth pixel onto the other image
                depth_pixel[0] = depth_x+0.5f; depth_pixel[1] = depth_y+0.5f;
                rs_deproject_pixel_to_point(depth_point, &depth_intrin, depth_pixel, depth);
                rs_transform_point_to_point(other_point, &depth_to_other, depth_point);
                rs_project_point_to_pixel(other_pixel, &other_intrin, other_point);
                const int other_x1 = static_cast<int>(other_pixel[0] + 0.5f);
                const int other_y1 = static_cast<int>(other_pixel[1] + 0.5f);

                if(other_x0 < 0 || other_y0 < 0 || other_x1 >= other_intrin.width || other_y1 >= other_intrin.height) continue;

                // Transfer between the depth pixels and the pixels inside the rectangle on the other image
                for(int y=other_y0; y<=other_y1; ++y) for(int x=other_x0; x<=other_x1; ++x) transfer_pixel(depth_pixel_index, y * other_intrin.width + x);
            }
        }
    }
}

void align_depth_to_other(uint8_t * z_aligned_to_other, const uint16_t * z_pixels, float z_scale, const rs_intrinsics & z_intrin, const rs_extrinsics & z_to_other, const rs_intrinsics & other_intrin)
{
    auto out_z = (uint16_t*)z_aligned_to_other;
    okvis::align_images(z_intrin, z_to_other, other_intrin,
                 [z_pixels, z_scale](int z_pixel_index) { return z_scale * z_pixels[z_pixel_index]; },
    [out_z, z_pixels](int z_pixel_index, int other_pixel_index) { out_z[other_pixel_index] = out_z[other_pixel_index] ? std::min(out_z[other_pixel_index],z_pixels[z_pixel_index]) : z_pixels[z_pixel_index]; });
}

//ZR300Reader::ZR300Reader()
//    :imgWidth(640),imgHeight(480),fps(60),output_path("") {
//    initRS();
//}

ZR300Reader::ZR300Reader(uint16_t w,int16_t h,int8_t f,okvis::VioInterface* ve,
            RSStreamType stream, bool toRecData, std::string targetFolder)
    :imgWidth(w),imgHeight(h),fps(f),okvis_estimator(ve),streamType(stream),isRecData(toRecData),output_path(targetFolder) {

    useOkvis = false;
    isDebug = false;

    if(isRecData) {
        initRecData();
    }

    initRS();

    if(ve != NULL) {
        useOkvis = true;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
}

ZR300Reader::ZR300Reader(uint16_t w, int16_t h, int8_t f, std::string folder)
    :imgWidth(w),imgHeight(h),fps(f),output_path(folder){

    recData = false;
    useOkvis = false;
    isDebug = false;

    initRS();
    if(isRecData) {
        initRecData();
    }
}

ZR300Reader::~ZR300Reader(){
    stopDevice();
}

bool ZR300Reader::readDataset(std::string path, okvis::VioParameters parameters){

    okvis::Duration deltaT(0.0);
    const unsigned int numCameras = parameters.nCameraSystem.numCameras();

    // open the IMU file
    std::string line;
    std::ifstream imu_file(path + "/imu0/data.csv");
    if (!imu_file.good()) {
      std::cerr << "no imu file found at " << path+ "/imu0/data.csv";
      return false;
    }

    int number_of_lines = 0;
    while (std::getline(imu_file, line))
      ++number_of_lines;
    std::cerr << "No. IMU measurements: " << number_of_lines-1;
    if (number_of_lines - 1 <= 0) {
      std::cerr << "no imu messages present in " << path+"/imu0/data.csv";
      return false;
    }
    // set reading position to second line
    imu_file.clear();
    imu_file.seekg(0, std::ios::beg);
    std::getline(imu_file, line);

    int num_camera_images = 0;
    std::vector < std::vector < std::string >> image_names(numCameras);
    for (size_t i = 0; i < numCameras; ++i) {
      num_camera_images = 0;
      std::string folder(path + "/cam" + std::to_string(i) + "/data");

      for (auto it = boost::filesystem::directory_iterator(folder);
           it != boost::filesystem::directory_iterator(); it++) {
        if (!boost::filesystem::is_directory(it->path())) {  //we eliminate directories
          num_camera_images++;
          image_names.at(i).push_back(it->path().filename().string());
        } else {
          continue;
        }
      }

      if (num_camera_images == 0) {
//        LOG(ERROR) << "no images at " << folder;
        return false;
      }

//      LOG(INFO)<< "No. cam " << i << " images: " << num_camera_images;
      // the filenames are not going to be sorted. So do this here
      std::sort(image_names.at(i).begin(), image_names.at(i).end());
    }

    std::vector < std::vector < std::string > ::iterator > cam_iterators(numCameras);
    for (size_t i = 0; i < numCameras; ++i) {
      cam_iterators.at(i) = image_names.at(i).begin();
    }

    int counter = 0;
    okvis::Time start(0.0);

    while (true) {

      // check if at the end
      for (size_t i = 0; i < numCameras; ++i) {
        if (cam_iterators[i] == image_names[i].end()) {
          std::cout << std::endl << "Finished. Press any key to exit." << std::endl << std::flush;
          cv::waitKey();
          break;
        }
      }

      /// add images
      okvis::Time t;

      for (size_t i = 0; i < numCameras; ++i) {
        if(parameters.nCameraSystem.isVirtual(i)) {
          continue; // nothing to do...
        }

        cv::Mat filtered = cv::imread(
            path + "/cam" + std::to_string(i) + "/data/" + *cam_iterators.at(i),
            cv::IMREAD_GRAYSCALE);
        //      std::cout << "RGB: " << path + "/cam" + std::to_string(i) + "/data/" + *cam_iterators.at(i) << std::endl;
        cv::Mat depth;
        if(parameters.nCameraSystem.isDepthCamera(i)) {

//          if(dataType == OkvisDatasetType::Realsense){
            depth = cv::imread(
                path + "/cam" + std::to_string(parameters.nCameraSystem.virtualCameraIdx(i)) +
                    "/data/" + *cam_iterators.at(i), cv::IMREAD_GRAYSCALE | CV_LOAD_IMAGE_ANYDEPTH);
//          }
//          else if(dataType == OkvisDatasetType::VISim){
//            depth = cv::imread(
//                path + "/depth0/data/" + *cam_iterators.at(i), cv::IMREAD_GRAYSCALE | CV_LOAD_IMAGE_ANYDEPTH);
//            depth.convertTo(depth,CV_16UC1, 5.0f);
//          }
//          else{
//            std::cerr << "okvis Error: wrong dataset format!" << std::endl;
//            return false;
//          }
  //            std::cout << "depth: " << path + "/cam" + std::to_string(parameters.nCameraSystem.virtualCameraIdx(i)) +
  //                         "/data/" + *cam_iterators.at(i) << std::endl;
        }
        std::string nanoseconds = cam_iterators.at(i)->substr(
            cam_iterators.at(i)->size() - 13, 9);
        std::string seconds = cam_iterators.at(i)->substr(
            0, cam_iterators.at(i)->size() - 13);
        t = okvis::Time(std::stoi(seconds), std::stoi(nanoseconds));
        if (start == okvis::Time(0.0)) {
          start = t;
        }

        // get all IMU measurements till then
        okvis::Time t_imu = start;
        do {
          if (!std::getline(imu_file, line)) {
            std::cout << std::endl << "Finished. Press any key to exit." << std::endl << std::flush;
            cv::waitKey();
            break;
          }

          std::stringstream stream(line);
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

          // add the IMU measurement for (blocking) processing
          if (t_imu - start + okvis::Duration(1.0) > deltaT) {
            okvis_estimator->addImuMeasurement(t_imu, acc, gyr);
  //          std::cout << "adding IMU: time: " << std::stoi(seconds) << " s; " << std::stoi(nanoseconds) << " ns" << std::endl;
          }
        } while (t_imu <= t);

        // add the image to the frontend for (blocking) processing
        if (t - start > deltaT) {
          if(parameters.nCameraSystem.isDepthCamera(i)) {
  //            std::cout << "adding IMG: time: " << std::stoi(seconds) << " s; " << std::stoi(nanoseconds) << " ns" << std::endl;
            okvis_estimator->addImage(t, i, filtered, depth);
            okvis_estimator->addImage(t, parameters.nCameraSystem.virtualCameraIdx(i),
                                     cv::Mat::zeros(filtered.rows, filtered.cols, CV_8U));
  //          std::cout << "depth camera index: " << parameters.nCameraSystem.virtualCameraIdx(i) << std::endl;
          } else {
            okvis_estimator->addImage(t, i, filtered);
  //            std::cout << "RGB camera index  : " << i << std::endl;
          }
        }

//        if (!config->no_gui){
//          poseVis.display(T_WS_);
//          okvis_estimator->display();
//          drawthem(inputRGB, depthRender, trackRender, volumeRender,
//                   trackRender, kfusion->getComputationResolution());
//        }

        cam_iterators[i]++;
      }
      ++counter;
    }
}

void ZR300Reader::setTimeShift(double t){
    rgb.timeShift_from_cam_imu = t;
    depth.timeShift_from_cam_imu = t;
    ir1.timeShift_from_cam_imu = t;
    ir2.timeShift_from_cam_imu = t;
}

void ZR300Reader::printCamsInfo(){

    std::cout << std::endl;
    for(int iStream = 0; iStream < 5; ++iStream){
//        if(iStream == 2 || iStream == 3) continue;
        rs::intrinsics intrs = dev->get_stream_intrinsics((rs::stream)iStream);
        std::cout << (rs::stream)iStream << " fx fy:             " << intrs.fx << ", " << intrs.fy << std::endl
                  << (rs::stream)iStream << " ppx ppy:           " << intrs.ppx << ", " << intrs.ppy << std::endl
                  << (rs::stream)iStream << " distortion model:  " << intrs.model() << std::endl
                  << (rs::stream)iStream << " distortion coeffs: " << intrs.coeffs[0] << ", " << intrs.coeffs[1]
                  << ", " << intrs.coeffs[2] << ", " << intrs.coeffs[3]
                  << ", " << intrs.coeffs[4] << std::endl << std::endl;
    }

    for(int iStream = 0; iStream < 5; ++iStream) {
        for(int iStreamFrom = 0; iStreamFrom < 5; ++iStreamFrom) {
            if(iStreamFrom == iStream) continue;
            rs::extrinsics ex;
            for(int flp = 0; flp < 2; ++ flp){
                if(flp == 0){
                    ex = dev->get_extrinsics((rs::stream)iStreamFrom,(rs::stream)iStream);
                    std::cout << "Extrinsics from " << (rs::stream)iStreamFrom << " to " << (rs::stream)iStream << ": " << std::endl;
                }
                else{
                    ex = dev->get_extrinsics((rs::stream)iStream, (rs::stream)iStreamFrom);
                    std::cout << "Extrinsics from " << (rs::stream)iStream << " to " << (rs::stream)iStreamFrom << ": " << std::endl;
                }

                std::cout << ex.rotation[0] << " " << ex.rotation[3] << " " << ex.rotation[6] << " " << ex.translation[0] << " " << std::endl
                                            << ex.rotation[1] << " " << ex.rotation[4] << " " << ex.rotation[7] << " " << ex.translation[1] << " " << std::endl
                                            << ex.rotation[2] << " " << ex.rotation[5] << " " << ex.rotation[8] << " " << ex.translation[2] << " " << std::endl
                                            << 0.0 << " " << 0.0 << " " << 0.0 << " " << 1.0 << " " << std::endl
                                            << std::endl;
            }
        }
    }

    rs::extrinsics ex = dev->get_motion_extrinsics_from(rs::stream::fisheye);
    std::cout << "Extrinsics from " << rs::stream::fisheye << " to IMU: " << std::endl;
    std::cout << ex.rotation[0] << " " << ex.rotation[3] << " " << ex.rotation[6] << " " << ex.translation[0] << " " << std::endl
                                << ex.rotation[1] << " " << ex.rotation[4] << " " << ex.rotation[7] << " " << ex.translation[1] << " " << std::endl
                                << ex.rotation[2] << " " << ex.rotation[5] << " " << ex.rotation[8] << " " << ex.translation[2] << " " << std::endl
                                << 0.0 << " " << 0.0 << " " << 0.0 << " " << 1.0 << " " << std::endl
                                << std::endl;

    rs::motion_intrinsics in_motion = dev->get_motion_intrinsics();

    std::cout << "IMU.acc intrinsics: " << std::endl
              << "Scale X        cross axis        cross axis      Bias X" << std::endl
              << "cross axis     Scale Y           cross axis      Bias Y" << std::endl
              << "cross axis     cross axis        Scale Z         Bias Z" << std::endl << std::endl
              << in_motion.acc.data[0][0] << " " << in_motion.acc.data[0][1] << " " << in_motion.acc.data[0][2] << " " << in_motion.acc.data[0][3] << std::endl
                                                                                                                                                   << in_motion.acc.data[1][0] << " " << in_motion.acc.data[1][1] << " " << in_motion.acc.data[1][2] << " " << in_motion.acc.data[1][3] << std::endl
                                                                                                                                                                                                                                                                                        << in_motion.acc.data[2][0] << " " << in_motion.acc.data[2][1] << " " << in_motion.acc.data[2][2] << " " << in_motion.acc.data[2][3] << std::endl << std::endl;
    std::cout << "IMU.acc noise variances:" << std::endl
              << in_motion.acc.noise_variances[0] << " " << in_motion.acc.noise_variances[1] << " " << in_motion.acc.noise_variances[2] << std::endl << std::endl
              << "IMU.acc bias variances:" << std::endl
              << in_motion.acc.bias_variances[0] << " " << in_motion.acc.bias_variances[1] << " " << in_motion.acc.bias_variances[2] << std::endl << std::endl;

    std::cout << "IMU.gyro intrinsics: " << std::endl
              << "Scale X        cross axis        cross axis      Bias X" << std::endl
              << "cross axis     Scale Y           cross axis      Bias Y" << std::endl
              << "cross axis     cross axis        Scale Z         Bias Z" << std::endl << std::endl
              << in_motion.gyro.data[0][0] << " " << in_motion.gyro.data[0][1] << " " << in_motion.gyro.data[0][2] << " " << in_motion.gyro.data[0][3] << std::endl
                                                                                                                                                       << in_motion.gyro.data[1][0] << " " << in_motion.gyro.data[1][1] << " " << in_motion.gyro.data[1][2] << " " << in_motion.gyro.data[1][3] << std::endl
                                                                                                                                                                                                                                                                                                << in_motion.gyro.data[2][0] << " " << in_motion.gyro.data[2][1] << " " << in_motion.gyro.data[2][2] << " " << in_motion.gyro.data[2][3] << std::endl << std::endl;
    std::cout << "IMU.gyro noise variances:" << std::endl
              << in_motion.gyro.noise_variances[0] << " " << in_motion.gyro.noise_variances[1] << " " << in_motion.gyro.noise_variances[2] << std::endl << std::endl
              << "IMU.gyro bias variances:" << std::endl
              << in_motion.gyro.bias_variances[0] << " " << in_motion.gyro.bias_variances[1] << " " << in_motion.gyro.bias_variances[2] << std::endl << std::endl;
}

void ZR300Reader::initRS(){

    // Create a context object. This object owns the handles to all connected realsense devices.

    ctx = new rs::context();

    std::cout << "There are "<< ctx->get_device_count() << " connected RealSense devices.\n";

    if (ctx->get_device_count() == 0){
        std::cout << "RealSense Error: Could not find the ZR300 device!\n" << std::endl;
        exit(0);
    }

    dev = ctx->get_device(0);

    printf("    device 0: %s\n", dev->get_name());
    printf("    Serial number: %s\n", dev->get_serial());
    printf("    Firmware version: %s\n", dev->get_firmware_version());

    rgb.initialize(imgWidth,imgHeight);
    depth.initialize(imgWidth,imgHeight);
    ir1.initialize(imgWidth,imgHeight);
    ir2.initialize(imgWidth,imgHeight);
    fisheye.initialize(imgWidth,imgHeight);

    if (!dev->supports(rs::capabilities::motion_events)){
        printf("RealSense Error: This device does not support motion tracking!");
        exit(0);
    }

    auto motion_callback = [&](rs::motion_data entry){
        runImuCallback(entry);
    };

    auto timestamp_callback = [&](rs::timestamp_data entry){
        runMCtimestampCallback(entry);
    };

    // keep IMU callback initialization BEFORE the camera
    if (dev->supports(rs::capabilities::motion_events))
        dev->enable_motion_tracking(motion_callback,timestamp_callback);


    // Please the FPS of depth and color as 60, please make sure the depth image res should be [0,0].
    dev->enable_stream(rs::stream::depth, 0, 0, rs::format::z16, fps, rs::output_buffer_format::native);
    dev->enable_stream(rs::stream::color, imgWidth, imgHeight, rs::format::bgr8, fps, rs::output_buffer_format::native);
//    dev->enable_stream(rs::stream::infrared, 0, 0, rs::format::y8, fps, rs::output_buffer_format::native);
//    dev->enable_stream(rs::stream::infrared2, 0, 0, rs::format::y8, fps, rs::output_buffer_format::native);
//    dev->enable_stream(rs::stream::fisheye, imgWidth, imgHeight, rs::format::raw8, fps, rs::output_buffer_format::native);

    depth_intrin = dev->get_stream_intrinsics(rs::stream::depth);
    depth_to_color = dev->get_extrinsics(rs::stream::depth, rs::stream::color);
    color_intrin = dev->get_stream_intrinsics(rs::stream::color);
    scale = dev->get_depth_scale();

    dev->set_option(rs::option::color_enable_auto_white_balance, 0);
    dev->set_option(rs::option::color_enable_auto_exposure, 1);
    dev->set_option(rs::option::r200_emitter_enabled, 1);
    dev->set_option(rs::option::r200_lr_auto_exposure_enabled, 1);
    //    dev->set_option(rs::option::r200_lr_gain, 2000);
  
     /*
        dev->set_option(rs::option::color_brightness, 8);
        dev->set_option(rs::option::color_contrast, 50);
        dev->set_option(rs::option::color_exposure, 1);
        dev->set_option(rs::option::color_gain, 10);
*/
    //    dev->set_option(rs::option::color_gamma, 300);
    //    dev->set_option(rs::option::color_hue, 0);
    //    dev->set_option(rs::option::color_saturation, 50);
    //    dev->set_option(rs::option::color_white_balance, 5000);

//    dev->set_option(rs::option::fisheye_strobe, 1);
    //    dev->set_option(rs::option::fisheye_exposure, 0);
//    dev->set_option(rs::option::fisheye_color_auto_exposure,0);
    //    dev->set_option(rs::option::fisheye_external_trigger,1);

    dev->set_option(rs::option::frames_queue_size, 1);

    // Set callbacks prior to calling start()

    auto rgb_callback = [&](rs::frame frame) {
        runRGBCallback(frame);
    };
    auto depth_callback = [&](rs::frame frame) {
        runDepthCallback(frame);
    };
    auto irLeft_callback = [&](rs::frame frame) {
        runIRLeftCallback(frame);
    };
    auto irRight_callback = [&](rs::frame frame) {
        runIRRightCallback(frame);
    };
    auto fisheye_callback = [&](rs::frame frame) {
        runFishEyeCallback(frame);
    };

    switch (streamType) {

    case RSStreamType::RGB_DEPTH:
        dev->set_option(rs::option::r200_emitter_enabled, 1);
        dev->set_frame_callback(rs::stream::depth, depth_callback);
        dev->set_frame_callback(rs::stream::color, rgb_callback);
        break;

    case RSStreamType::IRSTEREO:
        dev->set_option(rs::option::r200_emitter_enabled, 0);
        dev->set_frame_callback(rs::stream::infrared, irLeft_callback);
        dev->set_frame_callback(rs::stream::infrared2, irRight_callback);
        break;

    case RSStreamType::ALL:
        dev->set_frame_callback(rs::stream::depth, depth_callback);
        dev->set_frame_callback(rs::stream::color, rgb_callback);
        dev->set_frame_callback(rs::stream::infrared, irLeft_callback);
        dev->set_frame_callback(rs::stream::infrared2, irRight_callback);
        dev->set_frame_callback(rs::stream::fisheye, fisheye_callback);
        break;
    }

    dev->start(rs::source::all_sources);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

void ZR300Reader::getColorIntrins(float& fx, float& fy, float& cx, float& cy){
    fx = color_intrin.fx;
    fy = color_intrin.fy;
    cx = color_intrin.ppx;
    cy = color_intrin.ppy;
}

void ZR300Reader::initRecData() {

    if(directory_exists(output_path)) {
        std::string c = "rm -r " + output_path;
        if(std::system(c.c_str())){
            std::cerr << "RealSense Dataset Info: folder exists, removing it." << std::endl;
        }
    }

    make_directory(output_path);

    for(int iCam = 0; iCam < 5; ++iCam) {
        std::string dataFolder = output_path + "/cam" + std::to_string(iCam);
        make_directory(dataFolder);
        std::string imFolder = dataFolder + "/data";
        make_directory(imFolder);
    }

    std::string imuFolder = output_path + "/imu0";
    make_directory(imuFolder);

    std::string imuCsvName = output_path + "/imu0/data.csv";
    imuCsv = new std::ofstream(imuCsvName.c_str(), std::ofstream::out);
    *imuCsv << "#timestamp,omega_x,omega_y,omega_z,alpha_x,alpha_y,alpha_z\n";

    std::string rgbCsvName = output_path + "/cam0/data.csv";
    rgbCsv = new std::ofstream(rgbCsvName.c_str(), std::ofstream::out);
    *rgbCsv << "#timestamp [ns],filename\n";

    std::string depthCsvName = output_path + "/cam1/data.csv";
    depthCsv = new std::ofstream(depthCsvName.c_str(), std::ofstream::out);
    *depthCsv << "#timestamp [ns],filename\n";

    std::string ir1CsvName = output_path + "/cam2/data.csv";
    ir1Csv = new std::ofstream(ir1CsvName.c_str(), std::ofstream::out);
    *ir1Csv << "#timestamp [ns],filename\n";

    std::string ir2CsvName = output_path + "/cam3/data.csv";
    ir2Csv = new std::ofstream(ir2CsvName.c_str(), std::ofstream::out);
    *ir2Csv << "#timestamp [ns],filename\n";

    std::string fisheyeCsvName = output_path + "/cam4/data.csv";
    fisheyeCsv = new std::ofstream(fisheyeCsvName.c_str(), std::ofstream::out);
    *fisheyeCsv << "#timestamp [ns],filename\n";

    std::this_thread::sleep_for(std::chrono::milliseconds(300));
}

void ZR300Reader::stopRecData(){
    isRecData = false;
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    depthCsv->close();
    rgbCsv->close();
    ir2Csv->close();
    ir1Csv->close();
    fisheyeCsv->close();
    imuCsv->close();

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

void ZR300Reader::stopDevice(){
    if(dev->is_streaming())
        dev->stop(rs::source::all_sources);
    if(dev->is_motion_tracking_active())
        dev->disable_motion_tracking();
}

void ZR300Reader::runImuCallback(rs::motion_data& entry){

    if (entry.timestamp_data.source_id == RS_EVENT_IMU_ACCEL) {

        for (std::deque<std::vector<rs::motion_data>>::iterator it = motionData.begin(); it != motionData.end(); ++it) {
            if (it->size() == 1) {
                it->at(0) = entry; // update the first accelerometer reading with the most recent one
            }
            else if (it->size() == 2) {
                it->push_back(entry); // complete the AGA vector (this can only be at the front of the deque)
                std::vector<rs::motion_data> AGA = motionData.front();
                motionData.pop_front();

                // process AGA
                Eigen::Vector3d gyr = Eigen::Vector3d((double)AGA.at(1).axes[0],(double)AGA.at(1).axes[1],(double)AGA.at(1).axes[2]);
                double timestampd = AGA.at(1).timestamp_data.timestamp; // in ms
                double lin_interp_factor = (timestampd - AGA.at(0).timestamp_data.timestamp) / (AGA.at(2).timestamp_data.timestamp - AGA.at(0).timestamp_data.timestamp);
                Eigen::Vector3d acc0 = Eigen::Vector3d((double)AGA.at(0).axes[0],(double)AGA.at(0).axes[1],(double)AGA.at(0).axes[2]);
                Eigen::Vector3d acc1 = Eigen::Vector3d((double)AGA.at(2).axes[0],(double)AGA.at(2).axes[1],(double)AGA.at(2).axes[2]);
                Eigen::Vector3d acc = acc0 + lin_interp_factor * (acc1 - acc0);
                imu.update(acc,gyr,timestampd);

                if(useOkvis){
                    okvis_estimator->addImuMeasurement(imu.getOkvisTimestamp(), imu.acc, imu.gyr);
//                    std::cout << "IMU timestamp:     secs: " << imu.getSec() << ", nanos: " << imu.getNSec() << std::endl;
                }

                // WRITE IMU MEASUREMENT TO CSV
                if(isRecData){
                    char filename[20];
                    sprintf(filename, "%014llu", (unsigned long long)imu.timestamp);
                    *imuCsv << filename << "," /*<< std::scientific << std::setprecision(18)*/ << imu.gyr(0) << "," << imu.gyr(1) << "," << imu.gyr(2)
                            << "," << imu.acc(0) << "," << imu.acc(1) << "," << imu.acc(2) << "\n";
                }

                if (motionData.empty()) {
                    std::vector<rs::motion_data> newVector;
                    newVector.push_back(entry);
                    motionData.push_back(newVector);
                }
            }
        }
        if (motionData.size() == 0) {
            std::vector<rs::motion_data> newVector;
            newVector.push_back(entry);
            motionData.push_back(newVector);
        }
    }
    else if (entry.timestamp_data.source_id == RS_EVENT_IMU_GYRO) {

        for (std::deque<std::vector<rs::motion_data>>::iterator it = motionData.begin(); it < motionData.end(); ++it) {
            if (it->size() == 1) {
                motionData.insert(it+1, std::vector<rs::motion_data>(*it));
                it->push_back(entry);
                ++it;
            }
        }
    }
}

void ZR300Reader::runRGBCallback(rs::frame& frame){
    if(frame.get_frame_number()%3 != 0 || frame.get_frame_number() < 5) return;
    rgb.update((uchar*)frame.get_data(),frame.get_timestamp());
    rgb.frameIndex.assign(frame.get_frame_number());

    if(isDebug){
        std::cout << "RGB timestamp:     secs: " << rgb.getSec() << ", nanos: " << rgb.getNSec() << std::endl;
        std::cout << "RGB Frame           No.: " << rgb.frameIndex.getValue() << std::endl;
    }

    if(streamType == RSStreamType::RGB_DEPTH && useOkvis){

        if(rgb.frameIndex.getValue()!=depth.frameIndex.getValue() || depth.timestamp != rgb.timestamp){
            std::cout << "RealSense Warning: Sync is wrong! Dropping the frames ID " << rgb.frameIndex.getValue() << std::endl;
            return;
        }

        cv::Mat depthMat = depth.img.clone();
        cv::Mat greyMat;

        if(rgb.img.channels() > 1){
            cv::cvtColor(rgb.img, greyMat, CV_BGR2GRAY);
        }else{
            greyMat = rgb.img.clone();
        }

        okvis_estimator->addImage(rgb.getOkvisTimestamp(), 0, greyMat, depthMat);
        okvis_estimator->addImage(rgb.getOkvisTimestamp(), 1, cv::Mat::zeros(greyMat.rows, greyMat.cols, CV_8U));
    }

    if(isRecData && frame.get_frame_number() > 5){
        char filename[20];
        sprintf(filename, "%014llu.png", (unsigned long long)rgb.timestamp);
        *rgbCsv << std::setfill('0') << std::setw(14) << (unsigned long long)rgb.timestamp << "," << filename << "\n";
        imwrite(output_path + "/cam0/data/" + filename, rgb.img);
    }
}

void ZR300Reader::runDepthCallback(rs::frame& frame){
    if(frame.get_frame_number() % 3 != 0 || frame.get_frame_number() < 5) return;
    uint8_t aligned[imgWidth*imgHeight*2];
    memset(aligned,0x00,imgWidth*imgHeight*2);
    okvis::align_depth_to_other(aligned,reinterpret_cast<const uint16_t *>(frame.get_data()),scale,depth_intrin,depth_to_color,color_intrin);
    depth.update(aligned,frame.get_timestamp());
    depth.frameIndex.assign(frame.get_frame_number());

    if(isDebug){
        std::cout << "Depth timestamp:   secs: " << depth.getSec() << ", nanos: " << depth.getNSec() << std::endl;
        std::cout << "Depth Frame         No.: " << depth.frameIndex.getValue() << std::endl;
    }

//    if(rgb.frameIndex.getValue()!=depth.frameIndex.getValue() || depth.timestamp != rgb.timestamp){
//        std::cout << "rgb.frameIndex  : " << rgb.frameIndex.getValue() << std::endl;
//        std::cout << "depth.frameIndex: " << depth.frameIndex.getValue() << std::endl;
//        std::cout << "rgb.timestamp   : " << rgb.timestamp << std::endl;
//        std::cout << "depth.timestamp : " << depth.timestamp << std::endl;
//        std::cout << "sync is wrong! Drop the frames ID " << rgb.frameIndex.getValue() << std::endl;
//        return;
//    }


    if(isRecData && frame.get_frame_number() > 5){
        char filename[20];
        sprintf(filename, "%014llu.png", (unsigned long long)depth.timestamp);
        *depthCsv << std::setfill('0') << std::setw(14) << (unsigned long long)depth.timestamp << "," << filename << "\n";
        imwrite(output_path + "/cam1/data/" + filename, depth.img);
    }
}

void ZR300Reader::runFishEyeCallback(rs::frame& frame){
    //    if(frame.get_frame_number()%2==0) return;
    fisheye.update((uchar*)frame.get_data(),frame.get_timestamp());
    fisheye.frameIndex.assign(frame.get_frame_number());

    if(isDebug){
        std::cout << "Fisheye timestamp: secs: " << fisheye.getSec() << ", nanos: " << fisheye.getNSec() << std::endl;
        std::cout << "Fisheye Frame       No.: " << fisheye.frameIndex.getValue() << std::endl;
    }

    if(isRecData){
        char filename[20];
        sprintf(filename, "%014llu.png", (unsigned long long)fisheye.timestamp);
        *fisheyeCsv << std::setfill('0') << std::setw(14) << (unsigned long long)fisheye.timestamp << "," << filename << "\n";
        imwrite(output_path + "/cam4/data/" + filename, fisheye.img);
    }
}

void ZR300Reader::runIRLeftCallback(rs::frame& frame){
    if(frame.get_frame_number()%2==0 || frame.get_frame_number() < 20) return;
    ir1.update((uchar*)frame.get_data(),frame.get_timestamp());
    ir1.frameIndex.assign(frame.get_frame_number());

    if(isDebug){
        std::cout << "IRLeft timestamp: secs: " << ir1.getSec() << ", nanos: " << ir1.getNSec() << std::endl;
        std::cout << "IRLeft Frame       No.: " << ir1.frameIndex.getValue() << std::endl;
    }

    if(isRecData && frame.get_frame_number() > 20){
        char filename[20];
        sprintf(filename, "%014llu.png", (unsigned long long)ir1.timestamp);
        *ir1Csv << std::setfill('0') << std::setw(14) << (unsigned long long)ir1.timestamp << "," << filename << "\n";
        imwrite(output_path + "/cam2/data/" + filename, ir1.img);
    }

}

void ZR300Reader::runIRRightCallback(rs::frame& frame){
    if(frame.get_frame_number()%2==0 || frame.get_frame_number() < 20) return;
    ir2.update((uchar*)frame.get_data(),frame.get_timestamp());
    ir2.frameIndex.assign(frame.get_frame_number());

    if(isDebug){
        std::cout << "IRRight timestamp: secs: " << ir2.getSec() << ", nanos: " << ir2.getNSec() << std::endl;
        std::cout << "IRRight Frame       No.: " << ir2.frameIndex.getValue() << std::endl;
    }

    if(isRecData && frame.get_frame_number() > 20){
        char filename[20];
        sprintf(filename, "%014llu.png", (unsigned long long)ir2.timestamp);
        *ir2Csv << std::setfill('0') << std::setw(14) << (unsigned long long)ir2.timestamp << "," << filename << "\n";
        imwrite(output_path + "/cam3/data/" + filename, ir2.img);
    }

    if(streamType == RSStreamType::IRSTEREO && useOkvis){
        if(ir1.frameIndex.getValue()!=ir2.frameIndex.getValue() || ir1.timestamp != ir2.timestamp){
            std::cerr << "RealSense Warning: Sync is wrong! Dropping the frames ID " << ir1.frameIndex.getValue() << std::endl;
            return;
        }

        if(ir1.img.empty() || ir2.img.empty()){
            std::cout << "RealSense Warning: IR image is empty" << std::endl;
            return;
        }

        cv::Mat r0,r1;
        cv::medianBlur(ir1.img, r0, 3);
        cv::medianBlur(ir2.img, r1, 3);

        okvis_estimator->addImage(ir1.getOkvisTimestamp(), 0, r0);
        okvis_estimator->addImage(ir1.getOkvisTimestamp(), 1, r1);
    }
}

bool ZR300Reader::file_exists(const std::string &path) {
    std::ifstream file(path);
    return file.good();
}

bool ZR300Reader::directory_exists(const std::string &path) {
#if defined(_WIN32)
    DWORD attributes = GetFileAttributes(path.c_str());
    return (attributes != INVALID_FILE_ATTRIBUTES &&
            (attributes & FILE_ATTRIBUTE_DIRECTORY));
#else
    struct stat sb;
    return stat(path.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode);
#endif
}

void ZR300Reader::make_directory(const std::string &path) {
#if defined(_WIN32)
    _mkdir(path.c_str());
#else
    mkdir(path.c_str(), 0755);
#endif
}

}
