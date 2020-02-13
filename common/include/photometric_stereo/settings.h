
#pragma once

#include <iostream>
#include <vector>
#include <yaml-cpp/yaml.h>

/**
 * @brief The Setting class provides the API for the parameter configuration.It loads parameters from yaml file
 * folder to perform needed functions
 *
 */
#define DUMPSTR_WNAME(os, name, a) \
    do { (os) << (name) << ": " << (a) << std::endl; } while(false)

#define DUMPSTR(os, a) DUMPSTR_WNAME((os), #a, (a))

struct Setting
{

    std::string gray_data_path_ = "../photometrics_images/SphereGray5/"; //default
    std::string rgb_data_path_ = "../photometrics_images/MonkeyColor/"; //default
    std::string yale_face_path_ = "../photometrics_images/yaleB02/"; //default
    ///
    /// surfix of input image
    std::string suffix_ = ".png";

    ///
    /// This need to change while testing with different data => if you change SphereColor then change model name to Sphere
    ///
    std::string model_name_ = "Sphere";
    std::string yale_face_name_ = "yaleB02";
    int num_of_images_ = 5;
    float z_ = 0.5f;

    void readConfigFile(const std::string& config_file_path)
    {

        // ********** try to load the yaml file that is located at the given path **********//
        YAML::Node config_file = YAML::LoadFile(config_file_path);

        if(config_file["gray_data_path"])
            gray_data_path_ = config_file["gray_data_path"].as<std::string>();
        if(config_file["rgb_data_path"])
            rgb_data_path_ = config_file["rgb_data_path"].as<std::string>();
        if(config_file["yale_face_path"])
            yale_face_path_ = config_file["yale_face_path"].as<std::string>();
        if(config_file["suffix"])
            suffix_ = config_file["suffix"].as<std::string>();
        if(config_file["model_name"])
            model_name_ = config_file["model_name"].as<std::string>();
        if(config_file["yale_face_name"])
            yale_face_name_ = config_file["yale_face_name"].as<std::string>();
        if(config_file["num_of_images"])
            num_of_images_ = config_file["num_of_images"].as<int>();
        if(config_file["z"])
            z_ = config_file["z"].as<float>();

    }
    ////
    /// \brief printInfo perform operator of variables
    /// \param os operator(std::cout, std::in, ...)
    /// \return
    ///
    std::ostream& printInfo(std::ostream& os)
    {

       // DUMPSTR(os, gray_data_path_);
        //DUMPSTR(os, rgb_data_path_);
       // DUMPSTR(os, yale_face_path_);
        DUMPSTR(os, suffix_);
        DUMPSTR(os, model_name_);
        DUMPSTR(os, yale_face_name_);
        DUMPSTR(os, num_of_images_);
        DUMPSTR(os, z_);

    }
};





//#endif
