#ifndef PHOTOMETRIC_STEREO_H
#define PHOTOMETRIC_STEREO_H

#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Core>
#include "photometric_stereo/settings.h"

class PhotoStereo
{
public:
    /**
     * @brief Ptr is a type-definition for a shared superclass-pointer.
     */
    typedef std::shared_ptr<PhotoStereo> Ptr;
    ///
    /// \brief PhotoStereo constructor of the class
    /// \param setting the setting struct contains some variables that can be changed from .yaml file
    ///
    PhotoStereo(const Setting& setting );
    ////
    /// \brief ~PhotoStereo destructor of the class
    ///
    virtual ~PhotoStereo();
    ////
    /// \brief readGrayData read given gray data and get the lighting directions from name of the images
    /// \return true if success otherwise failed
    ///
    bool readGrayData();
    ////
    /// \brief readRGBData read given color data and get the lighting directions from name of the images
    /// \return true if success otherwise failed
    ///
    bool readRGBData();
    ////
    /// \brief readYaleFaceDataset read Yale Face data and get the lighting directions from name of the images
    /// \return true if success otherwise failed
    ///
    bool readYaleFaceDataset();
    ////
    /// \brief estimateAlbNrm estimate surface normal and albedo of the object given lighting directions and intensity images
    /// \return surface normal map (3 channels) and albedo (1 channel)
    ///
    std::tuple<cv::Mat,cv::Mat> estimateAlbNrm();
    //estimate albedo and normal of color images
    std::tuple<cv::Mat,cv::Mat, std::vector<cv::Mat> > estimateAlbNrmColor();
    ////
    /// \brief checkIntegrability check the integrability mentioned in section 5.4 of the book (Forsyth and Ponce, Computer Vision: A
    /// Modern Approach).
    /// \param normal_maps the surface normal map obtained from Photometric Stereo technique
    /// \return The gradients in x,y directions of height map z = f(x,y) and the error map S.
    ///
    std::tuple<cv::Mat,cv::Mat,cv::Mat> checkIntegrability(const cv::Mat& normal_maps);
    ////
    /// \brief getDepthMap get height map using 3 integration paths (Row-Major, Column-Major and Average)
    /// \param normal_map the surface normal map obtained from Photometric Stereo technique
    /// \param grads_x The gradient in x direction of height map z = f(x,y)
    /// \param grads_yThe gradient in y direction of height map z = f(x,y)
    /// \return 3 depth map (1 channel) result using 3 different integration maps (Row-Major, Column-Major and Average)
    ///
    std::tuple<cv::Mat,cv::Mat,cv::Mat> getDepthMap(const cv::Mat& normal_map,
                                                    const cv::Mat& grads_x,
                                                    const cv::Mat& grads_y);

    void saveTriangleMesh(const std::string& mesh_file,const cv::Mat& depth_map,
                          const cv::Mat& albedo_map);
    ////
    /// \brief setting_  the setting struct contains some variables that can be changed from .yaml file. It will be assigned in constructor
    ///
    Setting setting_;
    ////
    /// \brief image_width_ the number of pixels in x direction of input image
    ///
    int image_width_;
    ////
    /// \brief image_width_ the number of pixels in y direction of input image
    ///
    int image_height_;
    ////
    /// \brief num_channels_ number of channels of input data (1 - gray, 3 - RGB)
    ///
    int num_channels_;
private:
    ////
    /// \brief model_images_
    ///
    std::vector<cv::Mat> model_images_;
    ////
    /// \brief lights_sources_ lighting directions that will be gotten from reading the data
    ///
    cv::Mat lights_sources_;
    ////
    /// \brief grads_x_ the gradient in x direction of the height map f(x,y)
    ///
    cv::Mat grads_x_;
    ////
    /// \brief grads_y_ the gradient in y direction of the height map f(x,y)
    ///
    cv::Mat grads_y_;
    ////
    /// \brief shadow_trick_ the shadow trick of photometric stereo mentioned in section 5.4 of the book (Forsyth and Ponce, Computer Vision: A
    /// Modern Approach) -> not yet implemented
    ///
    bool shadow_trick_;
    ////
    /// \brief sph2cart  transforms the corresponding elements of spherical coordinate arrays to Cartesian coordinates
    /// \param theta  the angle relative to the positive x-axis.
    /// \param phi  the angle relative to the xy-plane.
    /// \param r the distance to the origin (0, 0, 0).
    /// \return Cartesian coordinates (x,y,z)
    ///
    Eigen::Vector3f sph2cart(const float& theta,const float& phi,const float& r);
};


#endif /* PHOTOMETRIC_STEREO_H */
