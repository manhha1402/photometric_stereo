#include "photometric_stereo/photometric_stereo.h"

PhotoStereo::PhotoStereo(const Setting& setting):setting_(setting),
    shadow_trick_(false),num_channels_(1)
{

}
PhotoStereo::~PhotoStereo()
{}
Eigen::Vector3f PhotoStereo::sph2cart(const float& theta,const float& phi,const float& r)
{
    Eigen::Vector3f cart;
    cart(0) = r * std::sin(theta) * std::cos(phi);
    cart(1) = r * std::sin(theta) * std::sin(phi);
    cart(2) = r * std::cos(theta);
    return cart;
}



bool PhotoStereo::readYaleFaceDataset()
{
    cv::Mat ambient_image = cv::imread(setting_.yale_face_path_+"/yaleB02_P00_Ambient.pgm",cv::IMREAD_UNCHANGED);

    cv::String path = setting_.yale_face_path_+"/yaleB02_P00A*"+".pgm";
    //https://answers.opencv.org/question/69712/load-multiple-images-from-a-single-folder/
    std::vector<cv::String> image_files;
    cv::glob(path,image_files,true); // recurse
    //Set size of light source matrix
    std::vector<Eigen::Vector3f> total_light_sources;
    for (std::size_t k = 0; k< image_files.size(); k++)
    {
        //read model images as gray images, if images are color, the function performs conversion
        // from bgr to gray :  gray = 0.21 R + 0.72 G + 0.07 B
        model_images_.push_back(cv::imread(image_files[k],cv::IMREAD_UNCHANGED));

        // get light sources from files
        std::string light_source_name = image_files[k].substr(setting_.yale_face_path_.size()+
                                                              setting_.yale_face_name_.size()+1);
        std::size_t m0 = image_files[k].find("A");
        std::size_t m1 = image_files[k].find("E");
        float x = std::stof(image_files[k].substr(m0+1,4));
        float y = std::stof(image_files[k].substr(m1+1,3));
        Eigen::Vector3f cart = sph2cart(M_PI*x/180.0f,M_PI*y/180.0f,1.0f);
        total_light_sources.push_back(cart);

    }

    std::cout<<"Number of model images: "<<model_images_.size()<<std::endl;
    if(setting_.num_of_images_<0 || setting_.num_of_images_> model_images_.size())
    {
        setting_.num_of_images_ = model_images_.size();
        std::cout<<"Number of test images is invalid, using total model images: "<<setting_.num_of_images_<<std::endl;
    }
    lights_sources_ = cv::Mat(setting_.num_of_images_, 3, CV_32FC1);
    for (int k = 0; k < setting_.num_of_images_; k++) {
        //     light_source_dirs_.row(k).array() = total_light_sources[k];
        lights_sources_.at<float>(k,0) = total_light_sources[k](0);
        lights_sources_.at<float>(k,1) = total_light_sources[k](1);
        lights_sources_.at<float>(k,2) = total_light_sources[k](2);

    }
    std::cout<<lights_sources_<<std::endl;
    image_width_ = model_images_[0].cols;
    image_height_= model_images_[0].rows;


}

bool PhotoStereo::readGrayData()
{
    cv::String path = setting_.gray_data_path_+"/*"+setting_.suffix_;

    //https://answers.opencv.org/question/69712/load-multiple-images-from-a-single-folder/
    std::vector<cv::String> image_files;
    cv::glob(path,image_files,true); // recurse
    //Set size of light source matrix
    std::vector<Eigen::Vector3f> total_light_sources;
    for (std::size_t k = 0; k< image_files.size(); k++)
    {

        model_images_.push_back(cv::imread(image_files[k],CV_LOAD_IMAGE_GRAYSCALE));
        num_channels_ = 1;

        // get light sources from files
        std::string light_source_name = image_files[k].substr(setting_.gray_data_path_.size()+setting_.model_name_.size()+1);
        std::size_t m0 = light_source_name.find("_");
        float x = std::stof(light_source_name.substr(0,m0));
        std::size_t m1 = light_source_name.find(setting_.suffix_);
        float y = std::stof(light_source_name.substr(m0+1,m1-setting_.suffix_.size()));
        total_light_sources.push_back(Eigen::Vector3f(x,y,setting_.z_));

    }
    std::cout<<"Number of model images: "<<model_images_.size()<<std::endl;
    if(setting_.num_of_images_<0 || setting_.num_of_images_> model_images_.size())
    {
        setting_.num_of_images_ = model_images_.size();
        std::cout<<"Number of test images is invalid, using total model images: "<<setting_.num_of_images_<<std::endl;
    }
    std::cout<<"Using "<<setting_.num_of_images_<<" for processing"<<std::endl;
    lights_sources_ = cv::Mat(setting_.num_of_images_, 3, CV_32FC1);
    for (int k = 0; k < setting_.num_of_images_; k++) {
        lights_sources_.at<float>(k,0) = -total_light_sources[k](0);
        lights_sources_.at<float>(k,1) = total_light_sources[k](1);
        lights_sources_.at<float>(k,2) = total_light_sources[k](2);

    }
    image_width_ = model_images_[0].cols;
    image_height_= model_images_[0].rows;



    return true;
}

bool PhotoStereo::readRGBData()
{
    cv::String path = setting_.rgb_data_path_+"/*"+setting_.suffix_;
    //https://answers.opencv.org/question/69712/load-multiple-images-from-a-single-folder/
    std::vector<cv::String> image_files;
    cv::glob(path,image_files,true); // recurse
    //Set size of light source matrix
    std::vector<Eigen::Vector3f> total_light_sources;
    for (std::size_t k = 0; k< image_files.size(); k++)
    {

        model_images_.push_back(cv::imread(image_files[k],CV_LOAD_IMAGE_COLOR));
        num_channels_ = 3;

        // get light sources from files
        std::string light_source_name = image_files[k].substr(setting_.rgb_data_path_.size()+setting_.model_name_.size()+1);
        std::size_t m0 = light_source_name.find("_");
        float x = std::stof(light_source_name.substr(0,m0));
        std::size_t m1 = light_source_name.find(setting_.suffix_);
        float y = std::stof(light_source_name.substr(m0+1,m1-setting_.suffix_.size()));
        total_light_sources.push_back(Eigen::Vector3f(x,y,setting_.z_));

    }
    std::cout<<"Number of model images: "<<model_images_.size()<<std::endl;
    if(setting_.num_of_images_<0 || setting_.num_of_images_> model_images_.size())
    {
        setting_.num_of_images_ = model_images_.size();
        std::cout<<"Number of test images is invalid, using total model images: "<<setting_.num_of_images_<<std::endl;
    }
    std::cout<<"Using "<<setting_.num_of_images_<<" for processing"<<std::endl;

    lights_sources_ = cv::Mat(setting_.num_of_images_, 3, CV_32FC1);
    for (int k = 0; k < setting_.num_of_images_; k++) {
        lights_sources_.at<float>(k,0) = -total_light_sources[k](0);
        lights_sources_.at<float>(k,1) = total_light_sources[k](1);
        lights_sources_.at<float>(k,2) = total_light_sources[k](2);

    }
    image_width_ = model_images_[0].cols;
    image_height_= model_images_[0].rows;
    return true;
}




std::tuple<cv::Mat,cv::Mat> PhotoStereo::estimateAlbNrm()
{
    cv::Mat normal_map = cv::Mat::zeros(image_height_,image_width_,CV_32FC3);
    cv::Mat albedo_map = cv::Mat::zeros(image_height_,image_width_,CV_32FC1);

    cv::Mat light_source_inv;
    cv::invert(lights_sources_, light_source_inv, cv::DECOMP_SVD);
    cv::Mat intensity_vec(setting_.num_of_images_,1,CV_32FC1);

    float albedo_val;
    //For each pixel in normal map, get intensity values then estimate normal
    for (int x = 0; x < image_width_; x++)
        for (int y = 0;y < image_height_; y++) {

            //get intensity value
            for (int n = 0; n < setting_.num_of_images_; n++) {

                intensity_vec.at<float>(n,0) =  static_cast<float>(model_images_[n].at<uchar>(cv::Point(x,y))/255.0f);
            }

            cv::Mat normal  = light_source_inv * intensity_vec;
            albedo_val = static_cast<float>(std::sqrt(normal.dot(normal)));
            if (albedo_val > 0.0f) {
                normal = normal/ albedo_val;
            }
            if (normal.at<float>(2,0) == 0.0f) {
                normal.at<float>(2,0) = 1.0;
            }

            normal_map.at<cv::Vec3f>(y,x) = normal;
            albedo_map.at<float>(y,x) = albedo_val;
        }

    return std::make_tuple(normal_map,albedo_map);



}

std::tuple<cv::Mat,cv::Mat,std::vector<cv::Mat>> PhotoStereo::estimateAlbNrmColor()
{

    cv::Mat normal_map = cv::Mat::zeros(image_height_,image_width_,CV_32FC3);
    cv::Mat albedo_map = cv::Mat::zeros(image_height_,image_width_,CV_32FC1);

    cv::Mat light_source_inv;
    cv::invert(lights_sources_, light_source_inv, cv::DECOMP_SVD);
    std::vector<cv::Mat> intensity_vec(num_channels_);
    std::vector<cv::Mat> normal_map_vec(num_channels_);
    std::vector<cv::Mat> normal_matrix(num_channels_);
    std::vector<cv::Mat> albedo_map_vec(num_channels_);
    std::vector<float> albedo_val(num_channels_);


    for (int i=0;i<num_channels_;i++) {
        intensity_vec[i] = cv::Mat::zeros(setting_.num_of_images_,1,CV_32FC1);
        albedo_map_vec[i] = cv::Mat::zeros(image_height_,image_width_,CV_32FC1);
        normal_map_vec[i] = cv::Mat::zeros(image_height_,image_width_,CV_32FC3);

    }

    //For each pixel in normal map, get rgb values then estimate normal
    for (int x = 0; x < image_width_; x++)
        for (int y = 0;y < image_height_; y++) {

            for (int n = 0; n < setting_.num_of_images_; n++) {
                int b =  model_images_[n].at<cv::Vec3b>(cv::Point(x,y))[0];
                int g =  model_images_[n].at<cv::Vec3b>(cv::Point(x,y))[1];
                int r =  model_images_[n].at<cv::Vec3b>(cv::Point(x,y))[2];
                intensity_vec[0].at<float>(n,0) = static_cast<float>(b)/255.0f;
                intensity_vec[1].at<float>(n,0) = static_cast<float>(g)/255.0f;
                intensity_vec[2].at<float>(n,0) = static_cast<float>(r)/255.0f;
            }

            for (int channel = 0; channel < num_channels_; channel++) {
                normal_matrix[channel] = light_source_inv * intensity_vec[channel];
                albedo_val[channel] = static_cast<float>(std::sqrt(cv::Mat(normal_matrix[channel]).dot(normal_matrix[channel])));
                if (albedo_val[channel] > 0.0f) {
                    normal_matrix[channel] = normal_matrix[channel]/ albedo_val[channel];
                }
                if (normal_matrix[channel].at<float>(2,0) == 0.0f) {
                    normal_matrix[channel].at<float>(2,0) = 1.0;
                }

                normal_map_vec[channel].at<cv::Vec3f>(y,x) = normal_matrix[channel];
                albedo_map_vec[channel].at<float>(y,x) = albedo_val[channel];
            }

        }

    //Sum up all channels and take average
    for (int x = 0; x < image_width_; x++)
    {
        for (int y = 0;y < image_height_; y++){
            float val=0.0f;

            for (int d = 0; d < 3; d++){
                int num_normal=0;
                for (int channel = 0; channel < num_channels_; channel++)
                {
                    if(normal_map_vec[channel].at<cv::Vec3f>(y,x)[d] != 0.0f)
                    {
                        val +=normal_map_vec[channel].at<cv::Vec3f>(y,x)[d];
                        num_normal++;
                    }
                }
                if(num_normal != 0)
                    normal_map.at<cv::Vec3f>(y,x)[d] = val/static_cast<float>(num_normal);

            }
        }
    }
    //albedo
    for (int x = 0; x < image_width_; x++)
    {
        for (int y = 0;y < image_height_; y++){
            float val=0.0f;


            int num_albedo=0;
            for (int channel = 0; channel < num_channels_; channel++)
            {
                if(albedo_map_vec[channel].at<float>(y,x) != 0.0f)
                {
                    val +=albedo_map_vec[channel].at<float>(y,x);
                    num_albedo++;
                }
            }
            if(num_albedo != 0)
                albedo_map.at<float>(y,x) = val/static_cast<float>(num_albedo);
        }
    }
    return std::make_tuple(normal_map,albedo_map,normal_map_vec);

}







std::tuple<cv::Mat,cv::Mat,cv::Mat> PhotoStereo::checkIntegrability(const cv::Mat& normal_maps)
{
    grads_x_ = cv::Mat::zeros(image_height_,image_width_,CV_32FC1);
    grads_y_ = cv::Mat::zeros(image_height_,image_width_,CV_32FC1);
    cv::Mat S = cv::Mat::zeros(image_height_,image_width_,CV_32FC1);
    float threshold = 0.004f;
    //For each pixel in normal map, compute derivatives in x,y directions ( Orthographic view)
    std::vector<cv::Mat> normals;
    split(normal_maps, normals);
    // nx/nz, ny/nz
    cv::divide(normals[0],normals[2],grads_x_);
    cv::divide(normals[1],normals[2],grads_y_);

    //integrability check error = (d(grads_x_)/d(y) - d(grads_y_)/d(x))^2
    // We can even use Sobel filter to estimate derivative of image using cv::Sobel
    cv::Mat second_grads_x_y(image_height_,image_width_,CV_32FC1,cv::Scalar::all(0));
    cv::Mat second_grads_y_x(image_height_,image_width_,CV_32FC1,cv::Scalar::all(0));
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int x = 0; x < image_width_; x++) {
        for (int y = 0; y < image_height_; y++) {
            if(x == 0 || y == 0 || x == (image_width_-1) || y == (image_height_-1) ) continue;
            second_grads_x_y.at<float>(y,x) = (grads_x_.at<float>(y+1,x) - grads_x_.at<float>(y-1,x))/2.0f;
            second_grads_y_x.at<float>(y,x) = (grads_y_.at<float>(y,x+1) - grads_y_.at<float>(y,x-1))/2.0f;
            float error = second_grads_x_y.at<float>(y,x) - second_grads_y_x.at<float>(y,x);
            S.at<float>(y,x) = error*error;
            if(S.at<float>(y,x) < threshold) {
                S.at<float>(y,x) = 0;
            }
        }
    }
    return std::make_tuple(grads_x_,grads_y_,S);

}

std::tuple<cv::Mat,cv::Mat,cv::Mat> PhotoStereo::getDepthMap(const cv::Mat& normal_map,
                                                             const cv::Mat& grads_x,
                                                             const cv::Mat& grads_y)
{

    cv::Mat avarage_depth_map(image_height_, image_width_, CV_32FC1, cv::Scalar::all(0));
    cv::Mat column_depth_map(image_height_, image_width_, CV_32FC1, cv::Scalar::all(0));
    cv::Mat row_depth_map(image_height_, image_width_, CV_32FC1, cv::Scalar::all(0));

    //Row major
    for (int row = 1; row < image_height_; row++)
    {
        if(normal_map.at<cv::Vec3f>(row,0)[2] !=0.0f)
            row_depth_map.at<float>(row,0) = row_depth_map.at<float>(row-1,0)
                    + grads_y.at<float>(row,0);
        else
            row_depth_map.at<float>(row,0) =  row_depth_map.at<float>(row-1,0);
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(int row=0; row< image_height_; row++)
        for (int col = 1; col < image_width_; col++) {
            if(normal_map.at<cv::Vec3f>(row,col)[2] !=0.0f)
                row_depth_map.at<float>(row,col) = row_depth_map.at<float>(row,col-1)
                        + grads_x.at<float>(row,col);
            else
                row_depth_map.at<float>(row,col) = row_depth_map.at<float>(row,col-1);

        }

    //Column major
    for (int col = 1; col < image_width_; col++)
    {
        if(normal_map.at<cv::Vec3f>(0,col)[2] !=0.0f)
            column_depth_map.at<float>(0,col) = column_depth_map.at<float>(0,col-1)
                    + grads_x.at<float>(0,col);
        else
            column_depth_map.at<float>(0,col) =  column_depth_map.at<float>(0,col-1);
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int col = 0; col < image_width_; col++)
        for(int row= 1; row< image_height_; row++)
        {
            if(normal_map.at<cv::Vec3f>(row,col)[2] !=0.0f)
                column_depth_map.at<float>(row,col) = column_depth_map.at<float>(row-1,col)
                        + grads_y.at<float>(row,col);
            else
                column_depth_map.at<float>(row,col) = column_depth_map.at<float>(row-1,col);
            //average
            avarage_depth_map.at<float>(row,col) =(column_depth_map.at<float>(row,col) + row_depth_map.at<float>(row,col))/2.0f;
        }



    return std::make_tuple(column_depth_map,row_depth_map,avarage_depth_map);
}



void PhotoStereo::saveTriangleMesh(const std::string& mesh_file,const cv::Mat& depth_map,
                                   const cv::Mat& albedo_map)
{
    //change range to 0-255
    cv::Mat color;
    cv::normalize(albedo_map, color, 0, 255, cv::NORM_MINMAX);

    std::ofstream file_out { mesh_file };
    if (!file_out.is_open())
        return;
    int num_vertices = image_width_*image_height_;
    int num_faces = (image_width_-1) * (image_height_-1) * 2;
    file_out << "ply" << std::endl;
    file_out << "format ascii 1.0" << std::endl;
    file_out << "element vertex " << num_vertices << std::endl;
    file_out << "property float x" << std::endl;
    file_out << "property float y" << std::endl;
    file_out << "property float z" << std::endl;
    file_out << "property uchar red" << std::endl;
    file_out << "property uchar green" << std::endl;
    file_out << "property uchar blue" << std::endl;
    file_out << "element face " << num_faces << std::endl;
    file_out << "property list uchar int vertex_indices" << std::endl;
    file_out << "end_header" << std::endl;

    for (int y=0; y<image_height_; y++) {
        for (int x=0; x<image_width_; x++) {
            int c = static_cast<int>(color.at<float>(y, x));
            if (c > 0) {
                file_out << x << " " << y << " " << depth_map.at<float>(y,x) << " "<< c << " " << c << " " << c<<std::endl;
            }
            else
                file_out << x << " " << y << " " << depth_map.at<float>(y,x) << " "<< 255 << " " << 255 << " " << 255<<std::endl;

        }
    }
    for (int i=0; i<image_height_-1; i++) {
        for (int j=0; j<image_width_-1; j++) {
            int v1,v2,v3;
            v1 = j+(i*image_width_);
            v2 = (i+1)*image_width_+j;
            v3 = j+(i*image_width_)+1;
            file_out << 3 << " " << v1 << " " << v2 << " " << v3 << std::endl;
            v1 = (i+1)*image_width_+j;
            v2 = (i+1)*image_width_+j+1;
            v3 = j+(i*image_width_)+1;
            file_out << 3 << " " << v1 << " " << v2 << " " << v3 << std::endl;
        }
    }

}


