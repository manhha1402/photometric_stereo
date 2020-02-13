#include <iostream>
#include "photometric_stereo/photometric_stereo.h"
#include "photometric_stereo/settings.h"

int main(int argc, char** argv)
{

    std::string config_file = "../config.yaml";
    Setting setting;
    setting.readConfigFile(config_file);
    setting.printInfo(std::cout);
    std::cout<<"Data path: "<<setting.rgb_data_path_<<std::endl;
    PhotoStereo::Ptr photo =  PhotoStereo::Ptr(new PhotoStereo(setting));
    photo->readRGBData();
    cv::Mat normal_map,albedo_map;
    std::vector<cv::Mat> normal_map_vec;
    std::cout<<"Estimate normal and albedo ..."<<std::endl;
    std::tie(normal_map,albedo_map,normal_map_vec) = photo->estimateAlbNrmColor();
    cv::Mat q_grads,p_grads,S;
    std::cout<<"Check integrability ..."<<std::endl;

    std::tie(p_grads,q_grads,S) = photo->checkIntegrability(normal_map);
    cv::Mat depth_map,column_depth_map,row_depth_map;
    std::cout<<"get height map using column, row, average methods..."<<std::endl;

    std::tie(column_depth_map,row_depth_map,depth_map) = photo->getDepthMap(normal_map,p_grads,q_grads);


    std::ostringstream ss;
    ss<<photo->setting_.num_of_images_;
    photo->saveTriangleMesh(photo->setting_.model_name_+"_meshRGB"+ss.str()+".ply",depth_map,albedo_map);
    photo->saveTriangleMesh(photo->setting_.model_name_+"_row_meshRGB.ply",row_depth_map,albedo_map);
    photo->saveTriangleMesh(photo->setting_.model_name_+"_column_meshRGB.ply",column_depth_map,albedo_map);

    //Store error map in text file
    std::string error_map_file = photo->setting_.model_name_+"_error_mapRGB"+ss.str()+".txt";
    std::ofstream file_out { error_map_file };
    for (int x=0;x<photo->image_width_;x++)
      for (int y=0;y<photo->image_height_;y++)
      {
        file_out << S.at<float>(y,x) << std::endl;
      }
    file_out.close();

    S.convertTo(S, CV_8UC1, 255, 0);
    cv::imwrite(photo->setting_.model_name_+"_error_mapRGB"+ss.str()+ ".png",S);
    q_grads.convertTo(q_grads, CV_8UC1, 255, 0);
    cv::imwrite(photo->setting_.model_name_+"_q_gradsRGB"+ss.str()+ ".png",q_grads);
    p_grads.convertTo(p_grads, CV_8UC1, 255, 0);
    cv::imwrite(photo->setting_.model_name_+"_p_gradsRGB"+ss.str()+ ".png",p_grads);
    normal_map.convertTo(normal_map, CV_8UC3, 255, 0);
    normal_map_vec[0].convertTo(normal_map_vec[0], CV_8UC3, 255, 0);
    normal_map_vec[1].convertTo(normal_map_vec[1], CV_8UC3, 255, 0);
    normal_map_vec[2].convertTo(normal_map_vec[2], CV_8UC3, 255, 0);


    cv::cvtColor(normal_map, normal_map, CV_BGR2RGB); //OpenCV store BGR
    cv::cvtColor(normal_map_vec[0], normal_map_vec[0], CV_BGR2RGB); //OpenCV store BGR
    cv::cvtColor(normal_map_vec[1], normal_map_vec[1], CV_BGR2RGB); //OpenCV store BGR
    cv::cvtColor(normal_map_vec[2], normal_map_vec[2], CV_BGR2RGB); //OpenCV store BGR


    cv::imwrite(photo->setting_.model_name_+"_normal_mapRGB"+ss.str()+".png",normal_map);
    cv::imwrite(photo->setting_.model_name_+"_normal_mapB"+ss.str()+".png",normal_map_vec[0]);
    cv::imwrite(photo->setting_.model_name_+"_normal_mapG"+ss.str()+".png",normal_map_vec[1]);
    cv::imwrite(photo->setting_.model_name_+"_normal_mapR"+ss.str()+".png",normal_map_vec[2]);

    albedo_map.convertTo(albedo_map, CV_8UC1, 255, 0);
    cv::imwrite(photo->setting_.model_name_+"_albedo_mapRGB"+ss.str()+".png",albedo_map);



    cv::normalize(depth_map, depth_map, 0, 255, cv::NORM_MINMAX);
    cv::normalize(column_depth_map, column_depth_map, 0, 255, cv::NORM_MINMAX);
    cv::normalize(row_depth_map, row_depth_map, 0, 255, cv::NORM_MINMAX);


    cv::imwrite(photo->setting_.model_name_+"_depth_map"+ss.str()+".png",depth_map);
    cv::imwrite(photo->setting_.model_name_+"_row_depth_map"+ss.str()+".png",row_depth_map);
    cv::imwrite(photo->setting_.model_name_+"_column_depth_map"+ss.str()+".png",column_depth_map);


    return 0;
}
