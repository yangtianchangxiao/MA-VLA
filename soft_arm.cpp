 
 
#include <iostream>
#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/Imu.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <nav_msgs/Path.h>
#include <tf/tf.h>
#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>
#include "std_msgs/String.h"
#include <stdlib.h>
#include <fstream>
#include <sstream>
#include <string>
#include "std_msgs/Float32.h"
#include "forward_kinematics.h"


struct Pcc_config sec1;
struct Pcc_config sec2;
struct Pcc_config sec3;
struct Pcc_config sec4;

struct Pcc_config cur_from_IMUs_sec1;
struct Pcc_config cur_from_IMUs_sec2;
struct Pcc_config cur_from_IMUs_sec3;
struct Pcc_config cur_from_LENs_sec1;
struct Pcc_config cur_from_LENs_sec2;
struct Pcc_config cur_from_LENs_sec3;

struct Pcc_config des_from_IK_sec1;
struct Pcc_config des_from_IK_sec2;
struct Pcc_config des_from_IK_sec3;
struct Pcc_config des_from_IK_sec4;

geometry_msgs::PoseStamped vic_basePos_info;
geometry_msgs::PoseStamped vic_sec3Pos_info;
geometry_msgs::PoseStamped vic_EEPos;

geometry_msgs::PoseStamped IMUs_EEPos;

int main_hz = 0;

void robot_worldFrame(ros::Publisher robotFrame_pub1, ros::Publisher robotFrame_pub2, ros::Publisher robotFrame_pub3);
Eigen::Vector3f config_to_translationMatrix(float alpha, float beta, float arcLength);
Eigen::Matrix3f config_to_rotationMatrix(float alpha, float beta);
visualization_msgs::MarkerArray backbone_generation(int id, int resolution, float alpha, float beta, float arcLength, Eigen::Matrix3f last_Rot, Eigen::Vector3f last_enddisk_pos);
void disk_visualization(int id, int resolution, float alpha, float beta, float arcLength, Eigen::Matrix3f last_Rot, Eigen::Vector3f last_enddisk_pos) ;


sensor_msgs::Imu des_config_info;
void des_config_cb(const sensor_msgs::Imu::ConstPtr& config) 
{
    des_config_info = *config;

    des_from_IK_sec1.alpha  = des_config_info.angular_velocity_covariance[0];
    des_from_IK_sec1.beta   = des_config_info.angular_velocity_covariance[1];
    des_from_IK_sec2.alpha  = des_config_info.angular_velocity_covariance[2];
    des_from_IK_sec2.beta   = des_config_info.angular_velocity_covariance[4];
    des_from_IK_sec3.alpha  = des_config_info.angular_velocity_covariance[5];
    des_from_IK_sec3.beta   = des_config_info.angular_velocity_covariance[6];
    des_from_IK_sec4.alpha  = des_config_info.angular_velocity_covariance[7];
    des_from_IK_sec4.beta   = des_config_info.angular_velocity_covariance[8];
    
    // 强制调试输出
    static int callback_count = 0;
    if (callback_count++ % 50 == 0) {  // 每50次回调打印一次
        ROS_ERROR("C++接收到角度回调 #%d: sec1(%.1f,%.1f) sec2(%.1f,%.1f) sec3(%.1f,%.1f) sec4(%.1f,%.1f)", 
                  callback_count,
                  des_from_IK_sec1.alpha, des_from_IK_sec1.beta,
                  des_from_IK_sec2.alpha, des_from_IK_sec2.beta,
                  des_from_IK_sec3.alpha, des_from_IK_sec3.beta,
                  des_from_IK_sec4.alpha, des_from_IK_sec4.beta);
    }
}

sensor_msgs::Imu cur_config_info;
void cur_config_cb(const sensor_msgs::Imu::ConstPtr& config) 
{
    cur_config_info = *config;
 
    // mcu:
    cur_from_IMUs_sec1.alpha  = cur_config_info.linear_acceleration_covariance[0];
    cur_from_IMUs_sec1.beta   = cur_config_info.linear_acceleration_covariance[1];
    cur_from_IMUs_sec2.alpha  = cur_config_info.linear_acceleration_covariance[2];
    cur_from_IMUs_sec2.beta   = cur_config_info.linear_acceleration_covariance[4];
    // mcu:
}


int main( int argc, char** argv )
{
  
    ros::init(argc, argv, "forward_kinematics");
    ros::NodeHandle nh;
    ros::Rate rate(10000.0);

    //----------- subscribe configuration space
    ros::Subscriber des_config_sub      = nh.subscribe<sensor_msgs::Imu>("des_config_space", 10, des_config_cb);
    ros::Subscriber cur_config_sub      = nh.subscribe<sensor_msgs::Imu>("cur_config_space", 10, cur_config_cb);
    //----------- subscribe configuration space

    //----------- visualization publisher
    ros::Publisher robotFrame_pub1   = nh.advertise<visualization_msgs::Marker>( "robotFrame1", 0 );
    ros::Publisher robotFrame_pub2   = nh.advertise<visualization_msgs::Marker>( "robotFrame2", 0 );
    ros::Publisher robotFrame_pub3   = nh.advertise<visualization_msgs::Marker>( "robotFrame3", 0 );

	ros::Publisher arm_section1      = nh.advertise<visualization_msgs::MarkerArray>("arm_section1", 10);
	ros::Publisher arm_section2      = nh.advertise<visualization_msgs::MarkerArray>("arm_section2", 10);
	ros::Publisher arm_section3      = nh.advertise<visualization_msgs::MarkerArray>("arm_section3", 10);
	ros::Publisher arm_section4      = nh.advertise<visualization_msgs::MarkerArray>("arm_section4", 10);
    //----------- visualization publisher

    //----------- angle publisher
    ros::Publisher sec1_beta_pub     = nh.advertise<std_msgs::Float32>("sec1/beta", 10);
    //----------- angle publisher

    //----------- path publisher
    ros::Publisher path_ee_pub = nh.advertise<nav_msgs::Path>("path_ee",1, true);
    ros::Publisher path_gt_pub = nh.advertise<nav_msgs::Path>("path_gt",1, true);

    ros::Time current_time;

    nav_msgs::Path path_ee;
    path_ee.header.frame_id="base_link";
    path_ee.header.stamp=current_time;

    nav_msgs::Path path_gt;
    path_gt.header.frame_id="base_link";
    path_gt.header.stamp=current_time;
    //----------- path publisher


 
    float cnt = 0, rmse_sum = 0;

    tf::TransformBroadcaster tf_broadcaster;
    float t = 0;
    // Moving demonstration:
    while (ros::ok())
    {  
        /* visualize IK solver's configuration space */
        sec1.alpha = des_from_IK_sec1.alpha;
        sec1.beta  = des_from_IK_sec1.beta;
        sec2.alpha = des_from_IK_sec2.alpha;
        sec2.beta  = des_from_IK_sec2.beta;
        sec3.alpha = des_from_IK_sec3.alpha;
        sec3.beta  = des_from_IK_sec3.beta;
        sec4.alpha = des_from_IK_sec4.alpha;
        sec4.beta  = des_from_IK_sec4.beta;
        
        // Debug: 打印接收到的角度值
        static int debug_counter = 0;
        if (debug_counter++ % 100 == 0) {  // 每100次循环打印一次
            ROS_INFO("C++接收到的角度: sec1(%.1f,%.1f) sec2(%.1f,%.1f) sec3(%.1f,%.1f) sec4(%.1f,%.1f)", 
                     sec1.alpha, sec1.beta, sec2.alpha, sec2.beta, 
                     sec3.alpha, sec3.beta, sec4.alpha, sec4.beta);
        }

        // Fallback to hardcoded values if no GNN data received
        if (sec1.alpha == 0 && sec1.beta == 0 && sec2.alpha == 0 && sec2.beta == 0 && 
            sec3.alpha == 0 && sec3.beta == 0 && sec4.alpha == 0 && sec4.beta == 0) {
            sec1.alpha = 90;  sec1.beta  = 180;
            sec2.alpha = 60;  sec2.beta  = 0;
            sec3.alpha = 60;  sec3.beta  = 0;
            sec4.alpha = 60;  sec4.beta  = 0;
        }

        /* visualize IK solver's configuration space */
 
        /*  arm parameter  */
        float arc_Length = 1.8;  // 分米单位 (1.8分米=0.18米)
        int resolution = 10;
        /*  arm parameter  */

        /* world frame -> arm base frame */
        Eigen::Vector3f base_position(0,0,0);

        Eigen::Matrix3f Rot_world_base;
        Rot_world_base << 1, 0, 0,
                          0, 1, 0,
                          0, 0, 1;
        /* world frame -> arm base frame */


        /*  world frame -> arm end1 frame  */
        float alpha_1 = sec1.alpha / 180.0 * M_PI, beta_1 = sec1.beta / 180.0 * M_PI;
        Eigen::Matrix3f  Rot_base_end1 = Rot_world_base * config_to_rotationMatrix(alpha_1, beta_1);
        Eigen::Vector3f  Pos_base_end1 = base_position + Rot_world_base * config_to_translationMatrix(alpha_1, beta_1, arc_Length);
        Eigen::Quaternionf q_base_end1(Rot_base_end1);
 
        tf::Vector3 origin_1 (Pos_base_end1(0,0), Pos_base_end1(1,0), Pos_base_end1(2,0));
        tf::Quaternion rotation_1 (q_base_end1.x(), q_base_end1.y(), q_base_end1.z(), q_base_end1.w());
        tf::Transform t_1 (rotation_1, origin_1);
        tf::StampedTransform end_1_Marker (t_1, ros::Time::now(), "/base_link", "enddisk_1");
        tf_broadcaster.sendTransform(end_1_Marker);

        visualization_msgs::MarkerArray sec1_backbone = backbone_generation(1, resolution, alpha_1, beta_1, arc_Length, Rot_world_base, base_position);
        arm_section1.publish( sec1_backbone );
        // disk_visualization(1, resolution, alpha_1, beta_1, arc_Length, Rot_world_base, base_position);
        /*  world frame -> arm end1 frame  */


        /*  world frame -> arm end2 frame  */
        float alpha_2 = sec2.alpha / 180.0 * M_PI, beta_2 = sec2.beta / 180.0 * M_PI;
        Eigen::Matrix3f  Rot_base_end2 = Rot_base_end1  * config_to_rotationMatrix(alpha_2, beta_2);
        Eigen::Vector3f  Pos_base_end2 = Pos_base_end1 + Rot_base_end1 * config_to_translationMatrix(alpha_2, beta_2, arc_Length);
        Eigen::Quaternionf q_base_end2(Rot_base_end2);

        tf::Vector3 origin_2 (Pos_base_end2(0,0), Pos_base_end2(1,0), Pos_base_end2(2,0));
        tf::Quaternion rotation_2 (q_base_end2.x(), q_base_end2.y(), q_base_end2.z(), q_base_end2.w());
        tf::Transform t_2 (rotation_2, origin_2);
        tf::StampedTransform end_2_Marker (t_2, ros::Time::now(), "/base_link", "enddisk_2");
        tf_broadcaster.sendTransform(end_2_Marker);

        visualization_msgs::MarkerArray sec2_backbone = backbone_generation(2, resolution, alpha_2, beta_2, arc_Length, Rot_base_end1, Pos_base_end1);
        arm_section2.publish( sec2_backbone );
        // disk_visualization(2, resolution, alpha_2, beta_2, arc_Length, Rot_base_end1, Pos_base_end1);
        /*  world frame -> arm end2 frame  */


        /*  world frame -> arm end3 frame  */
        float alpha_3 = sec3.alpha / 180.0 * M_PI, beta_3 = sec3.beta / 180.0 * M_PI;
        Eigen::Matrix3f  Rot_base_end3 = Rot_base_end2  * config_to_rotationMatrix(alpha_3, beta_3);
        Eigen::Vector3f  Pos_base_end3 = Pos_base_end2 + Rot_base_end2 * config_to_translationMatrix(alpha_3, beta_3, arc_Length);

        visualization_msgs::MarkerArray sec3_backbone = backbone_generation(3, resolution, alpha_3, beta_3, arc_Length, Rot_base_end2, Pos_base_end2);
        arm_section3.publish( sec3_backbone );
        // disk_visualization(2, resolution, alpha_2, beta_2, arc_Length, Rot_base_end1, Pos_base_end1);
        /*  world frame -> arm end2 frame  */

        /*  world frame -> arm end4 frame  */
        float alpha_4 = sec4.alpha / 180.0 * M_PI, beta_4 = sec4.beta / 180.0 * M_PI;
        Eigen::Matrix3f  Rot_base_end4 = Rot_base_end3  * config_to_rotationMatrix(alpha_4, beta_4);
        Eigen::Vector3f  Pos_base_end4 = Pos_base_end3 + Rot_base_end3 * config_to_translationMatrix(alpha_4, beta_4, arc_Length);

        visualization_msgs::MarkerArray sec4_backbone = backbone_generation(4, resolution, alpha_4, beta_4, arc_Length, Rot_base_end3, Pos_base_end3);
        arm_section4.publish( sec4_backbone );
        // disk_visualization(2, resolution, alpha_2, beta_2, arc_Length, Rot_base_end1, Pos_base_end1);
        /*  world frame -> arm end2 frame  */








        // 发布sec1.beta角度
        std_msgs::Float32 beta_msg;
        beta_msg.data = sec1.beta;
        sec1_beta_pub.publish(beta_msg);

          
        ROS_INFO( "sec1.alpha %f  sec1.beta %f ", sec1.alpha, sec1.beta);
        ROS_INFO( "sec2.alpha %f  sec2.beta %f ", sec2.alpha, sec2.beta);
        ROS_INFO( "sec3.alpha %f  sec3.beta %f ", sec3.alpha, sec3.beta);
        ROS_INFO( "sec4.alpha %f  sec4.beta %f \n", sec4.alpha, sec4.beta);

        robot_worldFrame(robotFrame_pub1, robotFrame_pub2, robotFrame_pub3);
        ros::spinOnce();
        rate.sleep();
    }
}

Eigen::Vector3f config_to_translationMatrix(float alpha, float beta, float arcLength)
{
    // alpha can't be exactly 0;
    if(alpha == 0)
        alpha = 0.000001;

    float x_temp = arcLength/alpha * (1-cos(alpha)) * sin(beta);
    float y_temp = arcLength/alpha * (1-cos(alpha)) * cos(beta);
    float z_temp = arcLength/alpha * sin(alpha);
    Eigen::Vector3f pos_in_this_enddisk_frame(x_temp, y_temp, z_temp);

    return pos_in_this_enddisk_frame;
}


Eigen::Matrix3f config_to_rotationMatrix(float alpha, float beta)
{
    Eigen::Matrix3f Rot;
    Rot << cos(beta)*cos(beta)*(1-cos(alpha)) + cos(alpha), -cos(beta)*sin(beta)*(1-cos(alpha)), 			 sin(alpha)*sin(beta),
           -cos(beta)*sin(beta)*(1-cos(alpha))   		   ,  sin(beta)*sin(beta)*(1-cos(alpha)) + cos(alpha), sin(alpha)*cos(beta),
           -sin(alpha)*sin(beta)						   , -sin(alpha)*cos(beta),  						 cos(alpha)         ;
    return Rot;
}

visualization_msgs::MarkerArray backbone_generation(int id, int resolution, float alpha, float beta, float arcLength, Eigen::Matrix3f last_Rot, Eigen::Vector3f last_enddisk_pos) 
{
    // alpha can't be exactly 0;
    if(alpha == 0)
        alpha = 0.000001;

    visualization_msgs::MarkerArray section_backbone;

    for(int i = 0; i < resolution; i++)
    {
        float sample_length = i * arcLength / (resolution - 1);
        float sample_alpha  = i * alpha     / (resolution - 1);
        float sample_beta    = beta ;
        
        float x_temp = sample_length / sample_alpha * (1-cos(sample_alpha)) * sin(sample_beta);
        float y_temp = sample_length / sample_alpha * (1-cos(sample_alpha)) * cos(sample_beta);
        float z_temp = sample_length / sample_alpha * sin(sample_alpha);

        Eigen::Vector3f pos_sample_local_frame(x_temp, y_temp, z_temp);
        Eigen::Vector3f pos_sample_base_frame;
        pos_sample_base_frame = last_enddisk_pos + last_Rot * pos_sample_local_frame;

        ////////////////////

        visualization_msgs::Marker sample_point;
	    sample_point.header.frame_id = "base_link";
	    sample_point.header.stamp = ros::Time::now();
	    sample_point.ns = "basic_shapes";
	    sample_point.id = i;
	    sample_point.type   = visualization_msgs::Marker::SPHERE;
	    sample_point.action = visualization_msgs::Marker::ADD;
	    sample_point.scale.x = .2;
	    sample_point.scale.y = .2;
	    sample_point.scale.z = .2;
	    sample_point.color.a = 1.0;

        if(id == 1)
        {sample_point.color.r = 0.0; sample_point.color.b = 0.0; sample_point.color.g = 1.0;}
    
        if(id == 2)
        {sample_point.color.r = 0.0; sample_point.color.b = 1.0; sample_point.color.g = 0.0;}

        if(id == 3)
        {sample_point.color.r = 1.0; sample_point.color.b = 0.0; sample_point.color.g = 0.0;}

        if(id == 4)
        {sample_point.color.r = 0.5; sample_point.color.b = 0.5; sample_point.color.g = 0.5;}

	    sample_point.lifetime = ros::Duration();

        sample_point.pose.position.x = pos_sample_base_frame.x();
        sample_point.pose.position.y = pos_sample_base_frame.y();
        sample_point.pose.position.z = pos_sample_base_frame.z();
        sample_point.pose.orientation.x = 0;  
        sample_point.pose.orientation.y = 0; 
        sample_point.pose.orientation.z = 0;   
        sample_point.pose.orientation.w = 1; 

        section_backbone.markers.push_back(sample_point);
    }

    return section_backbone;
}


void disk_visualization(int id, int resolution, float alpha, float beta, float arcLength, Eigen::Matrix3f last_Rot, Eigen::Vector3f last_enddisk_pos) 
{
    // alpha can't be exactly 0;
    if(alpha == 0)
        alpha = 0.000001;

    for(int i = 1; i <= 8; i++)
    {
        
        float sample_length = i * arcLength / 8.0;  // 从5.0改为8.0，使关节分布更密集
        float sample_alpha  = i * alpha     / 8.0;  // 从5.0改为8.0，使关节分布更密集
        float sample_beta    = beta ;
        
        float x_temp = sample_length / sample_alpha * (1-cos(sample_alpha)) * sin(sample_beta);
        float y_temp = sample_length / sample_alpha * (1-cos(sample_alpha)) * cos(sample_beta);
        float z_temp = sample_length / sample_alpha * sin(sample_alpha);

        Eigen::Vector3f pos_sample_local_frame(x_temp, y_temp, z_temp);
        Eigen::Vector3f pos_sample_base_frame;
        pos_sample_base_frame = last_enddisk_pos + last_Rot * pos_sample_local_frame;

        Eigen::Matrix3f  Rot =  last_Rot * config_to_rotationMatrix(sample_alpha, sample_beta);
        Eigen::Quaternionf q(Rot);

        tf::Vector3 origin (pos_sample_base_frame.x(), pos_sample_base_frame.y(), pos_sample_base_frame.z());
        tf::Quaternion rotation (q.x(), q.y(), q.z(), q.w());
        tf::Transform t (rotation, origin);

        // // // 添加特殊的位置修正
        // tf::Vector3 origin_corrected = origin;

        // // 添加修正偏移量
        // origin_corrected.setX(origin.x() + -0.15);  // X方向偏移
        // origin_corrected.setY(origin.y() + -0.15);  // Y方向偏移  
        // origin_corrected.setZ(origin.z() + -0.0); // Z方向偏移

        if(id == 1)
        {
 
            char disk_FrameName[30];
            tf::TransformBroadcaster tf_broadcaster;

            tf::Vector3 origin_base (last_enddisk_pos.x() + 0.15, last_enddisk_pos.y() + 0.15, last_enddisk_pos.z()-0.0);
            Eigen::Quaternionf q_base(last_Rot);
            tf::Quaternion rotation_base (q_base.x(), q_base.y(), q_base.z(), q_base.w());
            tf::Transform t_base (rotation_base, origin_base);

            sprintf(disk_FrameName, "S%dL%d", 0, 0);
            tf::StampedTransform base_Marker (t_base, ros::Time::now(), "/base_link", disk_FrameName);
            tf_broadcaster.sendTransform(base_Marker);

            sprintf(disk_FrameName, "S%dL%d", id-1, i);
            tf::StampedTransform disk_Marker (t, ros::Time::now(), "/base_link", disk_FrameName);
            tf_broadcaster.sendTransform(disk_Marker);
        }    
        if(id == 2)
        {
            char disk_FrameName[30];
            tf::TransformBroadcaster tf_broadcaster;
            

            sprintf(disk_FrameName, "S%dL%d", id-1, i-1);
            tf::StampedTransform disk_Marker (t, ros::Time::now(), "/base_link", disk_FrameName);
            tf_broadcaster.sendTransform(disk_Marker);
        }
    }
}

void robot_worldFrame(ros::Publisher robotFrame_pub1, ros::Publisher robotFrame_pub2, ros::Publisher robotFrame_pub3)
{
    geometry_msgs::Point marker1_p1, marker1_p2;
    geometry_msgs::Point marker2_p1, marker2_p2;
    geometry_msgs::Point marker3_p1, marker3_p2;
    visualization_msgs::Marker marker1;
    visualization_msgs::Marker marker2;
    visualization_msgs::Marker marker3;

    // robotFrame_pub1  
    marker1.header.frame_id = "base_link";
    marker1.header.stamp = ros::Time();
    marker1.ns = "my_namespace";
    marker1.id = 0;
    marker1.type = visualization_msgs::Marker::ARROW;
    marker1.action = visualization_msgs::Marker::ADD;
 
    marker1_p1.x = + 0.15;
    marker1_p1.y = 0;
    marker1_p1.z = 0;
    marker1_p2.x = + 0.15;
    marker1_p2.y = 1;
    marker1_p2.z = 0;
    marker1.points.push_back(marker1_p1) ;
    marker1.points.push_back(marker1_p2) ;

    marker1.scale.x = 0.1;
    marker1.scale.y = 0.1;
    marker1.scale.z = 0.1;
    marker1.color.a = 1.0; // Don't forget to set the alpha!
    marker1.color.r = 1.0;
    marker1.color.g = 0.0;
    marker1.color.b = 0.0;
    robotFrame_pub1.publish( marker1 );

    /////////////////////////////////////////////////

    // robotFrame_pub2  
    marker2.header.frame_id = "base_link";
    marker2.header.stamp = ros::Time();
    marker2.ns = "my_namespace";
    marker2.id = 0;
    marker2.type = visualization_msgs::Marker::ARROW;
    marker2.action = visualization_msgs::Marker::ADD;
 
    marker2_p1.x = 0;
    marker2_p1.y = + 0.15;
    marker2_p1.z = 0;
    marker2_p2.x = 1;
    marker2_p2.y = + 0.15;
    marker2_p2.z = 0;
    marker2.points.push_back(marker2_p1) ;
    marker2.points.push_back(marker2_p2) ;

    marker2.scale.x = 0.1;
    marker2.scale.y = 0.1;
    marker2.scale.z = 0.1;
    marker2.color.a = 1.0; // Don't forget to set the alpha!
    marker2.color.r = 0.0;
    marker2.color.g = 0.0;
    marker2.color.b = 1.0;
    robotFrame_pub2.publish( marker2 );
    
    /////////////////////////////////////////////////
}





