    #include <rclcpp/rclcpp.hpp>
    #include <sensor_msgs/msg/point_cloud2.hpp>
    #include <sensor_msgs/msg/laser_scan.hpp>
    #include <sensor_msgs/point_cloud2_iterator.hpp>
    #include <vector>
    #include <cmath>
    #include <limits>
    #include <algorithm>

    class PointCloudToLaserScanNode : public rclcpp::Node
    {
    public:
        PointCloudToLaserScanNode() : Node("pointcloud_to_laserscan"), msg_count_(0)
        {
            RCLCPP_INFO(get_logger(), "🔄 PointCloud2 to LaserScan converter initialized");

            // =========================
            // PARAMETERS
            // =========================
            // CAMBIO CLAVE: Altura específica del plano que quieres escanear
            this->declare_parameter<double>("min_height", 0.0);  // Altura mínima del plano
            this->declare_parameter<double>("max_height", 0.25);  // Altura máxima del plano (grosor del "slice")
            
            // Parámetros angulares (cobertura completa 360°)
            this->declare_parameter<double>("angle_min", -M_PI);
            this->declare_parameter<double>("angle_max", M_PI);
            this->declare_parameter<double>("angle_increment", 0.25 * M_PI / 180.0); // 0.25° (mejor resolución)
            
            // Rango de distancias
            this->declare_parameter<double>("range_min", 0.05);
            this->declare_parameter<double>("range_max", 30.0);
            
            // Offset de rotación (si el LiDAR está orientado diferente)
            this->declare_parameter<double>("rotation_offset", 0.0);
            
            // NUEVO: Frame del LaserScan
            this->declare_parameter<std::string>("target_frame", "base_link");
            
            // NUEVO: Filtro de outliers
            this->declare_parameter<double>("outlier_threshold", 5.0); // Eliminar puntos > 5m si mayoría están más cerca
            
            this->get_parameter("min_height", min_height_);
            this->get_parameter("max_height", max_height_);
            this->get_parameter("angle_min", angle_min_);
            this->get_parameter("angle_max", angle_max_);
            this->get_parameter("angle_increment", angle_increment_);
            this->get_parameter("range_min", range_min_);
            this->get_parameter("range_max", range_max_);
            this->get_parameter("rotation_offset", rotation_offset_);
            this->get_parameter("target_frame", target_frame_);
            this->get_parameter("outlier_threshold", outlier_threshold_);

            // Calculate number of beams
            num_beams_ = static_cast<int>((angle_max_ - angle_min_) / angle_increment_) + 1;

            // =========================
            // PUBLISHER & SUBSCRIBER
            // =========================
            pointcloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
                "/unilidar/cloud", 10,
                std::bind(&PointCloudToLaserScanNode::pointcloudCallback, this, std::placeholders::_1));

            laserscan_pub_ = this->create_publisher<sensor_msgs::msg::LaserScan>("/scan", 10);

            RCLCPP_INFO(get_logger(), "📡 Subscribed to: /unilidar/cloud");
            RCLCPP_INFO(get_logger(), "📤 Publishing to: /scan");
            RCLCPP_INFO(get_logger(), "🎯 Height filter: [%.2f, %.2f] m (grosor plano: %.2f m)", 
                        min_height_, max_height_, max_height_ - min_height_);
            RCLCPP_INFO(get_logger(), "🎯 Angle range: [%.1f°, %.1f°]", 
                        angle_min_ * 180.0 / M_PI, angle_max_ * 180.0 / M_PI);
            RCLCPP_INFO(get_logger(), "🎯 Number of beams: %d (res: %.2f°)", 
                        num_beams_, angle_increment_ * 180.0 / M_PI);
            RCLCPP_INFO(get_logger(), "🎯 Rotation offset: %.1f°", rotation_offset_ * 180.0 / M_PI);
            RCLCPP_INFO(get_logger(), "🎯 Target frame: %s", target_frame_.c_str());
        }

    private:
        void pointcloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg)
        {
            try
            {
                msg_count_++;

                // Create LaserScan message
                auto scan = sensor_msgs::msg::LaserScan();
                scan.header = cloud_msg->header;
                scan.header.frame_id = target_frame_;

                scan.angle_min = angle_min_;
                scan.angle_max = angle_max_;
                scan.angle_increment = angle_increment_;
                scan.time_increment = 0.0;
                scan.scan_time = 0.1; // Approximate
                scan.range_min = range_min_;
                scan.range_max = range_max_;

                // Initialize ranges to infinity (no detection)
                scan.ranges.resize(num_beams_, std::numeric_limits<float>::infinity());
                
                // NUEVO: Contador de puntos por beam para filtro de outliers
                std::vector<int> points_per_beam(num_beams_, 0);

                // =========================
                // PROCESS POINT CLOUD
                // =========================
                sensor_msgs::PointCloud2ConstIterator<float> iter_x(*cloud_msg, "x");
                sensor_msgs::PointCloud2ConstIterator<float> iter_y(*cloud_msg, "y");
                sensor_msgs::PointCloud2ConstIterator<float> iter_z(*cloud_msg, "z");

                int total_points = 0;
                int height_filtered = 0;
                int valid_beams = 0;
                int range_filtered = 0;

                for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z)
                {
                    float x = *iter_x;
                    float y = *iter_y;
                    float z = *iter_z;

                    // Skip invalid points
                    if (std::isnan(x) || std::isnan(y) || std::isnan(z))
                        continue;

                    total_points++;

                    // Filter by height (ESTE ES EL FILTRO CLAVE PARA EXTRAER EL PLANO)
                    if (z < min_height_ || z > max_height_)
                        continue;

                    height_filtered++;

                    // Calculate distance and angle
                    float distance = std::sqrt(x * x + y * y);
                    
                    // Skip if outside distance range
                    if (distance < range_min_ || distance > range_max_)
                    {
                        range_filtered++;
                        continue;
                    }

                    // Calculate angle with rotation offset
                    float angle = std::atan2(y, x) + rotation_offset_;

                    // Normalize angle to [-pi, pi]
                    angle = std::atan2(std::sin(angle), std::cos(angle));

                    // Skip if outside angular range
                    if (angle < angle_min_ || angle > angle_max_)
                        continue;

                    // Calculate beam index
                    int beam_index = static_cast<int>((angle - angle_min_) / angle_increment_);

                    // Ensure index is within bounds
                    if (beam_index >= 0 && beam_index < num_beams_)
                    {
                        // Keep minimum distance for each beam (closest obstacle)
                        if (distance < scan.ranges[beam_index])
                        {
                            scan.ranges[beam_index] = distance;
                            points_per_beam[beam_index]++;
                        }
                    }
                }

                // Count valid beams
                for (int i = 0; i < num_beams_; i++)
                {
                    if (std::isfinite(scan.ranges[i]))
                    {
                        valid_beams++;
                    }
                }

                // =========================
                // PUBLISH LASERSCAN
                // =========================
                // =========================
                // TEMPORAL SMOOTHING
                // =========================
                // Add current scan to circular buffer
                if (range_buffer_.empty())
                {
                    range_buffer_.resize(SMOOTH_N, std::vector<float>(num_beams_, std::numeric_limits<float>::infinity()));
                }
                range_buffer_[buffer_index_] = scan.ranges;
                buffer_index_ = (buffer_index_ + 1) % SMOOTH_N;
                if (buffer_index_ == 0) buffer_full_ = true;

                // Compute minimum per beam across buffer
                int n = buffer_full_ ? SMOOTH_N : buffer_index_;
                for (int i = 0; i < num_beams_; i++)
                {
                    float min_val = std::numeric_limits<float>::infinity();
                    for (int j = 0; j < n; j++)
                    {
                        if (range_buffer_[j][i] < min_val)
                            min_val = range_buffer_[j][i];
                    }
                    scan.ranges[i] = min_val;
                }

                laserscan_pub_->publish(scan);

                // Log statistics (every 30 messages to avoid spam)
                if (msg_count_ % 30 == 0)
                {
                    float min_dist = std::numeric_limits<float>::infinity();
                    for (const auto& r : scan.ranges)
                    {
                        if (std::isfinite(r) && r < min_dist)
                            min_dist = r;
                    }
                    
                    RCLCPP_INFO(get_logger(),
                        "✅ Total: %d | En plano [%.2f,%.2f]m: %d (%.1f%%) | Beams válidos: %d/%d | Min dist: %.2fm",
                        total_points, min_height_, max_height_, 
                        height_filtered, 
                        total_points > 0 ? (100.0 * height_filtered / total_points) : 0.0,
                        valid_beams, num_beams_,
                        std::isfinite(min_dist) ? min_dist : 0.0f
                    );
                }
            }
            catch (const std::exception &e)
            {
                RCLCPP_ERROR(get_logger(), "❌ Error processing point cloud: %s", e.what());
            }
        }

        rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;
        rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr laserscan_pub_;

        double min_height_, max_height_;
        double angle_min_, angle_max_, angle_increment_;
        double range_min_, range_max_;
        double rotation_offset_;
        double outlier_threshold_;
        std::string target_frame_;
        int num_beams_;
        int msg_count_;

        // Smoothing buffer
        static const int SMOOTH_N = 5;
        std::vector<std::vector<float>> range_buffer_;
        int buffer_index_ = 0;
        bool buffer_full_ = false;
    };

    int main(int argc, char *argv[])
    {
        rclcpp::init(argc, argv);
        auto node = std::make_shared<PointCloudToLaserScanNode>();

        RCLCPP_INFO(node->get_logger(), "🟢 Converter ready - Waiting for PointCloud2 data...");

        rclcpp::spin(node);

        RCLCPP_INFO(node->get_logger(), "🛑 Stopping converter node...");
        rclcpp::shutdown();
        return 0;
    }