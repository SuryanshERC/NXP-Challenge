#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    """
    Launch file for the NXP B3RB Warehouse Challenge.
    
    This launch file sets up all necessary nodes for the warehouse challenge including:
    - Navigation node for autonomous movement
    - Main controller for state machine and workflow orchestration
    - QR code decoder for shelf identification
    - Object recognition integration
    - Optional debugging nodes
    """
    
    # Package directories
    pkg_warehouse_challenge = get_package_share_directory('warehouse_challenge')
    pkg_b3rb_gz_bringup = get_package_share_directory('b3rb_gz_bringup')
    
    # Launch arguments
    warehouse_id_arg = DeclareLaunchArgument(
        'warehouse_id',
        default_value='1',
        description='Warehouse world ID (1-4)'
    )
    
    shelf_count_arg = DeclareLaunchArgument(
        'shelf_count',
        default_value='2',
        description='Number of shelves in the warehouse'
    )
    
    initial_angle_arg = DeclareLaunchArgument(
        'initial_angle',
        default_value='135.0',
        description='Initial heuristic angle to first shelf'
    )
    
    world_arg = DeclareLaunchArgument(
        'world',
        default_value='nxp_aim_india_2025/warehouse_1',
        description='Gazebo world to load'
    )
    
    x_arg = DeclareLaunchArgument(
        'x',
        default_value='0.0',
        description='Initial robot x position'
    )
    
    y_arg = DeclareLaunchArgument(
        'y',
        default_value='0.0',
        description='Initial robot y position'
    )
    
    yaw_arg = DeclareLaunchArgument(
        'yaw',
        default_value='0.0',
        description='Initial robot yaw orientation'
    )
    
    debug_mode_arg = DeclareLaunchArgument(
        'debug_mode',
        default_value='false',
        description='Enable debug visualization nodes'
    )
    
    use_gui_arg = DeclareLaunchArgument(
        'use_gui',
        default_value='false',
        description='Enable progress table GUI'
    )
    
    # Parameter file path
    config_file = os.path.join(pkg_warehouse_challenge, 'config', 'params.yaml')
    
    # Include B3RB simulation launch
    b3rb_sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('b3rb_gz_bringup'),
                'launch',
                'sil.launch.py'
            ])
        ]),
        launch_arguments={
            'world': LaunchConfiguration('world'),
            'warehouse_id': LaunchConfiguration('warehouse_id'),
            'shelf_count': LaunchConfiguration('shelf_count'),
            'initial_angle': LaunchConfiguration('initial_angle'),
            'x': LaunchConfiguration('x'),
            'y': LaunchConfiguration('y'),
            'yaw': LaunchConfiguration('yaw'),
        }.items()
    )
    
    # Navigation Node
    navigation_node = Node(
        package='warehouse_challenge',
        executable='navigation_node',
        name='navigation_controller',
        output='screen',
        parameters=[
            config_file,
            {
                'warehouse_id': LaunchConfiguration('warehouse_id'),
                'shelf_count': LaunchConfiguration('shelf_count'),
                'initial_angle': LaunchConfiguration('initial_angle'),
            }
        ],
        remappings=[
            ('/cmd_vel', '/cerebri/in/cmd_vel'),
            ('/odom', '/cerebri/out/odometry'),
        ]
    )
    
    # Main Controller Node (State Machine)
    main_controller_node = Node(
        package='warehouse_challenge',
        executable='main_controller',
        name='warehouse_controller',
        output='screen',
        parameters=[
            config_file,
            {
                'warehouse_id': LaunchConfiguration('warehouse_id'),
                'shelf_count': LaunchConfiguration('shelf_count'),
                'initial_angle': LaunchConfiguration('initial_angle'),
                'use_gui': LaunchConfiguration('use_gui'),
            }
        ]
    )
    
    # QR Code Decoder Node
    qr_decoder_node = Node(
        package='warehouse_challenge',
        executable='qr_decoder',
        name='qr_code_decoder',
        output='screen',
        parameters=[config_file],
        remappings=[
            ('/camera/image_raw/compressed', '/camera/image_raw/compressed'),
            ('/qr_code_data', '/qr_code_data'),
        ]
    )
    
    # Object Recognition Node (Enhanced YOLO processing)
    object_recognition_node = Node(
        package='warehouse_challenge',
        executable='object_recognizer',
        name='object_recognition_processor',
        output='screen',
        parameters=[config_file],
        remappings=[
            ('/shelf_objects', '/shelf_objects'),
            ('/processed_shelf_objects', '/processed_shelf_objects'),
        ]
    )
    
    # Shelf Detection Node
    shelf_detector_node = Node(
        package='warehouse_challenge',
        executable='shelf_detector',
        name='shelf_detector',
        output='screen',
        parameters=[config_file],
        remappings=[
            ('/map', '/map'),
            ('/global_costmap/costmap', '/global_costmap/costmap'),
            ('/detected_shelves', '/detected_shelves'),
        ]
    )
    
    # Debug Visualization Node (conditional)
    debug_visualization_node = Node(
        package='warehouse_challenge',
        executable='debug_visualizer',
        name='debug_visualizer',
        output='screen',
        parameters=[config_file],
        condition=IfCondition(LaunchConfiguration('debug_mode')),
        remappings=[
            ('/debug_images/qr_code', '/debug_images/qr_code'),
            ('/debug_images/object_recog', '/debug_images/object_recog'),
            ('/debug_images/shelf_detection', '/debug_images/shelf_detection'),
        ]
    )
    
    # Map Visualizer Node (conditional)
    map_visualizer_node = Node(
        package='warehouse_challenge',
        executable='map_visualizer',
        name='map_visualizer',
        output='screen',
        parameters=[config_file],
        condition=IfCondition(LaunchConfiguration('debug_mode'))
    )
    
    return LaunchDescription([
        # Launch arguments
        warehouse_id_arg,
        shelf_count_arg,
        initial_angle_arg,
        world_arg,
        x_arg,
        y_arg,
        yaw_arg,
        debug_mode_arg,
        use_gui_arg,
        
        # Include B3RB simulation
        b3rb_sim_launch,
        
        # Core challenge nodes
        navigation_node,
        main_controller_node,
        qr_decoder_node,
        object_recognition_node,
        shelf_detector_node,
        
        # Debug nodes (conditional)
        debug_visualization_node,
        map_visualizer_node,
    ])
