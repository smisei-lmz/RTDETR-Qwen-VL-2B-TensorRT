from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    defaults = {
        'source': 'synthetic', 'detector': 'color', 'weights': 'PekingU/rtdetr_v2_r18vd',
        'backend': 'mock', 'model': 'Qwen/Qwen2-VL-2B-Instruct',
        'device': 'cuda:0', 'detector_device': '0', 'mock_delay_s': '2.0',
        'auto_ask': 'true', 'timeout_s': '15.0', 'max_answer_age_s': '20.0',
        'inspection_enabled': 'false', 'occupied_s': '2.0', 'absent_s': '10.0',
        'workstation_roi': '0,0,1,1', 'report_dir': 'runs/workstation',
    }
    args = [DeclareLaunchArgument(k, default_value=v) for k, v in defaults.items()]
    def arg(name, kind=str):
        return ParameterValue(LaunchConfiguration(name), value_type=kind)
    return LaunchDescription(args + [
        Node(package='edge_assistant', executable='perception', output='screen', parameters=[{
            'source': arg('source'), 'detector': arg('detector'), 'weights': arg('weights'),
            'device': arg('detector_device')}]),
        Node(package='edge_assistant', executable='vlm_server', output='screen', parameters=[{
            'backend': arg('backend'), 'model': arg('model'), 'device': arg('device'),
            'mock_delay_s': arg('mock_delay_s', float)}]),
        Node(package='edge_assistant', executable='coordinator', output='screen', parameters=[{
            'auto_ask': arg('auto_ask', bool), 'timeout_s': arg('timeout_s', float),
            'max_answer_age_s': arg('max_answer_age_s', float),
            'inspection_enabled': arg('inspection_enabled', bool),
            'occupied_s': arg('occupied_s', float), 'absent_s': arg('absent_s', float),
            'workstation_roi': arg('workstation_roi'), 'report_dir': arg('report_dir')}]),
    ])
