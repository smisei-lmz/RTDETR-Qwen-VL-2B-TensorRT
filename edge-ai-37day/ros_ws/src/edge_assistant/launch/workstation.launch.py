"""Real workstation inspection: Python camera/video or the C++ TRT pipeline."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    defaults = dict(use_cpp='false', source='0', weights='PekingU/rtdetr_v2_r18vd',
        pipeline_exe='', engine='', names_json='', producer_kind='pc',
        camera='/dev/video0', camera_mode='shared', camera_format='YUYV',
        camera_width='640', camera_height='480', matrix='601', full_range='false',
        vlm_backend='qwen2_vl', model='Qwen/Qwen2-VL-2B-Instruct',
        occupied_s='2.0', absent_s='10.0', workstation_roi='0,0,1,1',
        report_dir='runs/workstation', timeout_s='60.0', max_answer_age_s='90.0')
    value = LaunchConfiguration
    common = {k: value(k) for k in ('source', 'model', 'occupied_s', 'absent_s',
        'workstation_roi', 'report_dir', 'timeout_s', 'max_answer_age_s')}
    common.update(auto_ask='false', inspection_enabled='true')
    python_args = dict(common, detector='rtdetr', weights=value('weights'), backend=value('vlm_backend'))
    cpp_args = dict(common, backend='trt', vlm_backend=value('vlm_backend'))
    cpp_args.update({k:value(k) for k in ('pipeline_exe','engine','names_json','producer_kind',
        'camera','camera_mode','camera_format','camera_width','camera_height','matrix','full_range')})
    def launch_file(name):
        return PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare('edge_assistant'), 'launch', name]))
    return LaunchDescription([DeclareLaunchArgument(k, default_value=v) for k,v in defaults.items()] + [
        IncludeLaunchDescription(launch_file('demo.launch.py'), launch_arguments=python_args.items(),
                                 condition=UnlessCondition(value('use_cpp'))),
        IncludeLaunchDescription(launch_file('course.launch.py'), launch_arguments=cpp_args.items(),
                                 condition=IfCondition(value('use_cpp'))),
    ])
