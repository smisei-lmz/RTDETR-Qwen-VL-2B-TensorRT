from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    defaults={'pipeline_exe':'','producer_kind':'pc','backend':'cpu','source':'synthetic',
              'engine':'','names_json':'','camera':'/dev/video0','camera_mode':'shared',
              'camera_format':'YUYV','matrix':'601','snapshot_hz':'1.0',
              'camera_width':'640','camera_height':'480','full_range':'false',
              'vlm_backend':'mock','model':'Qwen/Qwen2-VL-2B-Instruct','mock_delay_s':'2.0',
              'auto_ask':'true','timeout_s':'15.0','max_answer_age_s':'20.0',
              'inspection_enabled':'false','occupied_s':'2.0','absent_s':'10.0',
              'workstation_roi':'0,0,1,1','report_dir':'runs/workstation'}
    def value(name,kind=str):return ParameterValue(LaunchConfiguration(name),value_type=kind)
    bridge={k:value(k) for k in ('pipeline_exe','producer_kind','backend','source','engine','names_json',
                                 'camera','camera_mode','camera_format')}
    bridge.update(matrix=value('matrix',int),snapshot_hz=value('snapshot_hz',float))
    bridge.update(camera_width=value('camera_width',int),camera_height=value('camera_height',int),full_range=value('full_range',bool))
    return LaunchDescription([DeclareLaunchArgument(k,default_value=v) for k,v in defaults.items()]+[
        Node(package='edge_assistant',executable='cpp_bridge',output='screen',parameters=[bridge]),
        Node(package='edge_assistant',executable='vlm_server',output='screen',parameters=[{
            'backend':value('vlm_backend'),'model':value('model'),'mock_delay_s':value('mock_delay_s',float)}]),
        Node(package='edge_assistant',executable='coordinator',output='screen',parameters=[{
            'auto_ask':value('auto_ask',bool),'timeout_s':value('timeout_s',float),
            'max_answer_age_s':value('max_answer_age_s',float),
            'inspection_enabled':value('inspection_enabled',bool),
            'occupied_s':value('occupied_s',float),'absent_s':value('absent_s',float),
            'workstation_roi':value('workstation_roi'),'report_dir':value('report_dir')}]),
    ])
