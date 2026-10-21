from setuptools import setup
from glob import glob

setup(name='edge_assistant', version='0.1.0', packages=['edge_assistant'],
      data_files=[
          ('share/ament_index/resource_index/packages', ['resource/edge_assistant']),
          ('share/edge_assistant', ['package.xml']),
          ('share/edge_assistant/launch', glob('launch/*.launch.py')),
      ],
      install_requires=['setuptools'], zip_safe=True,
      maintainer='Project maintainer', maintainer_email='maintainer@example.invalid',
      description='Bounded ROS2 VLM pipeline demo', license='Apache-2.0',
      entry_points={'console_scripts': [
          'perception = edge_assistant.perception_node:main',
          'vlm_server = edge_assistant.vlm_server:main',
          'coordinator = edge_assistant.coordinator:main',
          'cpp_bridge = edge_assistant.cpp_bridge:main',
      ]})
