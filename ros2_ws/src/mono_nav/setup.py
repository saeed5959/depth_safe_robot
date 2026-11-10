from setuptools import find_packages, setup

package_name = 'mono_nav'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='saeed',
    maintainer_email='s.firouzi@speax.ai',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={'console_scripts': [
    'mono_depth = mono_nav.mono_depth:main',
    'compare_depth = mono_nav.compare_depth:main',
    'odom_to_tf = mono_nav.odom_to_tf:main',
    'mono_depth_safe = mono_nav.mono_depth_safe:main',
    ]},
)
