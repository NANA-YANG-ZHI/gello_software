import glob

from setuptools import find_packages, setup

package_name = "ur_move_to_start"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob.glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="gello",
    maintainer_email="noreply@example.com",
    description="Moves a UR arm driven by ur_robot_driver back to a start joint position.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "move_to_start = ur_move_to_start.move_to_start:main",
        ],
    },
)
