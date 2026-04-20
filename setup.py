from setuptools import find_packages, setup

setup(
    name="acs-planner-nuplan",
    version="0.1.0",
    description="Action-conditioned symbolic support planner for nuPlan",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy>=1.23",
        "PyYAML>=6.0",
        "torch>=2.0",
        "tqdm>=4.64",
    ],
)
