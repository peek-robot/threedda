from setuptools import setup, find_packages

setup(
    name="problem_reduction",
    version="0.0.1",
    description="",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
        "mujoco",
        "shapely",
        "wandb",
        "wandb[media]",
        "meshcat",
        "trimesh",
        "h5py",
        "opencv-python",
        "matplotlib",
        "numpy<2"
    ],
)
