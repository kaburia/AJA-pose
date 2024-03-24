from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import sys
import os

class BuildExt(build_ext):
    def run(self):
        os.chdir('aja_pose/lib')
        os.system('make')
        build_ext.run(self)


with open("README.md", "r") as fh:
    long_description = fh.read()

# get the required packages from requirements.txt
with open('requirements.txt') as f:
    required = f.read().splitlines()

# make the nms package available at code\hrnet\lib\dataset


setup(
    name='aja_pose',
    version='0.0.1',
    packages=find_packages(),
    include_package_data=True,
    description='Animal pose estimation using Vision Transformers and HRNet(VHR)',
    author='Austin Kaburia, Joan Kabura, Antony Gitau',
    author_email='kaburiaaustin1@gmail.com, joankabura1@gmail.com, antonym.gitau9@gmail.com',
    url = 'https://github.com/Antony-gitau/2024_ICME_Challenge',
    install_requires=required,
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires='>=3.8',
    ext_modules=[Extension('my_module', [])],
    cmdclass={'build_ext': BuildExt},
)
