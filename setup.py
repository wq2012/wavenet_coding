"""Setup script for the WaveNet Low Rate Speech Coding package."""

import setuptools

VERSION = "0.2.1"

with open("README.md", "r", encoding="utf-8") as file_object:
  LONG_DESCRIPTION = file_object.read()

with open("requirements.txt", "r", encoding="utf-8") as file_object:
  INSTALL_REQUIRES = [
      line.strip()
      for line in file_object
      if line.strip() and not line.strip().startswith("#")
  ]

setuptools.setup(
    name="wavenet-coding",
    version=VERSION,
    author="Quan Wang",
    author_email="quanw@google.com",
    description="WaveNet Based Low Rate Speech Coding (Parametric & Waveform)",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/wq2012/wavenet_coding",
    packages=setuptools.find_packages(
        include=["wavenet_coding", "wavenet_coding.*"]
    ),
    scripts=[
        "scripts/prepare_data.py",
        "scripts/train.py",
        "scripts/evaluate.py",
        "scripts/export_model.py",
        "scripts/inference.py",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=INSTALL_REQUIRES,
)
