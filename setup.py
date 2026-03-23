from pathlib import Path

from setuptools import find_packages, setup


def read_requirements():
    requirements_path = Path(__file__).with_name("requirements.txt")
    if not requirements_path.exists():
        return []
    return [
        line.strip()
        for line in requirements_path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


setup(
    name="diffsynth",
    version="1.1.9",
    description="Enjoy the magic of Diffusion models!",
    author="Artiprocher",
    packages=find_packages(),
    install_requires=read_requirements(),
    extras_require={
        "npu_aarch64": [
            "torch==2.7.1",
            "torch-npu==2.7.1",
            "torchvision==0.22.1",
        ],
        "npu": [
            "torch==2.7.1+cpu",
            "torch-npu==2.7.1",
            "torchvision==0.22.1+cpu",
        ],
        "audio": [
            "torchaudio",
            "torchcodec",
        ],
    },
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    package_data={"diffsynth": ["tokenizer_configs/**/**/*.*"]},
    python_requires=">=3.6",
)
