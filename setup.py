import setuptools

setuptools.setup(
    name="gabriel_lego",
    version="0.1.7",
    author="Manuel Olguin Munoz",
    author_email="molguin@kth.se",
    description="Gabriel LEGO Assembly cognitive engine.",
    long_description_content_type="text/markdown",
    url="https://github.com/cmusatyalab/gabriel-lego-py3",
    packages=setuptools.find_packages(exclude=['gabriel-server*']),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 2 - Pre-Alpha",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires='>=3.7',
    install_requires=[
        'opencv-contrib-python',
        'numpy',
        'dlib',
        'scikit-image',
        'protobuf'
    ]
)
