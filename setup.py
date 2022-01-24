from setuptools import setup, find_packages

setup(
    name="stray",
    version="0.0.1",
    author="Stray Robots",
    author_email="hello@strayrobots.io",
    description="Stray Robots SDK",
    url="https://strayrobots.io",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "opencv-python",
        "trimesh",
        "pytorch-lightning",
        "torchvision",
        "pycocotools",
        "scipy"
    ],

    entry_points={
        'console_scripts': [
            'stray-train=stray.scripts.train:main',
            'stray-show=stray.scripts.show:main',
        ]
    }
)
