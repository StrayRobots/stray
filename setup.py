from setuptools import setup, find_packages

setup(
    name="stray",
    version="0.0.1",
    author="Stray Robots",
    author_email="hello@strayrobots.io",
    description="Stray SDK",
    url="https://strayrobots.io",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        'trimesh',
        'scipy',
        'pillow',
        'numpy',
        'scikit-video',
        'pycocotools',
        'open3d'
    ],
)
