import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nnc", # Replace with your own username
    version="0.0.1",
    author="Thomas Asikis",
    author_email="-",
    description="Neural Network control package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/asikist/nnc/',
    packages=setuptools.find_packages(),
    #packages=['nnc'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
