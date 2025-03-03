import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
     name='fractal_dimension',  
     version='1.0.0',
     author="Jack Featherstone",
     author_email="jack.featherstone@oist.jp",
     license='MIT',
     url='https://github.com/jfeatherstone/fractal_dimension',
     description="Techniques to approximate the fractal dimension of a set of points or trajectory.",
     long_description=long_description,
     long_description_content_type="text/markdown",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3.11",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )
