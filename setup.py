from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='cutoml',
    packages=['cutoml'],
    version='0.0.9.post12',
    license='gpl-3.0',
    description='A lightweight automl library',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Omkar Udawant',
    author_email='omkarudawant97@gmail.com',
    url='https://github.com/omkarudawant/CutoML',
    download_url='https://github.com/omkarudawant/CutoML/archive/0.0.9.post1.2.tar.gz',
    keywords=[
        'pipeline optimization',
        'automated hyperparameter optimization',
        'data science',
        'machine learning',
    ],
    install_requires=[
        'scipy>=1.5.1',
        'numpy>=1.19.1',
        'scikit-learn>=0.24.0',
        'xgboost<1.3.0',
        'tqdm>=4.56.0',
        'pathos>=0.2.7',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: GNU Lesser General Public License v3 ('
        'LGPLv3)',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    python_requires='>=3.6',
)
