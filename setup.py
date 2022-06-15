from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='cutoml',
    packages=['cutoml'],
    version='0.0.11',
    license='gpl-3.0',
    description='A lightweight automl library',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Omkar Udawant',
    author_email='omkarudawant97@gmail.com',
    url='https://github.com/omkarudawant/CutoML',
    download_url='https://github.com/omkarudawant/CutoML/archive/refs/tags/0.0.11.tar.gz',
    keywords=[
        'pipeline optimization',
        'automated hyperparameter optimization',
        'data science',
        'machine learning',
    ],
    install_requires=[
        'scipy==1.5.1',
        'numpy>=1.21',
        'scikit-learn==0.24.2',
        'pydantic==1.8.2',
        'xgboost==1.2.1',
        'tqdm==4.62.3',
        'pathos==0.2.8',
        'imbalanced-learn==0.8.1',
        'setuptools==58.0.3'
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
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.6',
)
