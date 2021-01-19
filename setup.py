from distutils.core import setup

setup(
    name='cutoml',
    packages=['cutoml'],
    version='0.0.1',
    license='gpl-3.0',
    description='A lightweight automl framework for text '
                'classification/regression.',
    author='Omkar Udawant',
    author_email='omkarudawant97@gmail.com',
    url='https://github.com/omkarudawant/CutoML',
    download_url='https://github.com/omkarudawant/CutoML/archive/0.0.1.tar.gz',
    keywords=[
        'automl',
        'pipeline optimization',
        'hyperparameter optimization',
        'data science',
        'machine learning',
    ],
    install_requires=[
        'scipy',
        'numpy',
        'joblib',
        'scikit-learn',
        'pandas',
        'pydantic',
        'xgboost'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: GNU Lesser General Public License v3 ('
        'LGPLv3)',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
