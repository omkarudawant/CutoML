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
    download_url='https://github.com/user/reponame/archive/v_01.tar.gz',
    keywords=[
        'automl',
        'machine learning',
        'ml pipelines',
        'automated '
        'machine learning'
    ],
    install_requires=[
        'scipy',
        'numpy',
        'joblib',
        'scikit-learn'
        'pandas',
        'pydantic',
        'xgboost'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: GNU General Public License v3.0',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
