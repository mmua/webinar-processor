#!/usr/bin/env python

"""Setup file"""

# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

with open('README.md', encoding="utf-8") as readme_file:
    readme = readme_file.read()

requirements = [
    'pytube==15.0.0',
    'Click==8.1.7',
    'pyannote.audio==3.1.1',
    'openai-whisper==20240930',
    'python_speech_features==0.6.0',
    'speechbrain==1.0.1',
    'python-dotenv==1.0.0',
    'requests==2.31.0',
    'openai==1.51.2',
    'spacy==3.6.1',
    'opencv-python==4.8.1.78',
    'moviepy==1.0.3',
    'tenacity==8.2.3'
]

setup(
    name='webinar_processor',
    version='0.9.1',
    description="HSE Organization & Project Management Webinar Processor",
    long_description=readme,
    author="Maxim Moroz",
    author_email='mimoroz@edu.hse.ru',
    url='https://github.com/mmua/webinar-processor',
    packages=find_packages(include=['webinar_processor', 'webinar_processor.*']),
    package_data={
        'webinar_processor': ['../conf/*', '../models/*'],  # include conf files in webinar package
    },
    entry_points={
        'console_scripts': [
            'webinar_processor=webinar_processor:cli'
        ]
    },
    include_package_data=True,
    install_requires=requirements,
    zip_safe=False,
    keywords='webinar_processor',
    classifiers=[
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.8',
    ]
)
