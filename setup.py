from setuptools import setup, find_packages


setup(
    name='streaming-stt-nemo',
    version="0.0.1",
    packages=find_packages(),
    python_requires='>3.7.0',
    install_requires=[
        "nemo_toolkit[asr]",
    ],
)