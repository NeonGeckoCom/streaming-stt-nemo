from setuptools import setup, find_packages



with open("./version.py", "r", encoding="utf-8") as v:
    for line in v.readlines():
        if line.startswith("__version__"):
            if '"' in line:
                version = line.split('"')[1]
            else:
                version = line.split("'")[1]


setup(
    name='streaming-stt-nemo',
    version=version,
    packages=find_packages(),
    python_requires='>3.7.0',
    install_requires=[
        "nemo_toolkit[asr]",
    ],
)