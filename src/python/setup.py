import pathlib

from setuptools import setup, find_packages


def get_version():
    path = pathlib.Path(__file__).absolute().parent / "bbn_medic" / "__init__.py"
    content = path.read_text()

    for line in content.splitlines():
        if line.startswith("__version__"):
            return line.strip().split()[-1].strip().strip('"')
    raise RuntimeError("bad version data in __init__.py")

setup(
    name='bbn_medic',
    version=get_version(),
    description="",
    long_description=open('../../README.md').read(),
    long_description_content_type='text/x-rst',
    packages=[
        package for package in find_packages()
            if package.startswith('bbn_medic')
    ],
    install_requires=[
    ],
    python_requires='>=3.10',
    package_data={
    },
    project_urls={
        #'Documentation': "",
        #'Source': "",
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
    ],
    zip_safe=False
)
