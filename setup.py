from setuptools import setup
import sys
import os
from stela import version
import json

data = [version.git_origin, version.git_hash, version.git_description, version.git_branch]
with open(os.path.join('stela', 'GIT_INFO'), 'w') as outfile:
    json.dump(data, outfile)

def package_files(package_dir, subdirectory):
    # walk the input package_dir/subdirectory
    # return a package_data list
    paths = []
    directory = os.path.join(package_dir, subdirectory)
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            path = path.replace(package_dir + '/', '')
            paths.append(os.path.join(path, filename))
    return paths
data_files = package_files('stela', 'data')

setup_args = {
    'name':         'stela',
    'author':       'StelaSquad',
    'url':          'https://github.com/StelaSquad/stela',
    'license':      'None',
    'version':      version.version,
    'description':  'STELA calculation code.',
    'packages':     ['stela'],
    'package_dir':  {'stela': 'stela'},
    'package_data': {'stela': data_files},
    'install_requires': ['numpy>=1.10', 'scipy>=0.19',],
    'include_package_data': True,
    'zip_safe':     False,
}

if __name__ == '__main__':
    apply(setup, (), setup_args)