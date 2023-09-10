import codecs
import os.path

from setuptools import setup


# Version meaning (X.Y.Z)
# X: Major version (e.g. vastly different scene, platform, etc)
# Y: Minor version (e.g. new tasks, major changes to existing tasks, etc)
# Z: Patch version (e.g. small changes to tasks, bug fixes, etc)


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


core_requirements = [
    "numpy",
    "Pillow",
    "pyquaternion",
    "html-testRunner",
    "natsort"
]

setup(name='sg_rlbench',
      version=get_version("sg_rlbench/__init__.py"),
      description='RLBench skill grounding version',
      author='Stephen James',
      author_email='slj12@ic.ac.uk',
      url='https://www.doc.ic.ac.uk/~slj12',
      install_requires=core_requirements,
      packages=[
            'sg_rlbench',
            'sg_rlbench.backend',
            'sg_rlbench.action_modes',
            'sg_rlbench.tasks',
            'sg_rlbench.task_ttms',
            'sg_rlbench.robot_ttms',
            'sg_rlbench.sim2real',
            'sg_rlbench.assets',
            'sg_rlbench.gym'
      ],
      package_data={'': ['*.ttm', '*.obj', '**/**/*.ttm', '**/**/*.obj'],
                    'sg_rlbench': ['task_design.ttt']},
      )
