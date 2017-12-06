from setuptools import setup

setup(name='risk_averse_matching',
      version='0.1dev',
      packages=['risk_averse_matching',],
      long_description='Second iteration of bounded-variance matching module',
      license='All Rights Reserved',
      install_requires=[
            'numpy',
            'networkx',
            'powerlaw',
            'ipykernel',
            'ipython-autotime'
          ]
)
