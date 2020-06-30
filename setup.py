from setuptools import setup

#
setup(name='connector',
      version='1.0',
      description='Dataset connector',
      url='www.vniitf.ru',
      author=['A.L. Karmanov', 'N.A. Teplykh'],
      author_email=['karmanoval@sils.local', 'teplykhna@sils.local'],
      license='VNIITF',
      packages=['connector', 'connector.tools'],
      setup_requires=['matplotlib', 'pandas', 'tqdm', 'numpy', 'beautifulsoup4'],
      zip_safe=False)
