from setuptools import setup

setup(name='connector',
      version='1.0',
      description='Пакет для работы с популярными датасетами',
      url='www.vniitf.ru',
      author=['A.L. Karmanov', 'N.A. Teplykh'],
      author_email=['karmanoval@sils.local', 'teplykhna@sils.local'],
      license='VNIITF',
      packages=['connector'],
      setup_requires=['matplotlib', 'pandas', 'tqdm', 'numpy'],
      zip_safe=False)
