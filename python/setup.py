from distutils.core import setup

setup(name='dolfinx_cuas',

      version='0.1',

      description='Collection of custom assemblers using DOLFINx and Basix',

      author='Jørgen S. Dokken',

      author_email='dokken92@gmail.com',

      packages=['dolfinx_cuas', "dolfinx_cuas.contact"])
