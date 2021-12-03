from skbuild import setup

import sys
import sysconfig
REQUIREMENTS = ["fenics-dolfinx>0.3.0@https://github.com/FEniCS/dolfinx/"]
setup(name="dolfinx_cuas",
      python_requires='>=3.7.0',
      version="0.3.1",
      description='DOLFINx custom assemblers',
      url="https://github.com/Wells-Group/asimov-custom-assemblers/",
      author='Sarah Roggendorf',
      maintainer="Jørgen S. Dokken",
      maintainer_email="dokken92@gmail.com",
      license="MIT",
      packages=['dolfinx_cuas'],
      cmake_args=[
          '-DPython3_EXECUTABLE=' + sys.executable,
          '-DPython3_LIBRARIES=' + sysconfig.get_config_var("LIBDEST"),
          '-DPython3_INCLUDE_DIRS=' + sysconfig.get_config_var("INCLUDEPY")],
      package_dir={"": "python"},
      cmake_install_dir="python/dolfinx_cuas/")
