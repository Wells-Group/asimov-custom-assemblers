import os
import platform
import subprocess
import sys
import sysconfig
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


VERSION = "0.3.1"

REQUIREMENTS = ["fenics-dolfinx@https://github.com/FEniCS/dolfinx/"]


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the"
                               + "following extensions:"
                               + ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            raise RuntimeError("Windows not supported")
        for ext in self.extensions:
            self.build_extension(ext)


class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: "
                               + ", ".join(e.name for e in self.extensions))

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPython3_EXECUTABLE=' + sys.executable,
                      '-DPython3_LIBRARIES=' + sysconfig.get_config_var("LIBDEST"),
                      '-DPython3_INCLUDE_DIRS=' + sysconfig.get_config_var("INCLUDEPY")]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
        build_args += ['--', '-j3']

        env = os.environ.copy()
        import pybind11
        env['pybind11_DIR'] = pybind11.get_cmake_dir()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp, env=env)


setup(name='dolfinx_cuas',

      version=VERSION,

      description='Collection of custom assemblers using DOLFINx and Basix',

      author='Jørgen S. Dokken',

      author_email='dokken92@gmail.com',
      python_requires='>3.6.0',
      packages=['dolfinx_cuas'],
      ext_modules=[CMakeExtension('dolfinx_cuas.cpp')],
      cmdclass=dict(build_ext=CMakeBuild),
      install_requires=REQUIREMENTS,
      zip_safe=False)
