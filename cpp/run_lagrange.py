import os

forms = ["a = inner(grad(u), grad(v)) * dx"]
degrees = [1, 2, 3, 4]
nrepeats = 5
out_file = "results"

cxx_flags_list = ["\"-O3 \"", "\"-O3 -march=native \""]
c_flags_list = ["\"-O3 \"", "\"-O3 -march=native \""]
# compilers = [["g++", "gcc"], ["clang++", "clang"], ["g++-10", "gcc-10"], ["clang++-12", "clang-12"]]
compilers = [["g++", "gcc"], ["clang++-12", "clang-12"]]

for i in range(len(cxx_flags_list)):
    c_flags = c_flags_list[i]
    cxx_flags = cxx_flags_list[i]
    for compiler in compilers:
        for form in forms:
            for degree in degrees:
                os.environ["CXX"] = compiler[0]
                os.environ["CC"] = compiler[1]
                os.environ["PETSC_DIR"] = "/usr/local/petsc"
                os.environ["PETSC_ARCH"] = "linux-gnu-real-32"

                with open("lagrange.ufl", "r") as file:
                    lines = file.readlines()

                lines[0] = "degree = " + str(degree) + "\n"
                lines[8] = form

                with open("lagrange.ufl", "w") as file:
                    file.writelines(lines)

                os.system("ffcx lagrange.ufl")
                os.system("rm -rf build")
                os.system("mkdir build")
                os.system(
                    f"cd build && cmake -DCMAKE_BUILD_TYPE=Release -DELEMENT_DEGREE={degree} -DCMAKE_C_FLAGS={c_flags} -DCMAKE_CXX_FLAGS={cxx_flags} .. && make")

                for i in range(nrepeats):
                    os.system(f"./build/lagrange >>{out_file}.txt")
