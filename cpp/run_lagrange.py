import os

forms = ["a = inner(u, v) * dx"]
degrees = [1, 2, 3, 4, 5, 6]
nrepeats = 5
out_file = "out.txt"

for form in forms:
    for degree in degrees:
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
        os.system(f"cd build && cmake -DCMAKE_BUILD_TYPE=Release -DELEMENT_DEGREE={degree} .. && make")

        for i in range(nrepeats):
            os.system(f"./build/lagrange >>{out_file}")
