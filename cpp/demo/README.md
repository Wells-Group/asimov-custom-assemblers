To run this demo, use the following commands:

1. Compile ufl header file: `ffcx mass.ufl`
2. Build demo: `cmake -G Ninja -B build-dir -DCMAKE_BUILD_TYPE=Developer .`
3. Install demo: `ninja -C build-dir`
4. Run demo: `./build-dir/custom-assembler`