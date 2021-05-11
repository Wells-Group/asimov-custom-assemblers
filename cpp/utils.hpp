#include<dolfinx.h>
#include<basix.h>

namespace dolfinx_cuas {

int test_func(int a){
    int b = a + a;
    return b;
}
}