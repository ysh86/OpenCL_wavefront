#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

//
// ---+----+----+
//   B| C C| D  |
//   1| 2 3| 4  |
// ---+----+----+
//   A|1234|
//   0| +a0|
// ---+----+
//

#define W 1024
#define H 1024
#define LW 16
#define LH 16
#define LW_SHIFT 4
#define LH_SHIFT 4

__kernel void pred(
    __global unsigned char *yPlane
)
{
    int gx = get_global_id(0);
    int gy = get_global_id(1);
    int lx = get_local_id(0);
    int ly = get_local_id(1);

    yPlane[W * gy + gx] = (gx >> LW_SHIFT) + (gy >> LH_SHIFT);
}
