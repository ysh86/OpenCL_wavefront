#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

// TODO: int mad24

//
// ---+----+----+
//   B| C C| D  |
//   1| 2 3| 4  |
// ---+----+----+
//   A|1234|
//   0| +a0|
// ---+----+
//

#define SYNC_ON_DEV 1

#define NULL 0

#define W 1024
#define H 1024
#define LW 16
#define LH 16
#define LW_SHIFT 4
#define LH_SHIFT 4

__kernel void pred(
    __global uchar *yPlane,
#if SYNC_ON_DEV
    volatile __global int *conds
#else
    int xth,
    int offsetY
#endif
)
{
#if SYNC_ON_DEV
    const int globalX = get_global_id(0);
    const int globalY = get_global_id(1);
    const int groupX = get_group_id(0);
    const int lastGroupX = get_num_groups(0) - 1;
    const int groupY = get_group_id(1);
    const int localX = get_local_id(0);
    const int localY = get_local_id(1);

    volatile __global int *condCur = conds + (W >> LW_SHIFT) * groupY + groupX;
    volatile __global int *condA = (groupX != 0) ? condCur - 1 : NULL;
    volatile __global int *condB = NULL;
    volatile __global int *condC = NULL;
    volatile __global int *condD = NULL;
    if (groupY != 0) {
        condC = condCur - (W >> LW_SHIFT);
        condB = (groupX != 0) ? condC - 1 : condC;
        condD = (groupX != lastGroupX) ? condC + 1 : condC;
    }

    if (localX == 0 && localY == 0) {
        if (condA) {
            while (atomic_cmpxchg(condA, 1, 1) == 0)
                ;
        }
        if (condB) {
            while (atomic_cmpxchg(condB, 1, 1) == 0)
                ;
        }
        if (condC) {
            while (atomic_cmpxchg(condC, 1, 1) == 0)
                ;
        }
        if (condD) {
            while (atomic_cmpxchg(condD, 1, 1) == 0)
                ;
        }
    }
    barrier(0);
#else
    /*const*/ int globalX = get_global_id(0);
    /*const*/ int globalY = get_global_id(1);
    //const int groupX = get_group_id(0);
    //const int lastGroupX = get_num_groups(0) - 1;
    /*const*/ int groupY = get_group_id(1);
    const int localX = get_local_id(0);
    const int localY = get_local_id(1);

    // -> xth
    // grpY0(0<<1): 0 1 2 3 4 5 6 7
    // grpY1(1<<1):     2 3 4 5 6 7 8 9
    // grpY2(2<<1):         4 5 6 7 8 9 10 11
    globalY += offsetY << LH_SHIFT;
    groupY += offsetY;
    const int groupX = xth - (groupY << 1);
    const int lastGroupX = (W >> LW_SHIFT) - 1;
    globalX += groupX << LW_SHIFT;
    if (groupX < 0 || groupX > lastGroupX) {
        return;
    }
#endif


    __global uchar *pgroup = yPlane + W * LH * groupY + LW * groupX;

    __local int a0[16];
    if (localX == 0) {
        a0[localY] = (globalX != 0) ? *(pgroup + W * localY - 1) + 1 : groupY;
    }

    __local int b1c2c3d4[16];
    if (globalY != 0) {
        if (localX < 4 && localY == 0) {
            uint *p;
            p = pgroup - (W + 4);
            p += localX << 1;
            uint add2110 = 2 - ((localX + 1) >> 1); // 00:2, 01:1, 10:1, 11:0
            if (globalX == 0) {
                p += 2;
                add2110 = 1;
            }
            if (localX == 3 && groupX == lastGroupX) {
                p -= 2;
                add2110 = 1;
            }
            uint value0123 = *p;
            b1c2c3d4[(localX << 2) + 0] = (value0123 >> 24) + add2110;
            b1c2c3d4[(localX << 2) + 1] = ((value0123 >> 16) & 0xff) + add2110;
            b1c2c3d4[(localX << 2) + 2] = ((value0123 >>  8) & 0xff) + add2110;
            b1c2c3d4[(localX << 2) + 3] = (value0123 & 0xff) + add2110;
        }
    } else {
        if (localY == 0) {
            b1c2c3d4[localX] = groupX;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    uchar value = (a0[localY] + b1c2c3d4[localX]) >> 1;
    yPlane[W * globalY + globalX] = value;

#if SYNC_ON_DEV
    // too slow!
    barrier(CLK_GLOBAL_MEM_FENCE);

    if (localX == 0 && localY == 0) {
        atomic_xchg(condCur, 1);
    }
#endif
}
