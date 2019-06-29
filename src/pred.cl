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

#if __OPENCL_VERSION__ >= 200
__global atomic_int conds2_0[(W >> LW_SHIFT)*(H >> LH_SHIFT)];
#endif

__kernel void pred(
    __global uchar *yPlane,
#if SYNC_ON_DEV
    volatile __global int *conds,
    const __global int *orders
#else
    int xth,
    int offsetY
#endif
)
{
#if SYNC_ON_DEV
    const int lastGroupX = get_num_groups(0) - 1;

    const int order = *(orders + (W >> LW_SHIFT) * get_group_id(1) + get_group_id(0));
    const int groupY = order / (W >> LW_SHIFT);
    const int groupX = order - (W >> LW_SHIFT) * groupY;

    const int localX = get_local_id(0);
    const int localY = get_local_id(1);
    const int globalX = (groupX << LW_SHIFT) + localX;
    const int globalY = (groupY << LH_SHIFT) + localY;

#if __OPENCL_VERSION__ >= 200
    __global atomic_int *condCur = conds2_0 + (W >> LW_SHIFT) * groupY + groupX;
    __global atomic_int *condA = (groupX != 0) ? condCur - 1 : NULL;
    __global atomic_int *condB = NULL;
    __global atomic_int *condC = NULL;
    __global atomic_int *condD = NULL;
    if (groupY != 0) {
        condC = condCur - (W >> LW_SHIFT);
        condB = (groupX != 0) ? condC - 1 : condC;
        condD = (groupX != lastGroupX) ? condC + 1 : condC;
    }

    if (localX == 0 && localY == 0) {
        if (condA) {
            while (atomic_load_explicit(condA, memory_order_acquire, memory_scope_device) == 0)
                ;
        }
        if (condB) {
            while (atomic_load_explicit(condB, memory_order_acquire, memory_scope_device) == 0)
                ;
        }
        if (condC) {
            while (atomic_load_explicit(condC, memory_order_acquire, memory_scope_device) == 0)
                ;
        }
        if (condD) {
            while (atomic_load_explicit(condD, memory_order_acquire, memory_scope_device) == 0)
                ;
        }
    }
    work_group_barrier(0);
#else
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
#endif
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

#if __OPENCL_VERSION__ >= 200
    work_group_barrier(CLK_LOCAL_MEM_FENCE);
#else
    barrier(CLK_LOCAL_MEM_FENCE);
    //mem_fence(CLK_GLOBAL_MEM_FENCE|CLK_LOCAL_MEM_FENCE);
#endif

    uchar value = (a0[localY] + b1c2c3d4[localX]) >> 1;
    yPlane[W * globalY + globalX] = value;

#if SYNC_ON_DEV
#if __OPENCL_VERSION__ >= 200
    // too slow!
    work_group_barrier(CLK_GLOBAL_MEM_FENCE);

    if (localX == 0 && localY == 0) {
        //if (localX == (LW - 1) && localY == (LH - 1)) {
        atomic_store_explicit(condCur, 1, memory_order_release, memory_scope_device);
    }
#else
    // too slow!
    barrier(CLK_GLOBAL_MEM_FENCE);
    //mem_fence(CLK_GLOBAL_MEM_FENCE/*|CLK_LOCAL_MEM_FENCE*/);

    if (localX == 0 && localY == 0) {
    //if (localX == (LW - 1) && localY == (LH - 1)) {
        atomic_xchg(condCur, 1);
    }
#endif
#endif
}
