typedef float4 my4_t;

#define MAD_4(x, y)     x = mad(y, x, y);   y = mad(x, y, x);   x = mad(y, x, y);   y = mad(x, y, x);
#define MAD_16(x, y)    MAD_4(x, y);        MAD_4(x, y);        MAD_4(x, y);        MAD_4(x, y);
#define MAD_64(x, y)    MAD_16(x, y);       MAD_16(x, y);       MAD_16(x, y);       MAD_16(x, y);


__kernel void mad1024(
    __global const unsigned int *a,
    __global const unsigned int *b,
    __global unsigned int *c
)
{
    int gid = get_global_id(0);

    my4_t temp_a = {get_local_id(0), get_local_id(0)+1, get_local_id(0)+2, get_local_id(0)+3};
    my4_t temp_0 = {get_global_id(0), get_global_id(0)+1, get_global_id(0)+2, get_global_id(0)+3};

    // MAD 1024
    MAD_64(temp_a, temp_0);
    MAD_64(temp_a, temp_0);
    MAD_64(temp_a, temp_0);
    MAD_64(temp_a, temp_0);
    MAD_64(temp_a, temp_0);
    MAD_64(temp_a, temp_0);
    MAD_64(temp_a, temp_0);
    MAD_64(temp_a, temp_0);
    MAD_64(temp_a, temp_0);
    MAD_64(temp_a, temp_0);
    MAD_64(temp_a, temp_0);
    MAD_64(temp_a, temp_0);
    MAD_64(temp_a, temp_0);
    MAD_64(temp_a, temp_0);
    MAD_64(temp_a, temp_0);
    MAD_64(temp_a, temp_0);

    c[gid] = temp_0.s0 + temp_0.s1 + temp_0.s2 + temp_0.s3 + temp_a.s0 + temp_a.s1 + temp_a.s2 + temp_a.s3;
}
