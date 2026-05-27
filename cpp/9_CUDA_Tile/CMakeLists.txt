# GCC auto-enables _FORTIFY_SOURCE at -O1 and above, which routes printf through
# a __host__ __device__ wrapper that tile kernels can't call. Turn it off for
# the tile category.
add_compile_options(
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-U_FORTIFY_SOURCE>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-D_FORTIFY_SOURCE=0>
)

add_subdirectory(helloTile)
add_subdirectory(tileVectorAdd)
add_subdirectory(tileTranspose)
add_subdirectory(tileMatmulAutotuner)
add_subdirectory(tileMatmul)
add_subdirectory(tileBmm)
add_subdirectory(tileLayerNorm)
add_subdirectory(tileRope)
add_subdirectory(tileSpMV)
