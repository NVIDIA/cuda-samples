NVIDIA CUDA Sample "radixSortThrust"
----------------------------------

--------
OVERVIEW
--------

This sample demonstrates a very fast and efficient parallel radix sort implemented in C for CUDA.  The included RadixSort class can sort either key-value pairs (with float or unsigned integer keys) or keys only. It can also sort unsigned integer keys based on a varying number of least-significant bits ranging from 4 to 32 in multiples of 4.

This radix sort code and the underlying algorithm is discussed in detail in the paper "Designing Efficient Sorting Algorithms for Manycore GPUs".  A PDF version of this paper is available at http://mgarland.org/files/papers/gpusort-ipdps09.pdf

-----
USAGE
-----

To run a sort with default options (Sort 1M unsigned integer key-value pairs), just invoke the executable ("radixSort.exe" on Windows, "radixSort" otherwise).

The following command line options are available:

 -n=<N>        : number of elements to sort
 -keysonly     : sort only an array of keys (the default is to sort key-value pairs)
 -float        : use 32-bit float keys
 -keybits=<B>  : Use only the B least-significant bits of the keys for the sort
               : B must be a multiple of 4.  This option does not apply to float keys
 -quiet        : Output only the number of elements and the time to sort
 -help         : Output a help message

The RadixSort class can also be used within your application by building the radixsort.cu file into your application or library, and including the radixsort.h header file.

--------
CITATION
--------

Satish, N., Harris, M., and Garland, M. "Designing Efficient Sorting 
Algorithms for Manycore GPUs". In Proceedings of IEEE International
Parallel & Distributed Processing Symposium 2009 (IPDPS 2009).

PDF:

http://mgarland.org/files/papers/gpusort-ipdps09.pdf
