Unified Virtual Memory Lite (UVM-lite) From NVVM IR
===================================================

This document is for the programming language and compiler
implementers who target NVVM IR and plan to support Unified Virtual
Memory Lite (UVM-lite) in their language.  It provides the low-level
details related to supporting kernel launches at the NVVM IR level.

This document assumes the CUDA runtime is used. For the limits and
restrictions, please refer to the official CUDA documents.

Allocating a variable in the unified virtual memory environment.
----------------------------------------------------------------

In a system that supports unified virtual memory environment, a
variable can be allocated at a location where host and other devices
in the system can reference the variable directly. We call such a
variable a managed variable and say that the variable has the
managed attribute or is managed. The attribute can be specified
using a metadata.

    @xxx = internal addrspace(1) global i32 10, align 4

    ...

    !1 = !{i32 addrspace(1)* @xxx, !"managed", i32 1}

A global variable, e.g., @xxx, can be defined and used as usual, but
here we have a metadata that specifies the managed attributes. (Note
that the attribute can only be used with variables in the global
address space.)

Accessing a managed variable in the host
---------------------------------------- 

To access a managed variable defined in the NVVM IR code, we should
retrieve a device pointer first, which can be done using cuModuleGetGlobal().

    CUdeviceptr devp_xxx; // device pointer to xxx
    size_t      size_xxx; // size of xxx
    result = cuModuleGetGlobal(&devp_xxx, &size_xxx, hModule, "xxx");

Whether or not the pointer points to managed memory may be queried
by calling cuPointerGetAttribute() with the pointer attribute
CU_POINTER_ATTRIBUTE_IS_MANAGED.

    unsigned int attrVal;
    result = cuPointerGetAttribute(&attrVal, CU_POINTER_ATTRIBUTE_IS_MANAGED, devp_xxx);
    // result will be 1 if the pointer is managed or zero otherwise.

It is safe to use cuPointerGetAttribute to get the host pointers,
since the device pointers are opaque.

    void *host_ptr_xxx;
    int *p_xxx;

    result = cuPointerGetAttribute(&host_ptr_xxx, CU_POINTER_ATTRIBUTE_HOST_POINTER, devp_xxx);
    p_xxx = (int *)devp_xxx;
    *p_xxx += 1;   // read & write without explicit copying
