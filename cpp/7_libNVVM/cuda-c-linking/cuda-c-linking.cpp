// Copyright (c) 1993-2025, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <cassert>
#include <cuda.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/Config/llvm-config.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/Program.h>
#include <llvm/Support/raw_ostream.h>
#if LLVM_VERSION_MAJOR >= 21
#include <llvm/TargetParser/Triple.h>
#endif
#include <memory>
#include <nvvm.h>
#include <string>

#include "DDSWriter.h"

static_assert(sizeof(void *) == 8, "Only 64bit targets are supported.");
using namespace llvm;

static cl::opt<bool> SaveCubin("save-cubin", cl::desc("Write linker cubin to disk"), cl::init(false));
static cl::opt<bool> SaveIR("save-ir", cl::desc("Write LLVM IR to disk"), cl::init(false));
static cl::opt<bool> SavePTX("save-ptx", cl::desc("Write PTX to disk"), cl::init(false));

// Width and height of the output image.
const unsigned width  = 1024;
const unsigned height = 512;

// If 'err' is non-zero, emit an error message and exit.
#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)
static void __checkCudaErrors(CUresult err, const char *filename, int line)
{
    assert(filename);
    if (CUDA_SUCCESS != err) {
        const char    *ename = NULL;
        const CUresult res   = cuGetErrorName(err, &ename);
        fprintf(stderr,
                "CUDA API Error %04d: \"%s\" from file <%s>, "
                "line %i.\n",
                err,
                ((CUDA_SUCCESS == res) ? ename : "Unknown"),
                filename,
                line);
        // Exit with EXIT_FAILURE rather than the error code, which 'main'
        // reserves to report an unmet requirement.
        exit(EXIT_FAILURE);
    }
}

// Verify that the NVVM result code is success, or terminate otherwise.
void checkNVVMCall(nvvmResult res)
{
    if (res != NVVM_SUCCESS) {
        errs() << "libnvvm call failed\n";
        exit(EXIT_FAILURE);
    }
}

// The libNVVM architecture string for a compute capability, e.g. 10.0 becomes
// "compute_100".
std::string computeArch(int devMajor, int devMinor) { return "compute_" + utostr(devMajor) + utostr(devMinor); }

/// generateModule - Generate and LLVM IR module that calls an
/// externally-defined function
std::unique_ptr<Module> generateModule(LLVMContext &context)
{
    // Create the module and setup the layout and triple.
    auto mod = std::make_unique<Module>("nvvm-module", context);
    mod->setDataLayout("e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-"
                       "f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:"
                       "64");
#if LLVM_VERSION_MAJOR >= 21
    // LLVM 21 dropped the overload that takes the triple as a string.
    mod->setTargetTriple(Triple("nvptx64-nvidia-cuda"));
#else
    mod->setTargetTriple("nvptx64-nvidia-cuda");
#endif

    // Get pointers to some commonly-used types.
    Type *voidTy = Type::getVoidTy(context);
#if LLVM_VERSION_MAJOR >= 17
    // Pointers are opaque from LLVM 15 on, and LLVM 17 dropped the overload
    // that takes a pointee type. The resulting IR spells this type 'ptr'.
    Type *floatGenericPtrTy = PointerType::get(context, /* address space */ 0);
#else
    Type *floatGenericPtrTy = PointerType::get(Type::getFloatTy(context), /* address space */ 0);
#endif

    // void @mandelbrot(float*)
    Type          *mandelbrotParamTys[] = {floatGenericPtrTy};
    FunctionType  *mandelbrotTy         = FunctionType::get(voidTy, mandelbrotParamTys, false);
    FunctionCallee mandelbrotFunc       = mod->getOrInsertFunction("mandelbrot", mandelbrotTy);

    // Kernel argument types.
    Type *paramTys[] = {floatGenericPtrTy};

    // Kernel function type.
    FunctionType *funcTy = FunctionType::get(voidTy, paramTys, false);

    // Kernel function.
    Function *func = Function::Create(funcTy, GlobalValue::ExternalLinkage, "kernel", *mod);
    func->arg_begin()->setName("ptr");

    // 'entry' basic block in kernel function.
    BasicBlock *entry = BasicBlock::Create(context, "entry", func);

    // Build the entry block.
    IRBuilder<> builder(entry);
    builder.CreateCall(mandelbrotFunc, func->arg_begin());
    builder.CreateRetVoid();

    // Create kernel metadata.
    Metadata    *mdVals[]  = {ValueAsMetadata::get(func),
                              MDString::get(context, "kernel"),
                              ConstantAsMetadata::get(ConstantInt::getTrue(context))};
    MDNode      *kernelMD  = MDNode::get(context, mdVals);
    NamedMDNode *nvvmAnnot = mod->getOrInsertNamedMetadata("nvvm.annotations");
    nvvmAnnot->addOperand(kernelMD);

    // Set the NVVM IR version to 2.0.
    auto        *two           = ConstantInt::get(Type::getInt32Ty(context), 2);
    auto        *zero          = ConstantInt::get(Type::getInt32Ty(context), 0);
    auto        *versionMD     = MDNode::get(context, {ConstantAsMetadata::get(two), ConstantAsMetadata::get(zero)});
    NamedMDNode *nvvmIRVersion = mod->getOrInsertNamedMetadata("nvvmir.version");
    nvvmIRVersion->addOperand(versionMD);

    return mod;
}

// Use libNVVM to compile an NVVM IR module to PTX.
std::string generatePtx(const std::string &module, const std::string &arch, const char *moduleName)
{
    assert(moduleName);

    // libNVVM initialization.
    nvvmProgram compileUnit;
    checkNVVMCall(nvvmCreateProgram(&compileUnit));

    // Create a libNVVM compilation unit from the NVVM IR.
    checkNVVMCall(nvvmAddModuleToProgram(compileUnit, module.c_str(), module.size(), moduleName));
    std::string computeArg = "-arch=" + arch;

    // Compile the NVVM IR into PTX.
    const char *options[] = {computeArg.c_str()};
    nvvmResult  res       = nvvmCompileProgram(compileUnit, 1, options);
    if (res != NVVM_SUCCESS) {
        errs() << "nvvmCompileProgram failed!\n";
        size_t logSize;
        nvvmGetProgramLogSize(compileUnit, &logSize);
        char *msg = new char[logSize];
        nvvmGetProgramLog(compileUnit, msg);
        errs() << msg << "\n";
        delete[] msg;
        exit(EXIT_FAILURE);
    }

    // Get the result PTX size and source.
    size_t ptxSize = 0;
    checkNVVMCall(nvvmGetCompiledResultSize(compileUnit, &ptxSize));
    char *ptx = new char[ptxSize];
    checkNVVMCall(nvvmGetCompiledResult(compileUnit, ptx));

    // Clean-up libNVVM.
    checkNVVMCall(nvvmDestroyProgram(&compileUnit));

    return std::string(ptx);
}

int main(int argc, char **argv)
{
    cl::ParseCommandLineOptions(argc, argv, "cuda-c-linking");

    // Locate the pre-built library.
    std::string      libpath0 = sys::fs::getMainExecutable(argv[0], (void *)main);
    SmallString<256> libpath(libpath0);
    const char      *mathlibFile = "libmathfuncs64.a";
    sys::path::remove_filename(libpath);
    sys::path::append(libpath, mathlibFile);

    if (!sys::fs::exists(libpath.c_str())) {
        errs() << "Unable to locate math library, expected at " << libpath << '\n';
        return EXIT_FAILURE;
    }

    outs() << "Using math library: " << libpath.str() << "\n";

    // Initialize CUDA and obtain device 0.
    checkCudaErrors(cuInit(0));
    int nDevices;
    checkCudaErrors(cuDeviceGetCount(&nDevices));
    if (nDevices == 0) {
        errs() << "Failed to locate any CUDA compute devices.\n";
        exit(EXIT_FAILURE);
    }
    CUdevice device;
    checkCudaErrors(cuDeviceGet(&device, 0));

    char name[128];
    checkCudaErrors(cuDeviceGetName(name, 128, device));
    outs() << "Using CUDA Device [0]: " << name << "\n";

    int devMajor = 0, devMinor = 0;
    checkCudaErrors(cuDeviceGetAttribute(&devMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
    checkCudaErrors(cuDeviceGetAttribute(&devMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
    outs() << "Device Compute Capability: " << devMajor << "." << devMinor << "\n";
    if (devMajor < 7 || (devMajor == 7 && devMinor < 5)) {
        outs() << "Device 0 is not sm_75 or later.\n";
        // 2 reports an unmet hardware/toolchain requirement rather than a failure.
        return 2;
    }

    const std::string arch = computeArch(devMajor, devMinor);

    // The LLVM IR that libNVVM accepts depends on the target architecture:
    // pre-Blackwell targets take LLVM 7 IR, which spells pointers with their
    // pointee type ('float*'), while Blackwell and later take a much newer IR
    // that also accepts the opaque pointers ('ptr') LLVM 15 and later emit.
    // Ask libNVVM which one applies to this device rather than assuming.
    int              nvvmLLVMMajor = 0;
    const nvvmResult archSupported = nvvmLLVMVersion(arch.c_str(), &nvvmLLVMMajor);
    if (archSupported == NVVM_ERROR_INVALID_INPUT) {
        // The device is newer than the toolkit this sample builds against.
        outs() << "The libNVVM of this CUDA toolkit does not support " << arch
               << ". Use a toolkit that supports this device.\n";
        // 2 reports an unmet hardware/toolchain requirement rather than a failure.
        return 2;
    }
    checkNVVMCall(archSupported);
    if (LLVM_VERSION_MAJOR >= 15 && nvvmLLVMMajor < 15) {
        outs() << "This sample was built against LLVM " << LLVM_VERSION_MAJOR
               << ", which emits opaque pointers, but libNVVM accepts only LLVM " << nvvmLLVMMajor << " IR for " << arch
               << ". Build against LLVM 14 or older to run on this device, or run on a "
                  "Blackwell or later device.\n";
        // 2 reports an unmet hardware/toolchain requirement rather than a failure.
        return 2;
    }

    // Generate the IR module
    LLVMContext ctx;
    std::string moduleStr;
    auto        module = generateModule(ctx);

    if (SaveIR) {
        std::error_code err;
        raw_fd_ostream  out("cuda-c-linking.kernel.ll", err);
        out << *(module.get());
    }

    // Write the module to a string.
    {
        llvm::raw_string_ostream str(moduleStr);
        str << *module.get();
    }

    // Generate PTX.
    std::string ptx = generatePtx(moduleStr, arch, module->getModuleIdentifier().c_str());
    if (SavePTX) {
        std::error_code err;
        raw_fd_ostream  out("cuda-c-linking.kernel.ptx", err);
        out << ptx;
    }

    // Create the CUDA context.
    CUcontext context;
    checkCudaErrors(cuCtxCreate(&context, NULL, 0, device));

    // Create a JIT linker and generate the result CUBIN.
    CUlinkState  linker;
    char         linkerInfo[1024]{};
    char         linkerErrors[1024]{};
    CUjit_option linkerOptions[]      = {CU_JIT_INFO_LOG_BUFFER,
                                         CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
                                         CU_JIT_ERROR_LOG_BUFFER,
                                         CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
                                         CU_JIT_LOG_VERBOSE};
    void        *linkerOptionValues[] = {linkerInfo,
                                         reinterpret_cast<void *>(1024),
                                         linkerErrors,
                                         reinterpret_cast<void *>(1024),
                                         reinterpret_cast<void *>(1)};
    checkCudaErrors(cuLinkCreate(5, linkerOptions, linkerOptionValues, &linker));
    checkCudaErrors(
        cuLinkAddData(linker, CU_JIT_INPUT_PTX, (void *)ptx.c_str(), ptx.size(), "<compiled-ptx>", 0, NULL, NULL));
    checkCudaErrors(cuLinkAddFile(linker, CU_JIT_INPUT_LIBRARY, libpath.c_str(), 0, NULL, NULL));
    void  *cubin;
    size_t cubinSize;
    checkCudaErrors(cuLinkComplete(linker, &cubin, &cubinSize));
    outs() << "Linker Log:\n" << linkerInfo << "\n" << linkerErrors << "\n";
    if (SaveCubin) {
        std::error_code err;
        raw_fd_ostream  out("cuda-c-linking.linked.cubin", err, sys::fs::OF_None);
        out.write(reinterpret_cast<char *>(cubin), cubinSize);
    }

    // Create a module and load the cubin into it.
    CUmodule cudaModule;
    checkCudaErrors(cuModuleLoadDataEx(&cudaModule, cubin, 0, 0, 0));

    // Now that the CUBIN is loaded, we can release the linker.
    checkCudaErrors(cuLinkDestroy(linker));

    // Get kernel function.
    CUfunction function;
    checkCudaErrors(cuModuleGetFunction(&function, cudaModule, "kernel"));

    // Device data.
    CUdeviceptr devBuffer;
    checkCudaErrors(cuMemAlloc(&devBuffer, sizeof(float) * width * height * 4));
    float *data = new float[width * height * 4];

    // Each thread will generate one pixel, and we'll subdivide the problem into
    // 16x16 chunks.
    const unsigned blockSizeX = 16;
    const unsigned blockSizeY = 16;
    const unsigned blockSizeZ = 1;
    const unsigned gridSizeX  = width / 16;
    const unsigned gridSizeY  = height / 16;
    const unsigned gridSizeZ  = 1;

    // Execute the kernel.
    outs() << "Launching kernel\n";
    void *params[] = {&devBuffer};
    checkCudaErrors(cuLaunchKernel(
        function, gridSizeX, gridSizeY, gridSizeZ, blockSizeX, blockSizeY, blockSizeZ, 0, NULL, params, NULL));

    // Retrieve the result data from the device.
    checkCudaErrors(cuMemcpyDtoH(&data[0], devBuffer, sizeof(float) * width * height * 4));

    writeDDS("mandelbrot.dds", data, width, height);
    outs() << "Output saved to mandelbrot.dds\n";

    // Cleanup.
    delete[] data;
    checkCudaErrors(cuMemFree(devBuffer));
    checkCudaErrors(cuModuleUnload(cudaModule));
    checkCudaErrors(cuCtxDestroy(context));

    return 0;
}
