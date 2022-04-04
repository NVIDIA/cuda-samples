#!/usr/bin/env python
from string import *
import os, getopt, sys, platform

g_Header = '''/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

////////////////////////////////////////////////////////////////////////////////
// This file is auto-generated, do not edit
////////////////////////////////////////////////////////////////////////////////
'''


def Usage():
    print("Usage: ptx2c.py in out")
    print("Description: performs embedding in.cubin or in.ptx file into out.c and out.h files as character array")
    sys.exit(0)


def FormatCharHex(d):
    s = hex(ord(d))
    if len(s) == 3:
        s = "0x0" + s[2]
    return s


args = sys.argv[1:]
if not(len(sys.argv[1:]) == 2):
    Usage()

out_h = args[1] + "_ptxdump.h"
out_c = args[1] + "_ptxdump.c"


h_in = open(args[0], 'r')
source_bytes = h_in.read()
source_bytes_len = len(source_bytes)

h_out_c = open(out_c, 'w')
h_out_c.writelines(g_Header)
h_out_c.writelines("#include \"" + out_h + "\"\n\n")
h_out_c.writelines("unsigned char " + args[1] + "_ptxdump[" + str(source_bytes_len+1) + "] = {\n")

h_out_h = open(out_h, 'w')
macro_h = "__" + args[1] + "_ptxdump_h__"
h_out_h.writelines(g_Header)
h_out_h.writelines("#ifndef " + macro_h + "\n")
h_out_h.writelines("#define " + macro_h + "\n\n")
h_out_h.writelines('#if defined __cplusplus\nextern "C" {\n#endif\n\n')
h_out_h.writelines("extern unsigned char " + args[1] + "_ptxdump[" + str(source_bytes_len+1) + "];\n\n")
h_out_h.writelines("#if defined __cplusplus\n}\n#endif\n\n")
h_out_h.writelines("#endif //" + macro_h + "\n")

newlinecnt = 0
for i in range(0, source_bytes_len):
    h_out_c.write(FormatCharHex(source_bytes[i]) + ", ")
    newlinecnt += 1
    if newlinecnt == 16:
        newlinecnt = 0
        h_out_c.write("\n")
h_out_c.write("0x00\n};\n")

h_in.close()
h_out_c.close()
h_out_h.close()

print("ptx2c: CUmodule " + args[0] + " packed successfully")
