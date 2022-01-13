/*
 * Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.
 * Any use, reproduction, disclosure, or distribution of this software
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 */


//////////////////////////////////////////////////////////////////////////////
//
//  Copyright (c) 2009 Microsoft Corporation.  All rights reserved.
//
//  File:       D3DX11Effect.h
//  Content:    D3DX11 Effect Types & APIs Header
//
//////////////////////////////////////////////////////////////////////////////

#ifndef __D3DX11EFFECT_H__
#define __D3DX11EFFECT_H__

#include "d3d11.h"
#include "d3d11shader.h"

//////////////////////////////////////////////////////////////////////////////
// File contents:
//
// 1) Stateblock enums, structs, interfaces, flat APIs
// 2) Effect enums, structs, interfaces, flat APIs
//////////////////////////////////////////////////////////////////////////////

#ifndef D3DX11_BYTES_FROM_BITS
#define D3DX11_BYTES_FROM_BITS(x) (((x) + 7) / 8)
#endif // D3DX11_BYTES_FROM_BITS

typedef struct _D3DX11_STATE_BLOCK_MASK
{
    BYTE VS;
    BYTE VSSamplers[D3DX11_BYTES_FROM_BITS(D3D11_COMMONSHADER_SAMPLER_SLOT_COUNT)];
    BYTE VSShaderResources[D3DX11_BYTES_FROM_BITS(D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT)];
    BYTE VSConstantBuffers[D3DX11_BYTES_FROM_BITS(D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT)];
    BYTE VSInterfaces[D3DX11_BYTES_FROM_BITS(D3D11_SHADER_MAX_INTERFACES)];

    BYTE HS;
    BYTE HSSamplers[D3DX11_BYTES_FROM_BITS(D3D11_COMMONSHADER_SAMPLER_SLOT_COUNT)];
    BYTE HSShaderResources[D3DX11_BYTES_FROM_BITS(D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT)];
    BYTE HSConstantBuffers[D3DX11_BYTES_FROM_BITS(D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT)];
    BYTE HSInterfaces[D3DX11_BYTES_FROM_BITS(D3D11_SHADER_MAX_INTERFACES)];

    BYTE DS;
    BYTE DSSamplers[D3DX11_BYTES_FROM_BITS(D3D11_COMMONSHADER_SAMPLER_SLOT_COUNT)];
    BYTE DSShaderResources[D3DX11_BYTES_FROM_BITS(D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT)];
    BYTE DSConstantBuffers[D3DX11_BYTES_FROM_BITS(D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT)];
    BYTE DSInterfaces[D3DX11_BYTES_FROM_BITS(D3D11_SHADER_MAX_INTERFACES)];

    BYTE GS;
    BYTE GSSamplers[D3DX11_BYTES_FROM_BITS(D3D11_COMMONSHADER_SAMPLER_SLOT_COUNT)];
    BYTE GSShaderResources[D3DX11_BYTES_FROM_BITS(D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT)];
    BYTE GSConstantBuffers[D3DX11_BYTES_FROM_BITS(D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT)];
    BYTE GSInterfaces[D3DX11_BYTES_FROM_BITS(D3D11_SHADER_MAX_INTERFACES)];

    BYTE PS;
    BYTE PSSamplers[D3DX11_BYTES_FROM_BITS(D3D11_COMMONSHADER_SAMPLER_SLOT_COUNT)];
    BYTE PSShaderResources[D3DX11_BYTES_FROM_BITS(D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT)];
    BYTE PSConstantBuffers[D3DX11_BYTES_FROM_BITS(D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT)];
    BYTE PSInterfaces[D3DX11_BYTES_FROM_BITS(D3D11_SHADER_MAX_INTERFACES)];
    BYTE PSUnorderedAccessViews;

    BYTE CS;
    BYTE CSSamplers[D3DX11_BYTES_FROM_BITS(D3D11_COMMONSHADER_SAMPLER_SLOT_COUNT)];
    BYTE CSShaderResources[D3DX11_BYTES_FROM_BITS(D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT)];
    BYTE CSConstantBuffers[D3DX11_BYTES_FROM_BITS(D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT)];
    BYTE CSInterfaces[D3DX11_BYTES_FROM_BITS(D3D11_SHADER_MAX_INTERFACES)];
    BYTE CSUnorderedAccessViews;

    BYTE IAVertexBuffers[D3DX11_BYTES_FROM_BITS(D3D11_IA_VERTEX_INPUT_RESOURCE_SLOT_COUNT)];
    BYTE IAIndexBuffer;
    BYTE IAInputLayout;
    BYTE IAPrimitiveTopology;

    BYTE OMRenderTargets;
    BYTE OMDepthStencilState;
    BYTE OMBlendState;

    BYTE RSViewports;
    BYTE RSScissorRects;
    BYTE RSRasterizerState;

    BYTE SOBuffers;

    BYTE Predication;
} D3DX11_STATE_BLOCK_MASK;

//----------------------------------------------------------------------------
// D3DX11_EFFECT flags:
// -------------------------------------
//
// These flags are passed in when creating an effect, and affect
// the runtime effect behavior:
//
// (Currently none)
//
//
// These flags are set by the effect runtime:
//
// D3DX11_EFFECT_OPTIMIZED
//   This effect has been optimized. Reflection functions that rely on
//   names/semantics/strings should fail. This is set when Optimize() is
//   called, but CEffect::IsOptimized() should be used to test for this.
//
// D3DX11_EFFECT_CLONE
//   This effect is a clone of another effect. Single CBs will never be
//   updated when internal variable values are changed.
//   This flag is not set when the D3DX11_EFFECT_CLONE_FORCE_NONSINGLE flag
//   is used in cloning.
//
//----------------------------------------------------------------------------

#define D3DX11_EFFECT_OPTIMIZED                         (1 << 21)
#define D3DX11_EFFECT_CLONE                             (1 << 22)

// These are the only valid parameter flags to D3DX11CreateEffect*
#define D3DX11_EFFECT_RUNTIME_VALID_FLAGS (0)

//----------------------------------------------------------------------------
// D3DX11_EFFECT_VARIABLE flags:
// ----------------------------
//
// These flags describe an effect variable (global or annotation),
// and are returned in D3DX11_EFFECT_VARIABLE_DESC::Flags.
//
// D3DX11_EFFECT_VARIABLE_ANNOTATION
//   Indicates that this is an annotation on a technique, pass, or global
//   variable. Otherwise, this is a global variable. Annotations cannot
//   be shared.
//
// D3DX11_EFFECT_VARIABLE_EXPLICIT_BIND_POINT
//   Indicates that the variable has been explicitly bound using the
//   register keyword.
//----------------------------------------------------------------------------

#define D3DX11_EFFECT_VARIABLE_ANNOTATION              (1 << 1)
#define D3DX11_EFFECT_VARIABLE_EXPLICIT_BIND_POINT     (1 << 2)

//----------------------------------------------------------------------------
// D3DX11_EFFECT_CLONE flags:
// ----------------------------
//
// These flags modify the effect cloning process and are passed into Clone.
//
// D3DX11_EFFECT_CLONE_FORCE_NONSINGLE
//   Ignore all "single" qualifiers on cbuffers.  All cbuffers will have their
//   own ID3D11Buffer's created in the cloned effect.
//----------------------------------------------------------------------------

#define D3DX11_EFFECT_CLONE_FORCE_NONSINGLE             (1 << 0)

//----------------------------------------------------------------------------
// D3DX11_EFFECT_PASS flags:
// ----------------------------
//
// These flags modify the effect cloning process and are passed into Clone.
//
// D3DX11_EFFECT_PASS_COMMIT_CHANGES
//   This flag tells the effect runtime to assume that the device state was
//   not modified outside of effects, so that only updated state needs to
//   be set.
//
// D3DX11_EFFECT_PASS_OMIT_*
//   When applying a pass, do not set the state indicated in the flag name.
//----------------------------------------------------------------------------

#define D3DX11_EFFECT_PASS_COMMIT_CHANGES               (1 << 0)    // TODO: not yet implemented
#define D3DX11_EFFECT_PASS_OMIT_SHADERS_AND_INTERFACES  (1 << 1)    // TODO: not yet implemented
#define D3DX11_EFFECT_PASS_OMIT_STATE_OBJECTS           (1 << 2)    // TODO: not yet implemented
#define D3DX11_EFFECT_PASS_OMIT_RTVS_AND_DSVS           (1 << 3)    // TODO: not yet implemented
#define D3DX11_EFFECT_PASS_OMIT_SAMPLERS                (1 << 4)    // TODO: not yet implemented
#define D3DX11_EFFECT_PASS_OMIT_CBS                     (1 << 5)    // TODO: not yet implemented
#define D3DX11_EFFECT_PASS_OMIT_SRVS                    (1 << 6)    // TODO: not yet implemented
#define D3DX11_EFFECT_PASS_OMIT_UAVS                    (1 << 7)    // TODO: not yet implemented

#define D3DX11_EFFECT_PASS_ONLY_SET_SHADERS_AND_CBS   ( D3DX11_EFFECT_PASS_OMIT_STATE_OBJECTS | \
                                                        D3DX11_EFFECT_PASS_OMIT_RTVS_AND_DSVS | \
                                                        D3DX11_EFFECT_PASS_OMIT_SAMPLERS | \
                                                        D3DX11_EFFECT_PASS_OMIT_SRVS | \
                                                        D3DX11_EFFECT_PASS_OMIT_UAVS );

//////////////////////////////////////////////////////////////////////////////
// ID3DX11EffectType //////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

//----------------------------------------------------------------------------
// D3DX11_EFFECT_TYPE_DESC:
//
// Retrieved by ID3DX11EffectType::GetDesc()
//----------------------------------------------------------------------------

typedef struct _D3DX11_EFFECT_TYPE_DESC
{
    LPCSTR  TypeName;               // Name of the type
    // (e.g. "float4" or "MyStruct")

    D3D10_SHADER_VARIABLE_CLASS    Class;  // (e.g. scalar, vector, object, etc.)
    D3D10_SHADER_VARIABLE_TYPE     Type;   // (e.g. float, texture, vertexshader, etc.)

    UINT    Elements;               // Number of elements in this type
    // (0 if not an array)
    UINT    Members;                // Number of members
    // (0 if not a structure)
    UINT    Rows;                   // Number of rows in this type
    // (0 if not a numeric primitive)
    UINT    Columns;                // Number of columns in this type
    // (0 if not a numeric primitive)

    UINT    PackedSize;             // Number of bytes required to represent
    // this data type, when tightly packed
    UINT    UnpackedSize;           // Number of bytes occupied by this data
    // type, when laid out in a constant buffer
    UINT    Stride;                 // Number of bytes to seek between elements,
    // when laid out in a constant buffer
} D3DX11_EFFECT_TYPE_DESC;

typedef interface ID3DX11EffectType ID3DX11EffectType;
typedef interface ID3DX11EffectType *LPD3D11EFFECTTYPE;

// {4250D721-D5E5-491F-B62B-587C43186285}
DEFINE_GUID(IID_ID3DX11EffectType,
            0x4250d721, 0xd5e5, 0x491f, 0xb6, 0x2b, 0x58, 0x7c, 0x43, 0x18, 0x62, 0x85);

#undef INTERFACE
#define INTERFACE ID3DX11EffectType

DECLARE_INTERFACE(ID3DX11EffectType)
{
    STDMETHOD_(BOOL, IsValid)(THIS) PURE;
    STDMETHOD(GetDesc)(THIS_ D3DX11_EFFECT_TYPE_DESC *pDesc) PURE;
    STDMETHOD_(ID3DX11EffectType *, GetMemberTypeByIndex)(THIS_ UINT Index) PURE;
    STDMETHOD_(ID3DX11EffectType *, GetMemberTypeByName)(THIS_ LPCSTR Name) PURE;
    STDMETHOD_(ID3DX11EffectType *, GetMemberTypeBySemantic)(THIS_ LPCSTR Semantic) PURE;
    STDMETHOD_(LPCSTR, GetMemberName)(THIS_ UINT Index) PURE;
    STDMETHOD_(LPCSTR, GetMemberSemantic)(THIS_ UINT Index) PURE;
};

//////////////////////////////////////////////////////////////////////////////
// ID3DX11EffectVariable //////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

//----------------------------------------------------------------------------
// D3DX11_EFFECT_VARIABLE_DESC:
//
// Retrieved by ID3DX11EffectVariable::GetDesc()
//----------------------------------------------------------------------------

typedef struct _D3DX11_EFFECT_VARIABLE_DESC
{
    LPCSTR  Name;                   // Name of this variable, annotation,
    // or structure member
    LPCSTR  Semantic;               // Semantic string of this variable
    // or structure member (NULL for
    // annotations or if not present)

    UINT    Flags;                  // D3DX11_EFFECT_VARIABLE_* flags
    UINT    Annotations;            // Number of annotations on this variable
    // (always 0 for annotations)

    UINT    BufferOffset;           // Offset into containing cbuffer or tbuffer
    // (always 0 for annotations or variables
    // not in constant buffers)

    UINT    ExplicitBindPoint;      // Used if the variable has been explicitly bound
    // using the register keyword. Check Flags for
    // D3DX11_EFFECT_VARIABLE_EXPLICIT_BIND_POINT;
} D3DX11_EFFECT_VARIABLE_DESC;

typedef interface ID3DX11EffectVariable ID3DX11EffectVariable;
typedef interface ID3DX11EffectVariable *LPD3D11EFFECTVARIABLE;

// {036A777D-B56E-4B25-B313-CC3DDAB71873}
DEFINE_GUID(IID_ID3DX11EffectVariable,
            0x036a777d, 0xb56e, 0x4b25, 0xb3, 0x13, 0xcc, 0x3d, 0xda, 0xb7, 0x18, 0x73);

#undef INTERFACE
#define INTERFACE ID3DX11EffectVariable

// Forward defines
typedef interface ID3DX11EffectScalarVariable ID3DX11EffectScalarVariable;
typedef interface ID3DX11EffectVectorVariable ID3DX11EffectVectorVariable;
typedef interface ID3DX11EffectMatrixVariable ID3DX11EffectMatrixVariable;
typedef interface ID3DX11EffectStringVariable ID3DX11EffectStringVariable;
typedef interface ID3DX11EffectClassInstanceVariable ID3DX11EffectClassInstanceVariable;
typedef interface ID3DX11EffectInterfaceVariable ID3DX11EffectInterfaceVariable;
typedef interface ID3DX11EffectShaderResourceVariable ID3DX11EffectShaderResourceVariable;
typedef interface ID3DX11EffectUnorderedAccessViewVariable ID3DX11EffectUnorderedAccessViewVariable;
typedef interface ID3DX11EffectRenderTargetViewVariable ID3DX11EffectRenderTargetViewVariable;
typedef interface ID3DX11EffectDepthStencilViewVariable ID3DX11EffectDepthStencilViewVariable;
typedef interface ID3DX11EffectConstantBuffer ID3DX11EffectConstantBuffer;
typedef interface ID3DX11EffectShaderVariable ID3DX11EffectShaderVariable;
typedef interface ID3DX11EffectBlendVariable ID3DX11EffectBlendVariable;
typedef interface ID3DX11EffectDepthStencilVariable ID3DX11EffectDepthStencilVariable;
typedef interface ID3DX11EffectRasterizerVariable ID3DX11EffectRasterizerVariable;
typedef interface ID3DX11EffectSamplerVariable ID3DX11EffectSamplerVariable;

DECLARE_INTERFACE(ID3DX11EffectVariable)
{
    STDMETHOD_(BOOL, IsValid)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectType *, GetType)(THIS) PURE;
    STDMETHOD(GetDesc)(THIS_ D3DX11_EFFECT_VARIABLE_DESC *pDesc) PURE;

    STDMETHOD_(ID3DX11EffectVariable *, GetAnnotationByIndex)(THIS_ UINT Index) PURE;
    STDMETHOD_(ID3DX11EffectVariable *, GetAnnotationByName)(THIS_ LPCSTR Name) PURE;

    STDMETHOD_(ID3DX11EffectVariable *, GetMemberByIndex)(THIS_ UINT Index) PURE;
    STDMETHOD_(ID3DX11EffectVariable *, GetMemberByName)(THIS_ LPCSTR Name) PURE;
    STDMETHOD_(ID3DX11EffectVariable *, GetMemberBySemantic)(THIS_ LPCSTR Semantic) PURE;

    STDMETHOD_(ID3DX11EffectVariable *, GetElement)(THIS_ UINT Index) PURE;

    STDMETHOD_(ID3DX11EffectConstantBuffer *, GetParentConstantBuffer)(THIS) PURE;

    STDMETHOD_(ID3DX11EffectScalarVariable *, AsScalar)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectVectorVariable *, AsVector)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectMatrixVariable *, AsMatrix)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectStringVariable *, AsString)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectClassInstanceVariable *, AsClassInstance)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectInterfaceVariable *, AsInterface)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectShaderResourceVariable *, AsShaderResource)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectUnorderedAccessViewVariable *, AsUnorderedAccessView)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectRenderTargetViewVariable *, AsRenderTargetView)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectDepthStencilViewVariable *, AsDepthStencilView)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectConstantBuffer *, AsConstantBuffer)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectShaderVariable *, AsShader)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectBlendVariable *, AsBlend)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectDepthStencilVariable *, AsDepthStencil)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectRasterizerVariable *, AsRasterizer)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectSamplerVariable *, AsSampler)(THIS) PURE;

    STDMETHOD(SetRawValue)(THIS_ void *pData, UINT Offset, UINT Count) PURE;
    STDMETHOD(GetRawValue)(THIS_ void *pData, UINT Offset, UINT Count) PURE;
};

//////////////////////////////////////////////////////////////////////////////
// ID3DX11EffectScalarVariable ////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

typedef interface ID3DX11EffectScalarVariable ID3DX11EffectScalarVariable;
typedef interface ID3DX11EffectScalarVariable *LPD3D11EFFECTSCALARVARIABLE;

// {921EF2E5-A65D-4E92-9FC6-4E9CC09A4ADE}
DEFINE_GUID(IID_ID3DX11EffectScalarVariable,
            0x921ef2e5, 0xa65d, 0x4e92, 0x9f, 0xc6, 0x4e, 0x9c, 0xc0, 0x9a, 0x4a, 0xde);

#undef INTERFACE
#define INTERFACE ID3DX11EffectScalarVariable

DECLARE_INTERFACE_(ID3DX11EffectScalarVariable, ID3DX11EffectVariable)
{
    STDMETHOD_(BOOL, IsValid)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectType *, GetType)(THIS) PURE;
    STDMETHOD(GetDesc)(THIS_ D3DX11_EFFECT_VARIABLE_DESC *pDesc) PURE;

    STDMETHOD_(ID3DX11EffectVariable *, GetAnnotationByIndex)(THIS_ UINT Index) PURE;
    STDMETHOD_(ID3DX11EffectVariable *, GetAnnotationByName)(THIS_ LPCSTR Name) PURE;

    STDMETHOD_(ID3DX11EffectVariable *, GetMemberByIndex)(THIS_ UINT Index) PURE;
    STDMETHOD_(ID3DX11EffectVariable *, GetMemberByName)(THIS_ LPCSTR Name) PURE;
    STDMETHOD_(ID3DX11EffectVariable *, GetMemberBySemantic)(THIS_ LPCSTR Semantic) PURE;

    STDMETHOD_(ID3DX11EffectVariable *, GetElement)(THIS_ UINT Index) PURE;

    STDMETHOD_(ID3DX11EffectConstantBuffer *, GetParentConstantBuffer)(THIS) PURE;

    STDMETHOD_(ID3DX11EffectScalarVariable *, AsScalar)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectVectorVariable *, AsVector)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectMatrixVariable *, AsMatrix)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectStringVariable *, AsString)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectClassInstanceVariable *, AsClassInstance)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectInterfaceVariable *, AsInterface)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectShaderResourceVariable *, AsShaderResource)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectUnorderedAccessViewVariable *, AsUnorderedAccessView)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectRenderTargetViewVariable *, AsRenderTargetView)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectDepthStencilViewVariable *, AsDepthStencilView)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectConstantBuffer *, AsConstantBuffer)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectShaderVariable *, AsShader)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectBlendVariable *, AsBlend)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectDepthStencilVariable *, AsDepthStencil)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectRasterizerVariable *, AsRasterizer)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectSamplerVariable *, AsSampler)(THIS) PURE;

    STDMETHOD(SetRawValue)(THIS_ void *pData, UINT ByteOffset, UINT ByteCount) PURE;
    STDMETHOD(GetRawValue)(THIS_ void *pData, UINT ByteOffset, UINT ByteCount) PURE;

    STDMETHOD(SetFloat)(THIS_ float Value) PURE;
    STDMETHOD(GetFloat)(THIS_ float *pValue) PURE;

    STDMETHOD(SetFloatArray)(THIS_ float *pData, UINT Offset, UINT Count) PURE;
    STDMETHOD(GetFloatArray)(THIS_ float *pData, UINT Offset, UINT Count) PURE;

    STDMETHOD(SetInt)(THIS_ int Value) PURE;
    STDMETHOD(GetInt)(THIS_ int *pValue) PURE;

    STDMETHOD(SetIntArray)(THIS_ int *pData, UINT Offset, UINT Count) PURE;
    STDMETHOD(GetIntArray)(THIS_ int *pData, UINT Offset, UINT Count) PURE;

    STDMETHOD(SetBool)(THIS_ BOOL Value) PURE;
    STDMETHOD(GetBool)(THIS_ BOOL *pValue) PURE;

    STDMETHOD(SetBoolArray)(THIS_ BOOL *pData, UINT Offset, UINT Count) PURE;
    STDMETHOD(GetBoolArray)(THIS_ BOOL *pData, UINT Offset, UINT Count) PURE;
};

//////////////////////////////////////////////////////////////////////////////
// ID3DX11EffectVectorVariable ////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

typedef interface ID3DX11EffectVectorVariable ID3DX11EffectVectorVariable;
typedef interface ID3DX11EffectVectorVariable *LPD3D11EFFECTVECTORVARIABLE;

// {5E785D4A-D87B-48D8-B6E6-0F8CA7E7467A}
DEFINE_GUID(IID_ID3DX11EffectVectorVariable,
            0x5e785d4a, 0xd87b, 0x48d8, 0xb6, 0xe6, 0x0f, 0x8c, 0xa7, 0xe7, 0x46, 0x7a);

#undef INTERFACE
#define INTERFACE ID3DX11EffectVectorVariable

DECLARE_INTERFACE_(ID3DX11EffectVectorVariable, ID3DX11EffectVariable)
{
    STDMETHOD_(BOOL, IsValid)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectType *, GetType)(THIS) PURE;
    STDMETHOD(GetDesc)(THIS_ D3DX11_EFFECT_VARIABLE_DESC *pDesc) PURE;

    STDMETHOD_(ID3DX11EffectVariable *, GetAnnotationByIndex)(THIS_ UINT Index) PURE;
    STDMETHOD_(ID3DX11EffectVariable *, GetAnnotationByName)(THIS_ LPCSTR Name) PURE;

    STDMETHOD_(ID3DX11EffectVariable *, GetMemberByIndex)(THIS_ UINT Index) PURE;
    STDMETHOD_(ID3DX11EffectVariable *, GetMemberByName)(THIS_ LPCSTR Name) PURE;
    STDMETHOD_(ID3DX11EffectVariable *, GetMemberBySemantic)(THIS_ LPCSTR Semantic) PURE;

    STDMETHOD_(ID3DX11EffectVariable *, GetElement)(THIS_ UINT Index) PURE;

    STDMETHOD_(ID3DX11EffectConstantBuffer *, GetParentConstantBuffer)(THIS) PURE;

    STDMETHOD_(ID3DX11EffectScalarVariable *, AsScalar)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectVectorVariable *, AsVector)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectMatrixVariable *, AsMatrix)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectStringVariable *, AsString)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectClassInstanceVariable *, AsClassInstance)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectInterfaceVariable *, AsInterface)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectShaderResourceVariable *, AsShaderResource)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectUnorderedAccessViewVariable *, AsUnorderedAccessView)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectRenderTargetViewVariable *, AsRenderTargetView)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectDepthStencilViewVariable *, AsDepthStencilView)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectConstantBuffer *, AsConstantBuffer)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectShaderVariable *, AsShader)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectBlendVariable *, AsBlend)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectDepthStencilVariable *, AsDepthStencil)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectRasterizerVariable *, AsRasterizer)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectSamplerVariable *, AsSampler)(THIS) PURE;

    STDMETHOD(SetRawValue)(THIS_ void *pData, UINT ByteOffset, UINT ByteCount) PURE;
    STDMETHOD(GetRawValue)(THIS_ void *pData, UINT ByteOffset, UINT ByteCount) PURE;

    STDMETHOD(SetBoolVector)(THIS_ BOOL *pData) PURE;
    STDMETHOD(SetIntVector)(THIS_ int *pData) PURE;
    STDMETHOD(SetFloatVector)(THIS_ float *pData) PURE;

    STDMETHOD(GetBoolVector)(THIS_ BOOL *pData) PURE;
    STDMETHOD(GetIntVector)(THIS_ int *pData) PURE;
    STDMETHOD(GetFloatVector)(THIS_ float *pData) PURE;

    STDMETHOD(SetBoolVectorArray)(THIS_ BOOL *pData, UINT Offset, UINT Count) PURE;
    STDMETHOD(SetIntVectorArray)(THIS_ int *pData, UINT Offset, UINT Count) PURE;
    STDMETHOD(SetFloatVectorArray)(THIS_ float *pData, UINT Offset, UINT Count) PURE;

    STDMETHOD(GetBoolVectorArray)(THIS_ BOOL *pData, UINT Offset, UINT Count) PURE;
    STDMETHOD(GetIntVectorArray)(THIS_ int *pData, UINT Offset, UINT Count) PURE;
    STDMETHOD(GetFloatVectorArray)(THIS_ float *pData, UINT Offset, UINT Count) PURE;
};

//////////////////////////////////////////////////////////////////////////////
// ID3DX11EffectMatrixVariable ////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

typedef interface ID3DX11EffectMatrixVariable ID3DX11EffectMatrixVariable;
typedef interface ID3DX11EffectMatrixVariable *LPD3D11EFFECTMATRIXVARIABLE;

// {E1096CF4-C027-419A-8D86-D29173DC803E}
DEFINE_GUID(IID_ID3DX11EffectMatrixVariable,
            0xe1096cf4, 0xc027, 0x419a, 0x8d, 0x86, 0xd2, 0x91, 0x73, 0xdc, 0x80, 0x3e);

#undef INTERFACE
#define INTERFACE ID3DX11EffectMatrixVariable

DECLARE_INTERFACE_(ID3DX11EffectMatrixVariable, ID3DX11EffectVariable)
{
    STDMETHOD_(BOOL, IsValid)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectType *, GetType)(THIS) PURE;
    STDMETHOD(GetDesc)(THIS_ D3DX11_EFFECT_VARIABLE_DESC *pDesc) PURE;

    STDMETHOD_(ID3DX11EffectVariable *, GetAnnotationByIndex)(THIS_ UINT Index) PURE;
    STDMETHOD_(ID3DX11EffectVariable *, GetAnnotationByName)(THIS_ LPCSTR Name) PURE;

    STDMETHOD_(ID3DX11EffectVariable *, GetMemberByIndex)(THIS_ UINT Index) PURE;
    STDMETHOD_(ID3DX11EffectVariable *, GetMemberByName)(THIS_ LPCSTR Name) PURE;
    STDMETHOD_(ID3DX11EffectVariable *, GetMemberBySemantic)(THIS_ LPCSTR Semantic) PURE;

    STDMETHOD_(ID3DX11EffectVariable *, GetElement)(THIS_ UINT Index) PURE;

    STDMETHOD_(ID3DX11EffectConstantBuffer *, GetParentConstantBuffer)(THIS) PURE;

    STDMETHOD_(ID3DX11EffectScalarVariable *, AsScalar)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectVectorVariable *, AsVector)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectMatrixVariable *, AsMatrix)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectStringVariable *, AsString)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectClassInstanceVariable *, AsClassInstance)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectInterfaceVariable *, AsInterface)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectShaderResourceVariable *, AsShaderResource)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectUnorderedAccessViewVariable *, AsUnorderedAccessView)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectRenderTargetViewVariable *, AsRenderTargetView)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectDepthStencilViewVariable *, AsDepthStencilView)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectConstantBuffer *, AsConstantBuffer)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectShaderVariable *, AsShader)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectBlendVariable *, AsBlend)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectDepthStencilVariable *, AsDepthStencil)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectRasterizerVariable *, AsRasterizer)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectSamplerVariable *, AsSampler)(THIS) PURE;

    STDMETHOD(SetRawValue)(THIS_ void *pData, UINT ByteOffset, UINT ByteCount) PURE;
    STDMETHOD(GetRawValue)(THIS_ void *pData, UINT ByteOffset, UINT ByteCount) PURE;

    STDMETHOD(SetMatrix)(THIS_ float *pData) PURE;
    STDMETHOD(GetMatrix)(THIS_ float *pData) PURE;

    STDMETHOD(SetMatrixArray)(THIS_ float *pData, UINT Offset, UINT Count) PURE;
    STDMETHOD(GetMatrixArray)(THIS_ float *pData, UINT Offset, UINT Count) PURE;

    STDMETHOD(SetMatrixTranspose)(THIS_ float *pData) PURE;
    STDMETHOD(GetMatrixTranspose)(THIS_ float *pData) PURE;

    STDMETHOD(SetMatrixTransposeArray)(THIS_ float *pData, UINT Offset, UINT Count) PURE;
    STDMETHOD(GetMatrixTransposeArray)(THIS_ float *pData, UINT Offset, UINT Count) PURE;
};

//////////////////////////////////////////////////////////////////////////////
// ID3DX11EffectStringVariable ////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

typedef interface ID3DX11EffectStringVariable ID3DX11EffectStringVariable;
typedef interface ID3DX11EffectStringVariable *LPD3D11EFFECTSTRINGVARIABLE;

// {F355C818-01BE-4653-A7CC-60FFFEDDC76D}
DEFINE_GUID(IID_ID3DX11EffectStringVariable,
            0xf355c818, 0x01be, 0x4653, 0xa7, 0xcc, 0x60, 0xff, 0xfe, 0xdd, 0xc7, 0x6d);

#undef INTERFACE
#define INTERFACE ID3DX11EffectStringVariable

DECLARE_INTERFACE_(ID3DX11EffectStringVariable, ID3DX11EffectVariable)
{
    STDMETHOD_(BOOL, IsValid)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectType *, GetType)(THIS) PURE;
    STDMETHOD(GetDesc)(THIS_ D3DX11_EFFECT_VARIABLE_DESC *pDesc) PURE;

    STDMETHOD_(ID3DX11EffectVariable *, GetAnnotationByIndex)(THIS_ UINT Index) PURE;
    STDMETHOD_(ID3DX11EffectVariable *, GetAnnotationByName)(THIS_ LPCSTR Name) PURE;

    STDMETHOD_(ID3DX11EffectVariable *, GetMemberByIndex)(THIS_ UINT Index) PURE;
    STDMETHOD_(ID3DX11EffectVariable *, GetMemberByName)(THIS_ LPCSTR Name) PURE;
    STDMETHOD_(ID3DX11EffectVariable *, GetMemberBySemantic)(THIS_ LPCSTR Semantic) PURE;

    STDMETHOD_(ID3DX11EffectVariable *, GetElement)(THIS_ UINT Index) PURE;

    STDMETHOD_(ID3DX11EffectConstantBuffer *, GetParentConstantBuffer)(THIS) PURE;

    STDMETHOD_(ID3DX11EffectScalarVariable *, AsScalar)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectVectorVariable *, AsVector)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectMatrixVariable *, AsMatrix)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectStringVariable *, AsString)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectClassInstanceVariable *, AsClassInstance)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectInterfaceVariable *, AsInterface)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectShaderResourceVariable *, AsShaderResource)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectUnorderedAccessViewVariable *, AsUnorderedAccessView)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectRenderTargetViewVariable *, AsRenderTargetView)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectDepthStencilViewVariable *, AsDepthStencilView)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectConstantBuffer *, AsConstantBuffer)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectShaderVariable *, AsShader)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectBlendVariable *, AsBlend)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectDepthStencilVariable *, AsDepthStencil)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectRasterizerVariable *, AsRasterizer)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectSamplerVariable *, AsSampler)(THIS) PURE;

    STDMETHOD(SetRawValue)(THIS_ void *pData, UINT Offset, UINT Count) PURE;
    STDMETHOD(GetRawValue)(THIS_ void *pData, UINT Offset, UINT Count) PURE;

    STDMETHOD(GetString)(THIS_ LPCSTR *ppString) PURE;
    STDMETHOD(GetStringArray)(THIS_ LPCSTR *ppStrings, UINT Offset, UINT Count) PURE;
};

//////////////////////////////////////////////////////////////////////////////
// ID3DX11EffectClassInstanceVariable ////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

typedef interface ID3DX11EffectClassInstanceVariable ID3DX11EffectClassInstanceVariable;
typedef interface ID3DX11EffectClassInstanceVariable *LPD3D11EFFECTCLASSINSTANCEVARIABLE;

// {926A8053-2A39-4DB4-9BDE-CF649ADEBDC1}
DEFINE_GUID(IID_ID3DX11EffectClassInstanceVariable,
            0x926a8053, 0x2a39, 0x4db4, 0x9b, 0xde, 0xcf, 0x64, 0x9a, 0xde, 0xbd, 0xc1);

#undef INTERFACE
#define INTERFACE ID3DX11EffectClassInstanceVariable

DECLARE_INTERFACE_(ID3DX11EffectClassInstanceVariable, ID3DX11EffectVariable)
{
    STDMETHOD_(BOOL, IsValid)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectType *, GetType)(THIS) PURE;
    STDMETHOD(GetDesc)(THIS_ D3DX11_EFFECT_VARIABLE_DESC *pDesc) PURE;

    STDMETHOD_(ID3DX11EffectVariable *, GetAnnotationByIndex)(THIS_ UINT Index) PURE;
    STDMETHOD_(ID3DX11EffectVariable *, GetAnnotationByName)(THIS_ LPCSTR Name) PURE;

    STDMETHOD_(ID3DX11EffectVariable *, GetMemberByIndex)(THIS_ UINT Index) PURE;
    STDMETHOD_(ID3DX11EffectVariable *, GetMemberByName)(THIS_ LPCSTR Name) PURE;
    STDMETHOD_(ID3DX11EffectVariable *, GetMemberBySemantic)(THIS_ LPCSTR Semantic) PURE;

    STDMETHOD_(ID3DX11EffectVariable *, GetElement)(THIS_ UINT Index) PURE;

    STDMETHOD_(ID3DX11EffectConstantBuffer *, GetParentConstantBuffer)(THIS) PURE;

    STDMETHOD_(ID3DX11EffectScalarVariable *, AsScalar)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectVectorVariable *, AsVector)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectMatrixVariable *, AsMatrix)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectStringVariable *, AsString)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectClassInstanceVariable *, AsClassInstance)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectInterfaceVariable *, AsInterface)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectUnorderedAccessViewVariable *, AsUnorderedAccessView)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectRenderTargetViewVariable *, AsRenderTargetView)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectDepthStencilViewVariable *, AsDepthStencilView)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectConstantBuffer *, AsConstantBuffer)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectShaderVariable *, AsShader)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectBlendVariable *, AsBlend)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectDepthStencilVariable *, AsDepthStencil)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectRasterizerVariable *, AsRasterizer)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectSamplerVariable *, AsSampler)(THIS) PURE;

    STDMETHOD(SetRawValue)(THIS_ void *pData, UINT Offset, UINT Count) PURE;
    STDMETHOD(GetRawValue)(THIS_ void *pData, UINT Offset, UINT Count) PURE;

    STDMETHOD(GetClassInstance)(ID3D11ClassInstance **ppClassInstance) PURE;
};

//////////////////////////////////////////////////////////////////////////////
// ID3DX11EffectInterfaceVariable ////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

typedef interface ID3DX11EffectInterfaceVariable ID3DX11EffectInterfaceVariable;
typedef interface ID3DX11EffectInterfaceVariable *LPD3D11EFFECTINTERFACEVARIABLE;

// {516C8CD8-1C80-40A4-B19B-0688792F11A5}
DEFINE_GUID(IID_ID3DX11EffectInterfaceVariable,
            0x516c8cd8, 0x1c80, 0x40a4, 0xb1, 0x9b, 0x06, 0x88, 0x79, 0x2f, 0x11, 0xa5);

#undef INTERFACE
#define INTERFACE ID3DX11EffectInterfaceVariable

DECLARE_INTERFACE_(ID3DX11EffectInterfaceVariable, ID3DX11EffectVariable)
{
    STDMETHOD_(BOOL, IsValid)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectType *, GetType)(THIS) PURE;
    STDMETHOD(GetDesc)(THIS_ D3DX11_EFFECT_VARIABLE_DESC *pDesc) PURE;

    STDMETHOD_(ID3DX11EffectVariable *, GetAnnotationByIndex)(THIS_ UINT Index) PURE;
    STDMETHOD_(ID3DX11EffectVariable *, GetAnnotationByName)(THIS_ LPCSTR Name) PURE;

    STDMETHOD_(ID3DX11EffectVariable *, GetMemberByIndex)(THIS_ UINT Index) PURE;
    STDMETHOD_(ID3DX11EffectVariable *, GetMemberByName)(THIS_ LPCSTR Name) PURE;
    STDMETHOD_(ID3DX11EffectVariable *, GetMemberBySemantic)(THIS_ LPCSTR Semantic) PURE;

    STDMETHOD_(ID3DX11EffectVariable *, GetElement)(THIS_ UINT Index) PURE;

    STDMETHOD_(ID3DX11EffectConstantBuffer *, GetParentConstantBuffer)(THIS) PURE;

    STDMETHOD_(ID3DX11EffectScalarVariable *, AsScalar)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectVectorVariable *, AsVector)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectMatrixVariable *, AsMatrix)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectStringVariable *, AsString)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectClassInstanceVariable *, AsClassInstance)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectInterfaceVariable *, AsInterface)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectUnorderedAccessViewVariable *, AsUnorderedAccessView)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectRenderTargetViewVariable *, AsRenderTargetView)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectDepthStencilViewVariable *, AsDepthStencilView)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectConstantBuffer *, AsConstantBuffer)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectShaderVariable *, AsShader)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectBlendVariable *, AsBlend)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectDepthStencilVariable *, AsDepthStencil)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectRasterizerVariable *, AsRasterizer)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectSamplerVariable *, AsSampler)(THIS) PURE;

    STDMETHOD(SetRawValue)(THIS_ void *pData, UINT Offset, UINT Count) PURE;
    STDMETHOD(GetRawValue)(THIS_ void *pData, UINT Offset, UINT Count) PURE;

    STDMETHOD(SetClassInstance)(ID3DX11EffectClassInstanceVariable *pEffectClassInstance) PURE;
    STDMETHOD(GetClassInstance)(ID3DX11EffectClassInstanceVariable **ppEffectClassInstance) PURE;
};

//////////////////////////////////////////////////////////////////////////////
// ID3DX11EffectShaderResourceVariable ////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

typedef interface ID3DX11EffectShaderResourceVariable ID3DX11EffectShaderResourceVariable;
typedef interface ID3DX11EffectShaderResourceVariable *LPD3D11EFFECTSHADERRESOURCEVARIABLE;

// {350DB233-BBE0-485C-9BFE-C0026B844F89}
DEFINE_GUID(IID_ID3DX11EffectShaderResourceVariable,
            0x350db233, 0xbbe0, 0x485c, 0x9b, 0xfe, 0xc0, 0x02, 0x6b, 0x84, 0x4f, 0x89);

#undef INTERFACE
#define INTERFACE ID3DX11EffectShaderResourceVariable

DECLARE_INTERFACE_(ID3DX11EffectShaderResourceVariable, ID3DX11EffectVariable)
{
    STDMETHOD_(BOOL, IsValid)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectType *, GetType)(THIS) PURE;
    STDMETHOD(GetDesc)(THIS_ D3DX11_EFFECT_VARIABLE_DESC *pDesc) PURE;

    STDMETHOD_(ID3DX11EffectVariable *, GetAnnotationByIndex)(THIS_ UINT Index) PURE;
    STDMETHOD_(ID3DX11EffectVariable *, GetAnnotationByName)(THIS_ LPCSTR Name) PURE;

    STDMETHOD_(ID3DX11EffectVariable *, GetMemberByIndex)(THIS_ UINT Index) PURE;
    STDMETHOD_(ID3DX11EffectVariable *, GetMemberByName)(THIS_ LPCSTR Name) PURE;
    STDMETHOD_(ID3DX11EffectVariable *, GetMemberBySemantic)(THIS_ LPCSTR Semantic) PURE;

    STDMETHOD_(ID3DX11EffectVariable *, GetElement)(THIS_ UINT Index) PURE;

    STDMETHOD_(ID3DX11EffectConstantBuffer *, GetParentConstantBuffer)(THIS) PURE;

    STDMETHOD_(ID3DX11EffectScalarVariable *, AsScalar)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectVectorVariable *, AsVector)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectMatrixVariable *, AsMatrix)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectStringVariable *, AsString)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectClassInstanceVariable *, AsClassInstance)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectInterfaceVariable *, AsInterface)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectShaderResourceVariable *, AsShaderResource)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectUnorderedAccessViewVariable *, AsUnorderedAccessView)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectRenderTargetViewVariable *, AsRenderTargetView)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectDepthStencilViewVariable *, AsDepthStencilView)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectConstantBuffer *, AsConstantBuffer)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectShaderVariable *, AsShader)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectBlendVariable *, AsBlend)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectDepthStencilVariable *, AsDepthStencil)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectRasterizerVariable *, AsRasterizer)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectSamplerVariable *, AsSampler)(THIS) PURE;

    STDMETHOD(SetRawValue)(THIS_ void *pData, UINT Offset, UINT Count) PURE;
    STDMETHOD(GetRawValue)(THIS_ void *pData, UINT Offset, UINT Count) PURE;

    STDMETHOD(SetResource)(THIS_ ID3D11ShaderResourceView *pResource) PURE;
    STDMETHOD(GetResource)(THIS_ ID3D11ShaderResourceView **ppResource) PURE;

    STDMETHOD(SetResourceArray)(THIS_ ID3D11ShaderResourceView **ppResources, UINT Offset, UINT Count) PURE;
    STDMETHOD(GetResourceArray)(THIS_ ID3D11ShaderResourceView **ppResources, UINT Offset, UINT Count) PURE;
};

//////////////////////////////////////////////////////////////////////////////
// ID3DX11EffectUnorderedAccessViewVariable ////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

typedef interface ID3DX11EffectUnorderedAccessViewVariable ID3DX11EffectUnorderedAccessViewVariable;
typedef interface ID3DX11EffectUnorderedAccessViewVariable *LPD3D11EFFECTUNORDEREDACCESSVIEWVARIABLE;

// {79B4AC8C-870A-47D2-B05A-8BD5CC3EE6C9}
DEFINE_GUID(IID_ID3DX11EffectUnorderedAccessViewVariable,
            0x79b4ac8c, 0x870a, 0x47d2, 0xb0, 0x5a, 0x8b, 0xd5, 0xcc, 0x3e, 0xe6, 0xc9);

#undef INTERFACE
#define INTERFACE ID3DX11EffectUnorderedAccessViewVariable

DECLARE_INTERFACE_(ID3DX11EffectUnorderedAccessViewVariable, ID3DX11EffectVariable)
{
    STDMETHOD_(BOOL, IsValid)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectType *, GetType)(THIS) PURE;
    STDMETHOD(GetDesc)(THIS_ D3DX11_EFFECT_VARIABLE_DESC *pDesc) PURE;

    STDMETHOD_(ID3DX11EffectVariable *, GetAnnotationByIndex)(THIS_ UINT Index) PURE;
    STDMETHOD_(ID3DX11EffectVariable *, GetAnnotationByName)(THIS_ LPCSTR Name) PURE;

    STDMETHOD_(ID3DX11EffectVariable *, GetMemberByIndex)(THIS_ UINT Index) PURE;
    STDMETHOD_(ID3DX11EffectVariable *, GetMemberByName)(THIS_ LPCSTR Name) PURE;
    STDMETHOD_(ID3DX11EffectVariable *, GetMemberBySemantic)(THIS_ LPCSTR Semantic) PURE;

    STDMETHOD_(ID3DX11EffectVariable *, GetElement)(THIS_ UINT Index) PURE;

    STDMETHOD_(ID3DX11EffectConstantBuffer *, GetParentConstantBuffer)(THIS) PURE;

    STDMETHOD_(ID3DX11EffectScalarVariable *, AsScalar)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectVectorVariable *, AsVector)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectMatrixVariable *, AsMatrix)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectStringVariable *, AsString)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectClassInstanceVariable *, AsClassInstance)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectInterfaceVariable *, AsInterface)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectShaderResourceVariable *, AsShaderResource)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectUnorderedAccessViewVariable *, AsUnorderedAccessView)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectRenderTargetViewVariable *, AsRenderTargetView)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectDepthStencilViewVariable *, AsDepthStencilView)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectConstantBuffer *, AsConstantBuffer)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectShaderVariable *, AsShader)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectBlendVariable *, AsBlend)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectDepthStencilVariable *, AsDepthStencil)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectRasterizerVariable *, AsRasterizer)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectSamplerVariable *, AsSampler)(THIS) PURE;

    STDMETHOD(SetRawValue)(THIS_ void *pData, UINT Offset, UINT Count) PURE;
    STDMETHOD(GetRawValue)(THIS_ void *pData, UINT Offset, UINT Count) PURE;

    STDMETHOD(SetUnorderedAccessView)(THIS_ ID3D11UnorderedAccessView *pResource) PURE;
    STDMETHOD(GetUnorderedAccessView)(THIS_ ID3D11UnorderedAccessView **ppResource) PURE;

    STDMETHOD(SetUnorderedAccessViewArray)(THIS_ ID3D11UnorderedAccessView **ppResources, UINT Offset, UINT Count) PURE;
    STDMETHOD(GetUnorderedAccessViewArray)(THIS_ ID3D11UnorderedAccessView **ppResources, UINT Offset, UINT Count) PURE;
};

//////////////////////////////////////////////////////////////////////////////
// ID3DX11EffectRenderTargetViewVariable //////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

typedef interface ID3DX11EffectRenderTargetViewVariable ID3DX11EffectRenderTargetViewVariable;
typedef interface ID3DX11EffectRenderTargetViewVariable *LPD3D11EFFECTRENDERTARGETVIEWVARIABLE;

// {D5066909-F40C-43F8-9DB5-057C2A208552}
DEFINE_GUID(IID_ID3DX11EffectRenderTargetViewVariable,
            0xd5066909, 0xf40c, 0x43f8, 0x9d, 0xb5, 0x05, 0x7c, 0x2a, 0x20, 0x85, 0x52);

#undef INTERFACE
#define INTERFACE ID3DX11EffectRenderTargetViewVariable

DECLARE_INTERFACE_(ID3DX11EffectRenderTargetViewVariable, ID3DX11EffectVariable)
{
    STDMETHOD_(BOOL, IsValid)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectType *, GetType)(THIS) PURE;
    STDMETHOD(GetDesc)(THIS_ D3DX11_EFFECT_VARIABLE_DESC *pDesc) PURE;

    STDMETHOD_(ID3DX11EffectVariable *, GetAnnotationByIndex)(THIS_ UINT Index) PURE;
    STDMETHOD_(ID3DX11EffectVariable *, GetAnnotationByName)(THIS_ LPCSTR Name) PURE;

    STDMETHOD_(ID3DX11EffectVariable *, GetMemberByIndex)(THIS_ UINT Index) PURE;
    STDMETHOD_(ID3DX11EffectVariable *, GetMemberByName)(THIS_ LPCSTR Name) PURE;
    STDMETHOD_(ID3DX11EffectVariable *, GetMemberBySemantic)(THIS_ LPCSTR Semantic) PURE;

    STDMETHOD_(ID3DX11EffectVariable *, GetElement)(THIS_ UINT Index) PURE;

    STDMETHOD_(ID3DX11EffectConstantBuffer *, GetParentConstantBuffer)(THIS) PURE;

    STDMETHOD_(ID3DX11EffectScalarVariable *, AsScalar)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectVectorVariable *, AsVector)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectMatrixVariable *, AsMatrix)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectStringVariable *, AsString)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectClassInstanceVariable *, AsClassInstance)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectInterfaceVariable *, AsInterface)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectShaderResourceVariable *, AsShaderResource)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectUnorderedAccessViewVariable *, AsUnorderedAccessView)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectRenderTargetViewVariable *, AsRenderTargetView)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectDepthStencilViewVariable *, AsDepthStencilView)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectConstantBuffer *, AsConstantBuffer)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectShaderVariable *, AsShader)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectBlendVariable *, AsBlend)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectDepthStencilVariable *, AsDepthStencil)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectRasterizerVariable *, AsRasterizer)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectSamplerVariable *, AsSampler)(THIS) PURE;

    STDMETHOD(SetRawValue)(THIS_ void *pData, UINT Offset, UINT Count) PURE;
    STDMETHOD(GetRawValue)(THIS_ void *pData, UINT Offset, UINT Count) PURE;

    STDMETHOD(SetRenderTarget)(THIS_ ID3D11RenderTargetView *pResource) PURE;
    STDMETHOD(GetRenderTarget)(THIS_ ID3D11RenderTargetView **ppResource) PURE;

    STDMETHOD(SetRenderTargetArray)(THIS_ ID3D11RenderTargetView **ppResources, UINT Offset, UINT Count) PURE;
    STDMETHOD(GetRenderTargetArray)(THIS_ ID3D11RenderTargetView **ppResources, UINT Offset, UINT Count) PURE;
};

//////////////////////////////////////////////////////////////////////////////
// ID3DX11EffectDepthStencilViewVariable //////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

typedef interface ID3DX11EffectDepthStencilViewVariable ID3DX11EffectDepthStencilViewVariable;
typedef interface ID3DX11EffectDepthStencilViewVariable *LPD3D11EFFECTDEPTHSTENCILVIEWVARIABLE;

// {33C648AC-2E9E-4A2E-9CD6-DE31ACC5B347}
DEFINE_GUID(IID_ID3DX11EffectDepthStencilViewVariable,
            0x33c648ac, 0x2e9e, 0x4a2e, 0x9c, 0xd6, 0xde, 0x31, 0xac, 0xc5, 0xb3, 0x47);

#undef INTERFACE
#define INTERFACE ID3DX11EffectDepthStencilViewVariable

DECLARE_INTERFACE_(ID3DX11EffectDepthStencilViewVariable, ID3DX11EffectVariable)
{
    STDMETHOD_(BOOL, IsValid)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectType *, GetType)(THIS) PURE;
    STDMETHOD(GetDesc)(THIS_ D3DX11_EFFECT_VARIABLE_DESC *pDesc) PURE;

    STDMETHOD_(ID3DX11EffectVariable *, GetAnnotationByIndex)(THIS_ UINT Index) PURE;
    STDMETHOD_(ID3DX11EffectVariable *, GetAnnotationByName)(THIS_ LPCSTR Name) PURE;

    STDMETHOD_(ID3DX11EffectVariable *, GetMemberByIndex)(THIS_ UINT Index) PURE;
    STDMETHOD_(ID3DX11EffectVariable *, GetMemberByName)(THIS_ LPCSTR Name) PURE;
    STDMETHOD_(ID3DX11EffectVariable *, GetMemberBySemantic)(THIS_ LPCSTR Semantic) PURE;

    STDMETHOD_(ID3DX11EffectVariable *, GetElement)(THIS_ UINT Index) PURE;

    STDMETHOD_(ID3DX11EffectConstantBuffer *, GetParentConstantBuffer)(THIS) PURE;

    STDMETHOD_(ID3DX11EffectScalarVariable *, AsScalar)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectVectorVariable *, AsVector)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectMatrixVariable *, AsMatrix)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectStringVariable *, AsString)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectClassInstanceVariable *, AsClassInstance)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectInterfaceVariable *, AsInterface)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectShaderResourceVariable *, AsShaderResource)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectUnorderedAccessViewVariable *, AsUnorderedAccessView)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectRenderTargetViewVariable *, AsRenderTargetView)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectDepthStencilViewVariable *, AsDepthStencilView)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectConstantBuffer *, AsConstantBuffer)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectShaderVariable *, AsShader)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectBlendVariable *, AsBlend)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectDepthStencilVariable *, AsDepthStencil)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectRasterizerVariable *, AsRasterizer)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectSamplerVariable *, AsSampler)(THIS) PURE;

    STDMETHOD(SetRawValue)(THIS_ void *pData, UINT Offset, UINT Count) PURE;
    STDMETHOD(GetRawValue)(THIS_ void *pData, UINT Offset, UINT Count) PURE;

    STDMETHOD(SetDepthStencil)(THIS_ ID3D11DepthStencilView *pResource) PURE;
    STDMETHOD(GetDepthStencil)(THIS_ ID3D11DepthStencilView **ppResource) PURE;

    STDMETHOD(SetDepthStencilArray)(THIS_ ID3D11DepthStencilView **ppResources, UINT Offset, UINT Count) PURE;
    STDMETHOD(GetDepthStencilArray)(THIS_ ID3D11DepthStencilView **ppResources, UINT Offset, UINT Count) PURE;
};

//////////////////////////////////////////////////////////////////////////////
// ID3DX11EffectConstantBuffer ////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

typedef interface ID3DX11EffectConstantBuffer ID3DX11EffectConstantBuffer;
typedef interface ID3DX11EffectConstantBuffer *LPD3D11EFFECTCONSTANTBUFFER;

// {2CB6C733-82D2-4000-B3DA-6219D9A99BF2}
DEFINE_GUID(IID_ID3DX11EffectConstantBuffer,
            0x2cb6c733, 0x82d2, 0x4000, 0xb3, 0xda, 0x62, 0x19, 0xd9, 0xa9, 0x9b, 0xf2);

#undef INTERFACE
#define INTERFACE ID3DX11EffectConstantBuffer

DECLARE_INTERFACE_(ID3DX11EffectConstantBuffer, ID3DX11EffectVariable)
{
    STDMETHOD_(ID3DX11EffectType *, GetType)(THIS) PURE;
    STDMETHOD(GetDesc)(THIS_ D3DX11_EFFECT_VARIABLE_DESC *pDesc) PURE;

    STDMETHOD_(ID3DX11EffectVariable *, GetAnnotationByIndex)(THIS_ UINT Index) PURE;
    STDMETHOD_(ID3DX11EffectVariable *, GetAnnotationByName)(THIS_ LPCSTR Name) PURE;

    STDMETHOD_(ID3DX11EffectVariable *, GetMemberByIndex)(THIS_ UINT Index) PURE;
    STDMETHOD_(ID3DX11EffectVariable *, GetMemberByName)(THIS_ LPCSTR Name) PURE;
    STDMETHOD_(ID3DX11EffectVariable *, GetMemberBySemantic)(THIS_ LPCSTR Semantic) PURE;

    STDMETHOD_(ID3DX11EffectVariable *, GetElement)(THIS_ UINT Index) PURE;

    STDMETHOD_(ID3DX11EffectConstantBuffer *, GetParentConstantBuffer)(THIS) PURE;

    STDMETHOD_(ID3DX11EffectScalarVariable *, AsScalar)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectVectorVariable *, AsVector)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectMatrixVariable *, AsMatrix)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectStringVariable *, AsString)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectClassInstanceVariable *, AsClassInstance)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectInterfaceVariable *, AsInterface)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectShaderResourceVariable *, AsShaderResource)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectUnorderedAccessViewVariable *, AsUnorderedAccessView)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectRenderTargetViewVariable *, AsRenderTargetView)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectDepthStencilViewVariable *, AsDepthStencilView)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectConstantBuffer *, AsConstantBuffer)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectShaderVariable *, AsShader)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectBlendVariable *, AsBlend)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectDepthStencilVariable *, AsDepthStencil)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectRasterizerVariable *, AsRasterizer)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectSamplerVariable *, AsSampler)(THIS) PURE;

    STDMETHOD(SetRawValue)(THIS_ void *pData, UINT Offset, UINT Count) PURE;
    STDMETHOD(GetRawValue)(THIS_ void *pData, UINT Offset, UINT Count) PURE;

    STDMETHOD(SetConstantBuffer)(THIS_ ID3D11Buffer *pConstantBuffer) PURE;
    STDMETHOD(UndoSetConstantBuffer)(THIS) PURE;
    STDMETHOD(GetConstantBuffer)(THIS_ ID3D11Buffer **ppConstantBuffer) PURE;

    STDMETHOD(SetTextureBuffer)(THIS_ ID3D11ShaderResourceView *pTextureBuffer) PURE;
    STDMETHOD(UndoSetTextureBuffer)(THIS) PURE;
    STDMETHOD(GetTextureBuffer)(THIS_ ID3D11ShaderResourceView **ppTextureBuffer) PURE;
};

//////////////////////////////////////////////////////////////////////////////
// ID3DX11EffectShaderVariable ////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

//----------------------------------------------------------------------------
// D3DX11_EFFECT_SHADER_DESC:
//
// Retrieved by ID3DX11EffectShaderVariable::GetShaderDesc()
//----------------------------------------------------------------------------

typedef struct _D3DX11_EFFECT_SHADER_DESC
{
    CONST BYTE *pInputSignature;            // Passed into CreateInputLayout,
    // valid on VS and GS only

    BOOL IsInline;                          // Is this an anonymous shader variable
    // resulting from an inline shader assignment?


    // -- The following fields are not valid after Optimize() --
    CONST BYTE *pBytecode;                  // Shader bytecode
    UINT BytecodeLength;

    LPCSTR SODecls[D3D11_SO_STREAM_COUNT];  // Stream out declaration string (for GS with SO)
    UINT RasterizedStream;

    UINT NumInputSignatureEntries;          // Number of entries in the input signature
    UINT NumOutputSignatureEntries;         // Number of entries in the output signature
    UINT NumPatchConstantSignatureEntries;  // Number of entries in the patch constant signature
} D3DX11_EFFECT_SHADER_DESC;


typedef interface ID3DX11EffectShaderVariable ID3DX11EffectShaderVariable;
typedef interface ID3DX11EffectShaderVariable *LPD3D11EFFECTSHADERVARIABLE;

// {7508B344-020A-4EC7-9118-62CDD36C88D7}
DEFINE_GUID(IID_ID3DX11EffectShaderVariable,
            0x7508b344, 0x020a, 0x4ec7, 0x91, 0x18, 0x62, 0xcd, 0xd3, 0x6c, 0x88, 0xd7);

#undef INTERFACE
#define INTERFACE ID3DX11EffectShaderVariable

DECLARE_INTERFACE_(ID3DX11EffectShaderVariable, ID3DX11EffectVariable)
{
    STDMETHOD_(ID3DX11EffectType *, GetType)(THIS) PURE;
    STDMETHOD(GetDesc)(THIS_ D3DX11_EFFECT_VARIABLE_DESC *pDesc) PURE;

    STDMETHOD_(ID3DX11EffectVariable *, GetAnnotationByIndex)(THIS_ UINT Index) PURE;
    STDMETHOD_(ID3DX11EffectVariable *, GetAnnotationByName)(THIS_ LPCSTR Name) PURE;

    STDMETHOD_(ID3DX11EffectVariable *, GetMemberByIndex)(THIS_ UINT Index) PURE;
    STDMETHOD_(ID3DX11EffectVariable *, GetMemberByName)(THIS_ LPCSTR Name) PURE;
    STDMETHOD_(ID3DX11EffectVariable *, GetMemberBySemantic)(THIS_ LPCSTR Semantic) PURE;

    STDMETHOD_(ID3DX11EffectVariable *, GetElement)(THIS_ UINT Index) PURE;

    STDMETHOD_(ID3DX11EffectConstantBuffer *, GetParentConstantBuffer)(THIS) PURE;

    STDMETHOD_(ID3DX11EffectScalarVariable *, AsScalar)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectVectorVariable *, AsVector)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectMatrixVariable *, AsMatrix)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectStringVariable *, AsString)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectClassInstanceVariable *, AsClassInstance)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectInterfaceVariable *, AsInterface)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectShaderResourceVariable *, AsShaderResource)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectUnorderedAccessViewVariable *, AsUnorderedAccessView)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectRenderTargetViewVariable *, AsRenderTargetView)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectDepthStencilViewVariable *, AsDepthStencilView)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectConstantBuffer *, AsConstantBuffer)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectShaderVariable *, AsShader)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectBlendVariable *, AsBlend)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectDepthStencilVariable *, AsDepthStencil)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectRasterizerVariable *, AsRasterizer)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectSamplerVariable *, AsSampler)(THIS) PURE;

    STDMETHOD(SetRawValue)(THIS_ void *pData, UINT Offset, UINT Count) PURE;
    STDMETHOD(GetRawValue)(THIS_ void *pData, UINT Offset, UINT Count) PURE;

    STDMETHOD(GetShaderDesc)(THIS_ UINT ShaderIndex, D3DX11_EFFECT_SHADER_DESC *pDesc) PURE;

    STDMETHOD(GetVertexShader)(THIS_ UINT ShaderIndex, ID3D11VertexShader **ppVS) PURE;
    STDMETHOD(GetGeometryShader)(THIS_ UINT ShaderIndex, ID3D11GeometryShader **ppGS) PURE;
    STDMETHOD(GetPixelShader)(THIS_ UINT ShaderIndex, ID3D11PixelShader **ppPS) PURE;
    STDMETHOD(GetHullShader)(THIS_ UINT ShaderIndex, ID3D11HullShader **ppPS) PURE;
    STDMETHOD(GetDomainShader)(THIS_ UINT ShaderIndex, ID3D11DomainShader **ppPS) PURE;
    STDMETHOD(GetComputeShader)(THIS_ UINT ShaderIndex, ID3D11ComputeShader **ppPS) PURE;

    STDMETHOD(GetInputSignatureElementDesc)(THIS_ UINT ShaderIndex, UINT Element, D3D11_SIGNATURE_PARAMETER_DESC *pDesc) PURE;
    STDMETHOD(GetOutputSignatureElementDesc)(THIS_ UINT ShaderIndex, UINT Element, D3D11_SIGNATURE_PARAMETER_DESC *pDesc) PURE;
    STDMETHOD(GetPatchConstantSignatureElementDesc)(THIS_ UINT ShaderIndex, UINT Element, D3D11_SIGNATURE_PARAMETER_DESC *pDesc) PURE;
};

//////////////////////////////////////////////////////////////////////////////
// ID3DX11EffectBlendVariable /////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

typedef interface ID3DX11EffectBlendVariable ID3DX11EffectBlendVariable;
typedef interface ID3DX11EffectBlendVariable *LPD3D11EFFECTBLENDVARIABLE;

// {D664F4D7-3B81-4805-B277-C1DF58C39F53}
DEFINE_GUID(IID_ID3DX11EffectBlendVariable,
            0xd664f4d7, 0x3b81, 0x4805, 0xb2, 0x77, 0xc1, 0xdf, 0x58, 0xc3, 0x9f, 0x53);

#undef INTERFACE
#define INTERFACE ID3DX11EffectBlendVariable

DECLARE_INTERFACE_(ID3DX11EffectBlendVariable, ID3DX11EffectVariable)
{
    STDMETHOD_(ID3DX11EffectType *, GetType)(THIS) PURE;
    STDMETHOD(GetDesc)(THIS_ D3DX11_EFFECT_VARIABLE_DESC *pDesc) PURE;

    STDMETHOD_(ID3DX11EffectVariable *, GetAnnotationByIndex)(THIS_ UINT Index) PURE;
    STDMETHOD_(ID3DX11EffectVariable *, GetAnnotationByName)(THIS_ LPCSTR Name) PURE;

    STDMETHOD_(ID3DX11EffectVariable *, GetMemberByIndex)(THIS_ UINT Index) PURE;
    STDMETHOD_(ID3DX11EffectVariable *, GetMemberByName)(THIS_ LPCSTR Name) PURE;
    STDMETHOD_(ID3DX11EffectVariable *, GetMemberBySemantic)(THIS_ LPCSTR Semantic) PURE;

    STDMETHOD_(ID3DX11EffectVariable *, GetElement)(THIS_ UINT Index) PURE;

    STDMETHOD_(ID3DX11EffectConstantBuffer *, GetParentConstantBuffer)(THIS) PURE;

    STDMETHOD_(ID3DX11EffectScalarVariable *, AsScalar)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectVectorVariable *, AsVector)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectMatrixVariable *, AsMatrix)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectStringVariable *, AsString)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectClassInstanceVariable *, AsClassInstance)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectInterfaceVariable *, AsInterface)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectShaderResourceVariable *, AsShaderResource)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectUnorderedAccessViewVariable *, AsUnorderedAccessView)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectRenderTargetViewVariable *, AsRenderTargetView)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectDepthStencilViewVariable *, AsDepthStencilView)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectConstantBuffer *, AsConstantBuffer)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectShaderVariable *, AsShader)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectBlendVariable *, AsBlend)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectDepthStencilVariable *, AsDepthStencil)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectRasterizerVariable *, AsRasterizer)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectSamplerVariable *, AsSampler)(THIS) PURE;

    STDMETHOD(SetRawValue)(THIS_ void *pData, UINT Offset, UINT Count) PURE;
    STDMETHOD(GetRawValue)(THIS_ void *pData, UINT Offset, UINT Count) PURE;

    STDMETHOD(GetBlendState)(THIS_ UINT Index, ID3D11BlendState **ppBlendState) PURE;
    STDMETHOD(SetBlendState)(THIS_ UINT Index, ID3D11BlendState *pBlendState) PURE;
    STDMETHOD(UndoSetBlendState)(THIS_ UINT Index) PURE;
    STDMETHOD(GetBackingStore)(THIS_ UINT Index, D3D11_BLEND_DESC *pBlendDesc) PURE;
};

//////////////////////////////////////////////////////////////////////////////
// ID3DX11EffectDepthStencilVariable //////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

typedef interface ID3DX11EffectDepthStencilVariable ID3DX11EffectDepthStencilVariable;
typedef interface ID3DX11EffectDepthStencilVariable *LPD3D11EFFECTDEPTHSTENCILVARIABLE;

// {69B5751B-61A5-48E5-BD41-D93988111563}
DEFINE_GUID(IID_ID3DX11EffectDepthStencilVariable,
            0x69b5751b, 0x61a5, 0x48e5, 0xbd, 0x41, 0xd9, 0x39, 0x88, 0x11, 0x15, 0x63);

#undef INTERFACE
#define INTERFACE ID3DX11EffectDepthStencilVariable

DECLARE_INTERFACE_(ID3DX11EffectDepthStencilVariable, ID3DX11EffectVariable)
{
    STDMETHOD_(ID3DX11EffectType *, GetType)(THIS) PURE;
    STDMETHOD(GetDesc)(THIS_ D3DX11_EFFECT_VARIABLE_DESC *pDesc) PURE;

    STDMETHOD_(ID3DX11EffectVariable *, GetAnnotationByIndex)(THIS_ UINT Index) PURE;
    STDMETHOD_(ID3DX11EffectVariable *, GetAnnotationByName)(THIS_ LPCSTR Name) PURE;

    STDMETHOD_(ID3DX11EffectVariable *, GetMemberByIndex)(THIS_ UINT Index) PURE;
    STDMETHOD_(ID3DX11EffectVariable *, GetMemberByName)(THIS_ LPCSTR Name) PURE;
    STDMETHOD_(ID3DX11EffectVariable *, GetMemberBySemantic)(THIS_ LPCSTR Semantic) PURE;

    STDMETHOD_(ID3DX11EffectVariable *, GetElement)(THIS_ UINT Index) PURE;

    STDMETHOD_(ID3DX11EffectConstantBuffer *, GetParentConstantBuffer)(THIS) PURE;

    STDMETHOD_(ID3DX11EffectScalarVariable *, AsScalar)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectVectorVariable *, AsVector)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectMatrixVariable *, AsMatrix)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectStringVariable *, AsString)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectClassInstanceVariable *, AsClassInstance)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectInterfaceVariable *, AsInterface)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectShaderResourceVariable *, AsShaderResource)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectUnorderedAccessViewVariable *, AsUnorderedAccessView)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectRenderTargetViewVariable *, AsRenderTargetView)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectDepthStencilViewVariable *, AsDepthStencilView)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectConstantBuffer *, AsConstantBuffer)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectShaderVariable *, AsShader)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectBlendVariable *, AsBlend)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectDepthStencilVariable *, AsDepthStencil)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectRasterizerVariable *, AsRasterizer)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectSamplerVariable *, AsSampler)(THIS) PURE;

    STDMETHOD(SetRawValue)(THIS_ void *pData, UINT Offset, UINT Count) PURE;
    STDMETHOD(GetRawValue)(THIS_ void *pData, UINT Offset, UINT Count) PURE;

    STDMETHOD(GetDepthStencilState)(THIS_ UINT Index, ID3D11DepthStencilState **ppDepthStencilState) PURE;
    STDMETHOD(SetDepthStencilState)(THIS_ UINT Index, ID3D11DepthStencilState *pDepthStencilState) PURE;
    STDMETHOD(UndoSetDepthStencilState)(THIS_ UINT Index) PURE;
    STDMETHOD(GetBackingStore)(THIS_ UINT Index, D3D11_DEPTH_STENCIL_DESC *pDepthStencilDesc) PURE;
};

//////////////////////////////////////////////////////////////////////////////
// ID3DX11EffectRasterizerVariable ////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

typedef interface ID3DX11EffectRasterizerVariable ID3DX11EffectRasterizerVariable;
typedef interface ID3DX11EffectRasterizerVariable *LPD3D11EFFECTRASTERIZERVARIABLE;

// {53A262F6-5F74-4151-A132-E3DD19A62C9D}
DEFINE_GUID(IID_ID3DX11EffectRasterizerVariable,
            0x53a262f6, 0x5f74, 0x4151, 0xa1, 0x32, 0xe3, 0xdd, 0x19, 0xa6, 0x2c, 0x9d);

#undef INTERFACE
#define INTERFACE ID3DX11EffectRasterizerVariable

DECLARE_INTERFACE_(ID3DX11EffectRasterizerVariable, ID3DX11EffectVariable)
{
    STDMETHOD_(ID3DX11EffectType *, GetType)(THIS) PURE;
    STDMETHOD(GetDesc)(THIS_ D3DX11_EFFECT_VARIABLE_DESC *pDesc) PURE;

    STDMETHOD_(ID3DX11EffectVariable *, GetAnnotationByIndex)(THIS_ UINT Index) PURE;
    STDMETHOD_(ID3DX11EffectVariable *, GetAnnotationByName)(THIS_ LPCSTR Name) PURE;

    STDMETHOD_(ID3DX11EffectVariable *, GetMemberByIndex)(THIS_ UINT Index) PURE;
    STDMETHOD_(ID3DX11EffectVariable *, GetMemberByName)(THIS_ LPCSTR Name) PURE;
    STDMETHOD_(ID3DX11EffectVariable *, GetMemberBySemantic)(THIS_ LPCSTR Semantic) PURE;

    STDMETHOD_(ID3DX11EffectVariable *, GetElement)(THIS_ UINT Index) PURE;

    STDMETHOD_(ID3DX11EffectConstantBuffer *, GetParentConstantBuffer)(THIS) PURE;

    STDMETHOD_(ID3DX11EffectScalarVariable *, AsScalar)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectVectorVariable *, AsVector)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectMatrixVariable *, AsMatrix)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectStringVariable *, AsString)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectClassInstanceVariable *, AsClassInstance)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectInterfaceVariable *, AsInterface)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectShaderResourceVariable *, AsShaderResource)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectUnorderedAccessViewVariable *, AsUnorderedAccessView)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectRenderTargetViewVariable *, AsRenderTargetView)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectDepthStencilViewVariable *, AsDepthStencilView)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectConstantBuffer *, AsConstantBuffer)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectShaderVariable *, AsShader)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectBlendVariable *, AsBlend)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectDepthStencilVariable *, AsDepthStencil)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectRasterizerVariable *, AsRasterizer)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectSamplerVariable *, AsSampler)(THIS) PURE;

    STDMETHOD(SetRawValue)(THIS_ void *pData, UINT Offset, UINT Count) PURE;
    STDMETHOD(GetRawValue)(THIS_ void *pData, UINT Offset, UINT Count) PURE;

    STDMETHOD(GetRasterizerState)(THIS_ UINT Index, ID3D11RasterizerState **ppRasterizerState) PURE;
    STDMETHOD(SetRasterizerState)(THIS_ UINT Index, ID3D11RasterizerState *pRasterizerState) PURE;
    STDMETHOD(UndoSetRasterizerState)(THIS_ UINT Index) PURE;
    STDMETHOD(GetBackingStore)(THIS_ UINT Index, D3D11_RASTERIZER_DESC *pRasterizerDesc) PURE;
};

//////////////////////////////////////////////////////////////////////////////
// ID3DX11EffectSamplerVariable ///////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

typedef interface ID3DX11EffectSamplerVariable ID3DX11EffectSamplerVariable;
typedef interface ID3DX11EffectSamplerVariable *LPD3D11EFFECTSAMPLERVARIABLE;

// {C6402E55-1095-4D95-8931-F92660513DD9}
DEFINE_GUID(IID_ID3DX11EffectSamplerVariable,
            0xc6402e55, 0x1095, 0x4d95, 0x89, 0x31, 0xf9, 0x26, 0x60, 0x51, 0x3d, 0xd9);

#undef INTERFACE
#define INTERFACE ID3DX11EffectSamplerVariable

DECLARE_INTERFACE_(ID3DX11EffectSamplerVariable, ID3DX11EffectVariable)
{
    STDMETHOD_(ID3DX11EffectType *, GetType)(THIS) PURE;
    STDMETHOD(GetDesc)(THIS_ D3DX11_EFFECT_VARIABLE_DESC *pDesc) PURE;

    STDMETHOD_(ID3DX11EffectVariable *, GetAnnotationByIndex)(THIS_ UINT Index) PURE;
    STDMETHOD_(ID3DX11EffectVariable *, GetAnnotationByName)(THIS_ LPCSTR Name) PURE;

    STDMETHOD_(ID3DX11EffectVariable *, GetMemberByIndex)(THIS_ UINT Index) PURE;
    STDMETHOD_(ID3DX11EffectVariable *, GetMemberByName)(THIS_ LPCSTR Name) PURE;
    STDMETHOD_(ID3DX11EffectVariable *, GetMemberBySemantic)(THIS_ LPCSTR Semantic) PURE;

    STDMETHOD_(ID3DX11EffectVariable *, GetElement)(THIS_ UINT Index) PURE;

    STDMETHOD_(ID3DX11EffectConstantBuffer *, GetParentConstantBuffer)(THIS) PURE;

    STDMETHOD_(ID3DX11EffectScalarVariable *, AsScalar)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectVectorVariable *, AsVector)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectMatrixVariable *, AsMatrix)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectStringVariable *, AsString)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectClassInstanceVariable *, AsClassInstance)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectInterfaceVariable *, AsInterface)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectShaderResourceVariable *, AsShaderResource)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectUnorderedAccessViewVariable *, AsUnorderedAccessView)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectRenderTargetViewVariable *, AsRenderTargetView)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectDepthStencilViewVariable *, AsDepthStencilView)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectConstantBuffer *, AsConstantBuffer)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectShaderVariable *, AsShader)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectBlendVariable *, AsBlend)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectDepthStencilVariable *, AsDepthStencil)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectRasterizerVariable *, AsRasterizer)(THIS) PURE;
    STDMETHOD_(ID3DX11EffectSamplerVariable *, AsSampler)(THIS) PURE;

    STDMETHOD(SetRawValue)(THIS_ void *pData, UINT Offset, UINT Count) PURE;
    STDMETHOD(GetRawValue)(THIS_ void *pData, UINT Offset, UINT Count) PURE;

    STDMETHOD(GetSampler)(THIS_ UINT Index, ID3D11SamplerState **ppSampler) PURE;
    STDMETHOD(SetSampler)(THIS_ UINT Index, ID3D11SamplerState *pSampler) PURE;
    STDMETHOD(UndoSetSampler)(THIS_ UINT Index) PURE;
    STDMETHOD(GetBackingStore)(THIS_ UINT Index, D3D11_SAMPLER_DESC *pSamplerDesc) PURE;
};

//////////////////////////////////////////////////////////////////////////////
// ID3DX11EffectPass //////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

//----------------------------------------------------------------------------
// D3DX11_PASS_DESC:
//
// Retrieved by ID3DX11EffectPass::GetDesc()
//----------------------------------------------------------------------------

typedef struct _D3DX11_PASS_DESC
{
    LPCSTR Name;                    // Name of this pass (NULL if not anonymous)
    UINT Annotations;               // Number of annotations on this pass

    BYTE *pIAInputSignature;        // Signature from VS or GS (if there is no VS)
    // or NULL if neither exists
    SIZE_T IAInputSignatureSize;    // Singature size in bytes

    UINT StencilRef;                // Specified in SetDepthStencilState()
    UINT SampleMask;                // Specified in SetBlendState()
    FLOAT BlendFactor[4];           // Specified in SetBlendState()
} D3DX11_PASS_DESC;

//----------------------------------------------------------------------------
// D3DX11_PASS_SHADER_DESC:
//
// Retrieved by ID3DX11EffectPass::Get**ShaderDesc()
//----------------------------------------------------------------------------

typedef struct _D3DX11_PASS_SHADER_DESC
{
    ID3DX11EffectShaderVariable *pShaderVariable;    // The variable that this shader came from.
    // If this is an inline shader assignment,
    //   the returned interface will be an
    //   anonymous shader variable, which is
    //   not retrievable any other way.  It's
    //   name in the variable description will
    //   be "$Anonymous".
    // If there is no assignment of this type in
    //   the pass block, pShaderVariable != NULL,
    //   but pShaderVariable->IsValid() == FALSE.

    UINT                        ShaderIndex;        // The element of pShaderVariable (if an array)
    // or 0 if not applicable
} D3DX11_PASS_SHADER_DESC;

typedef interface ID3DX11EffectPass ID3DX11EffectPass;
typedef interface ID3DX11EffectPass *LPD3D11EFFECTPASS;

// {3437CEC4-4AC1-4D87-8916-F4BD5A41380C}
DEFINE_GUID(IID_ID3DX11EffectPass,
            0x3437cec4, 0x4ac1, 0x4d87, 0x89, 0x16, 0xf4, 0xbd, 0x5a, 0x41, 0x38, 0x0c);

#undef INTERFACE
#define INTERFACE ID3DX11EffectPass

DECLARE_INTERFACE(ID3DX11EffectPass)
{
    STDMETHOD_(BOOL, IsValid)(THIS) PURE;
    STDMETHOD(GetDesc)(THIS_ D3DX11_PASS_DESC *pDesc) PURE;

    STDMETHOD(GetVertexShaderDesc)(THIS_ D3DX11_PASS_SHADER_DESC *pDesc) PURE;
    STDMETHOD(GetGeometryShaderDesc)(THIS_ D3DX11_PASS_SHADER_DESC *pDesc) PURE;
    STDMETHOD(GetPixelShaderDesc)(THIS_ D3DX11_PASS_SHADER_DESC *pDesc) PURE;
    STDMETHOD(GetHullShaderDesc)(THIS_ D3DX11_PASS_SHADER_DESC *pDesc) PURE;
    STDMETHOD(GetDomainShaderDesc)(THIS_ D3DX11_PASS_SHADER_DESC *pDesc) PURE;
    STDMETHOD(GetComputeShaderDesc)(THIS_ D3DX11_PASS_SHADER_DESC *pDesc) PURE;

    STDMETHOD_(ID3DX11EffectVariable *, GetAnnotationByIndex)(THIS_ UINT Index) PURE;
    STDMETHOD_(ID3DX11EffectVariable *, GetAnnotationByName)(THIS_ LPCSTR Name) PURE;

    STDMETHOD(Apply)(THIS_ UINT Flags, ID3D11DeviceContext* pContext) PURE;

    STDMETHOD(ComputeStateBlockMask)(THIS_ D3DX11_STATE_BLOCK_MASK *pStateBlockMask) PURE;
};

//////////////////////////////////////////////////////////////////////////////
// ID3DX11EffectTechnique /////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

//----------------------------------------------------------------------------
// D3DX11_TECHNIQUE_DESC:
//
// Retrieved by ID3DX11EffectTechnique::GetDesc()
//----------------------------------------------------------------------------

typedef struct _D3DX11_TECHNIQUE_DESC
{
    LPCSTR  Name;                   // Name of this technique (NULL if not anonymous)
    UINT    Passes;                 // Number of passes contained within
    UINT    Annotations;            // Number of annotations on this technique
} D3DX11_TECHNIQUE_DESC;

typedef interface ID3DX11EffectTechnique ID3DX11EffectTechnique;
typedef interface ID3DX11EffectTechnique *LPD3D11EFFECTTECHNIQUE;

// {51198831-1F1D-4F47-BD76-41CB0835B1DE}
DEFINE_GUID(IID_ID3DX11EffectTechnique,
            0x51198831, 0x1f1d, 0x4f47, 0xbd, 0x76, 0x41, 0xcb, 0x08, 0x35, 0xb1, 0xde);

#undef INTERFACE
#define INTERFACE ID3DX11EffectTechnique

DECLARE_INTERFACE(ID3DX11EffectTechnique)
{
    STDMETHOD_(BOOL, IsValid)(THIS) PURE;
    STDMETHOD(GetDesc)(THIS_ D3DX11_TECHNIQUE_DESC *pDesc) PURE;

    STDMETHOD_(ID3DX11EffectVariable *, GetAnnotationByIndex)(THIS_ UINT Index) PURE;
    STDMETHOD_(ID3DX11EffectVariable *, GetAnnotationByName)(THIS_ LPCSTR Name) PURE;

    STDMETHOD_(ID3DX11EffectPass *, GetPassByIndex)(THIS_ UINT Index) PURE;
    STDMETHOD_(ID3DX11EffectPass *, GetPassByName)(THIS_ LPCSTR Name) PURE;

    STDMETHOD(ComputeStateBlockMask)(THIS_ D3DX11_STATE_BLOCK_MASK *pStateBlockMask) PURE;
};

//////////////////////////////////////////////////////////////////////////////
// ID3DX11EffectTechnique /////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

//----------------------------------------------------------------------------
// D3DX11_GROUP_DESC:
//
// Retrieved by ID3DX11EffectTechnique::GetDesc()
//----------------------------------------------------------------------------

typedef struct _D3DX11_GROUP_DESC
{
    LPCSTR  Name;                   // Name of this group (only NULL if global)
    UINT    Techniques;             // Number of techniques contained within
    UINT    Annotations;            // Number of annotations on this group
} D3DX11_GROUP_DESC;

typedef interface ID3DX11EffectGroup ID3DX11EffectGroup;
typedef interface ID3DX11EffectGroup *LPD3D11EFFECTGROUP;

// {03074acf-97de-485f-b201-cb775264afd6}
DEFINE_GUID(IID_ID3DX11EffectGroup,
            0x03074acf, 0x97de, 0x485f, 0xb2, 0x01, 0xcb, 0x77, 0x52, 0x64, 0xaf, 0xd6);

#undef INTERFACE
#define INTERFACE ID3DX11EffectGroup

DECLARE_INTERFACE(ID3DX11EffectGroup)
{
    STDMETHOD_(BOOL, IsValid)(THIS) PURE;
    STDMETHOD(GetDesc)(THIS_ D3DX11_GROUP_DESC *pDesc) PURE;

    STDMETHOD_(ID3DX11EffectVariable *, GetAnnotationByIndex)(THIS_ UINT Index) PURE;
    STDMETHOD_(ID3DX11EffectVariable *, GetAnnotationByName)(THIS_ LPCSTR Name) PURE;

    STDMETHOD_(ID3DX11EffectTechnique *, GetTechniqueByIndex)(THIS_ UINT Index) PURE;
    STDMETHOD_(ID3DX11EffectTechnique *, GetTechniqueByName)(THIS_ LPCSTR Name) PURE;
};

//////////////////////////////////////////////////////////////////////////////
// ID3DX11Effect //////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

//----------------------------------------------------------------------------
// D3DX11_EFFECT_DESC:
//
// Retrieved by ID3DX11Effect::GetDesc()
//----------------------------------------------------------------------------

typedef struct _D3DX11_EFFECT_DESC
{
    UINT    ConstantBuffers;        // Number of constant buffers in this effect
    UINT    GlobalVariables;        // Number of global variables in this effect
    UINT    InterfaceVariables;     // Number of global interfaces in this effect
    UINT    Techniques;             // Number of techniques in this effect
    UINT    Groups;                 // Number of groups in this effect
} D3DX11_EFFECT_DESC;

typedef interface ID3DX11Effect ID3DX11Effect;
typedef interface ID3DX11Effect *LPD3D11EFFECT;

// {FA61CA24-E4BA-4262-9DB8-B132E8CAE319}
DEFINE_GUID(IID_ID3DX11Effect,
            0xfa61ca24, 0xe4ba, 0x4262, 0x9d, 0xb8, 0xb1, 0x32, 0xe8, 0xca, 0xe3, 0x19);

#undef INTERFACE
#define INTERFACE ID3DX11Effect

DECLARE_INTERFACE_(ID3DX11Effect, IUnknown)
{
    // IUnknown
    STDMETHOD(QueryInterface)(THIS_ REFIID iid, LPVOID *ppv) PURE;
    STDMETHOD_(ULONG, AddRef)(THIS) PURE;
    STDMETHOD_(ULONG, Release)(THIS) PURE;

    STDMETHOD_(BOOL, IsValid)(THIS) PURE;

    // Managing D3D Device
    STDMETHOD(GetDevice)(THIS_ ID3D11Device **ppDevice) PURE;

    // New Reflection APIs
    STDMETHOD(GetDesc)(THIS_ D3DX11_EFFECT_DESC *pDesc) PURE;

    STDMETHOD_(ID3DX11EffectConstantBuffer *, GetConstantBufferByIndex)(THIS_ UINT Index) PURE;
    STDMETHOD_(ID3DX11EffectConstantBuffer *, GetConstantBufferByName)(THIS_ LPCSTR Name) PURE;

    STDMETHOD_(ID3DX11EffectVariable *, GetVariableByIndex)(THIS_ UINT Index) PURE;
    STDMETHOD_(ID3DX11EffectVariable *, GetVariableByName)(THIS_ LPCSTR Name) PURE;
    STDMETHOD_(ID3DX11EffectVariable *, GetVariableBySemantic)(THIS_ LPCSTR Semantic) PURE;

    STDMETHOD_(ID3DX11EffectGroup *, GetGroupByIndex)(THIS_ UINT Index) PURE;
    STDMETHOD_(ID3DX11EffectGroup *, GetGroupByName)(THIS_ LPCSTR Name) PURE;

    STDMETHOD_(ID3DX11EffectTechnique *, GetTechniqueByIndex)(THIS_ UINT Index) PURE;
    STDMETHOD_(ID3DX11EffectTechnique *, GetTechniqueByName)(THIS_ LPCSTR Name) PURE;

    STDMETHOD_(ID3D11ClassLinkage *, GetClassLinkage)(THIS) PURE;

    STDMETHOD(CloneEffect)(THIS_ UINT Flags, ID3DX11Effect **ppClonedEffect) PURE;
    STDMETHOD(Optimize)(THIS) PURE;
    STDMETHOD_(BOOL, IsOptimized)(THIS) PURE;
};

//////////////////////////////////////////////////////////////////////////////
// APIs //////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#ifdef __cplusplus
extern "C" {
#endif //__cplusplus

//----------------------------------------------------------------------------
// D3DX11CreateEffectFromMemory:
// --------------------------
// Creates an effect from a binary effect or file
//
// Parameters:
//
// [in]
//
//
//  pData
//      Blob of compiled effect data
//  DataLength
//      Length of the data blob
//  FXFlags
//      Compilation flags pertaining to Effect compilation, honored
//      by the Effect compiler
//  pDevice
//      Pointer to the D3D11 device on which to create Effect resources
//
// [out]
//
//  ppEffect
//      Address of the newly created Effect interface
//
//----------------------------------------------------------------------------

HRESULT WINAPI D3DX11CreateEffectFromMemory(void *pData, SIZE_T DataLength, UINT FXFlags, ID3D11Device *pDevice, ID3DX11Effect **ppEffect);

#ifdef __cplusplus
}
#endif //__cplusplus

#endif //__D3DX11EFFECT_H__

