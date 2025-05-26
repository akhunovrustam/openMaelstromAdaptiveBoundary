#pragma once
#ifdef __CUDACC__
#pragma nv_diag_suppress = field_without_dll_interface	
#pragma nv_diag_suppress = useless_type_qualifier_on_return_type	
#endif
#include <string>
#include <array>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <limits>
#include <limits.h>
#include <float.h>
#include <sstream>
#include <tuple>
#ifndef __CUDACC__
#include <filesystem>
#endif
#include <math/math.h>
#include <math/unit_math.h>
#include <math/matrix.h>
#ifndef WIN32
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wparentheses"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunknown-pragmas"
#endif
#ifdef __CUDACC__
#pragma nv_diag_suppress = declared_but_not_referenced		
#endif
#include <gvdb.h>
#ifndef WIN32
#pragma GCC diagnostic pop
#endif
//#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
//#include <experimental/filesystem>
#ifdef __CUDACC__
#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#include <experimental/filesystem>
namespace stdfs = std::experimental::filesystem;
#else
namespace stdfs = std::filesystem;
#endif

#if !defined(_WIN32) || defined(__clang__)
#define TEMPLATE_TOKEN template
#else
#define TEMPLATE_TOKEN
#endif
 
#define FLUID_PARTICLE 0
#define RIGID_PARTICLE 1

//#define RIGID_PARTICLE_SUPPORT

//#define ANISOTROPIC_SURFACE
#define ZHU_BRIDSON_SURFACE
//#define BLOBBY_SURFACE
//#define ISO_DENSITY_SURFACE

enum class resource_t : int {
	array_type,
	uniform_type,
	swap_type,
	aggregate_uniform_type
};

enum struct memory_kind {
	particleData, cellData, customData, diffuseData, singleData, rigidData, spareData, individualData
};

template <typename T> struct complex_type {
	using type = T;
	std::string jsonName;
	T value;
};

struct complex_uniform {};

enum struct launch_config { device, host, debug, pure_host, _used_for_template_specializations };
enum struct hash_length { bit_64, bit_32 };
enum struct cell_ordering { z_order, linear_order };
enum struct cell_structuring { hashed, MLM, complete, compactMLM};
enum struct neighbor_list { basic, constrained, cell_based, compactCell, compactMLM, masked };


#define MAX_VAL_00BIT 0
#define MAX_VAL_01BIT 1
#define MAX_VAL_02BIT 3
#define MAX_VAL_03BIT 7
#define MAX_VAL_04BIT 15
#define MAX_VAL_05BIT 31
#define MAX_VAL_06BIT 63
#define MAX_VAL_07BIT 127
#define MAX_VAL_08BIT 255
#define MAX_VAL_09BIT 511
#define MAX_VAL_10BIT 1023
#define MAX_VAL_11BIT 2047
#define MAX_VAL_12BIT 4095
#define MAX_VAL_13BIT 8191
#define MAX_VAL_14BIT 16383
#define MAX_VAL_15BIT 32767
#define MAX_VAL_16BIT 65535
#define MAX_VAL_17BIT 131071
#define MAX_VAL_18BIT 262143
#define MAX_VAL_19BIT 524287
#define MAX_VAL_20BIT 1048575
#define MAX_VAL_21BIT 2097151
#define MAX_VAL_22BIT 4194303
#define MAX_VAL_23BIT 8388607
#define MAX_VAL_24BIT 16777215
#define MAX_VAL_25BIT 33554431
#define MAX_VAL_26BIT 67108863
#define MAX_VAL_27BIT 134217727
#define MAX_VAL_28BIT 268435455
#define MAX_VAL_29BIT 536870911
#define MAX_VAL_30BIT 1073741823
#define MAX_VAL_31BIT 2147483647
#define MAX_VAL_32BIT 4294967295

#define MAX_VAL_00BITU 0u
#define MAX_VAL_01BITU 1u
#define MAX_VAL_02BITU 3u
#define MAX_VAL_03BITU 7u
#define MAX_VAL_04BITU 15u
#define MAX_VAL_05BITU 31u
#define MAX_VAL_06BITU 63u
#define MAX_VAL_07BITU 127u
#define MAX_VAL_08BITU 255u
#define MAX_VAL_09BITU 511u
#define MAX_VAL_10BITU 1023u
#define MAX_VAL_11BITU 2047u
#define MAX_VAL_12BITU 4095u
#define MAX_VAL_13BITU 8191u
#define MAX_VAL_14BITU 16383u
#define MAX_VAL_15BITU 32767u
#define MAX_VAL_16BITU 65535u
#define MAX_VAL_17BITU 131071u
#define MAX_VAL_18BITU 262143u
#define MAX_VAL_19BITU 524287u
#define MAX_VAL_20BITU 1048575u
#define MAX_VAL_21BITU 2097151u
#define MAX_VAL_22BITU 4194303u
#define MAX_VAL_23BITU 8388607u
#define MAX_VAL_24BITU 16777215u
#define MAX_VAL_25BITU 33554431u
#define MAX_VAL_26BITU 67108863u
#define MAX_VAL_27BITU 134217727u
#define MAX_VAL_28BITU 268435455u
#define MAX_VAL_29BITU 536870911u
#define MAX_VAL_30BITU 1073741823u
#define MAX_VAL_31BITU 2147483647u
#define MAX_VAL_32BITU 4294967295u


//#define BASIC_NEIGHBORLIST_COMPILED
#define CONSTRAINED_NEIGHBORLIST_COMPILED
//#define CELLBASED_NEIGHBORLIST_COMPILED
//#define COMPACTCELL_NEIGHBORLIST_COMPILED
#define COMPACTMLM_NEIGHBORLIST_COMPILED
//#define MASKED_NEIGHBORLIST_COMPILED

//#define MLM32_CELL_ALGORITHM
//#define HASHED64_CELL_ALGORITHM
//#define HASHED32_CELL_ALGORITHM
//#define MLM64_CELL_ALGORITHM
#define COMPACT_MLM_CELL_ALGORITHM
//#define LINEAR_CELL_ALGORITHM

//#define BITFIELD_STRUCTURES
#define BITFIELD_WIDTH 25
#define INVALID_BEG_VALUE ((1 << BITFIELD_WIDTH)-1) 

#ifndef BITFIELD_STRUCTURES
struct neigh_span {
	uint32_t beginning;
	uint32_t length;
};
struct hash_span {
  uint32_t beginning;
  uint32_t length;
};
#else
struct neigh_span {
	uint32_t beginning : BITFIELD_WIDTH;
	uint32_t length : (32 - BITFIELD_WIDTH);
};
struct hash_span {
  uint32_t beginning : BITFIELD_WIDTH;
  uint32_t length : (32 - BITFIELD_WIDTH);
};
#endif
using cell_span = hash_span;

#define OFFSET_INVALID 0x00
struct compactCellNeighbors {
	uint32_t xi : 2;
	uint32_t yi : 2;
	uint32_t zi : 2;
	uint32_t mask : 26;
};

#define COMPACT_IDX 25
#define COMPACT_LEN (32 - COMPACT_IDX)
#define COMPACT_LEN_MAX ((1 << COMPACT_LEN)-1) 
#define COMPACT_IDX_MAX ((1 << COMPACT_IDX)-1) 

struct compactSpan {
	uint32_t idx : COMPACT_IDX;
	uint32_t len : COMPACT_LEN;
};

struct cellInformation {
	cellInformation() : occupied (0), inside(0), xPos(0), xNeg(0), yPos(0), yNeg(0), zPos(0), zNeg(0) {}
	uint8_t occupied : 1;
	uint8_t inside : 1;

	uint8_t xPos : 1;
	uint8_t xNeg : 1;
	uint8_t yPos : 1;
	uint8_t yNeg : 1;
	uint8_t zPos : 1;
	uint8_t zNeg : 1;
};

struct cellSurface {
	float dummy;
};

#define UINT31_MAX 2147483647u

#ifdef BITFIELD_STRUCTURES
#define INVALID_BEG MAX_VAL_25BITU
#define INVALID_LEN MAX_VAL_05BITU
struct compactHashSpan {
  uint32_t compacted : 1;
  uint32_t beginning : 25;
  uint32_t length : 5;
};
#else
#define INVALID_BEG MAX_VAL_31BITU
#define INVALID_LEN MAX_VAL_32BITU
struct compactHashSpan {
	uint32_t compacted : 1;
	uint32_t beginning : 31;
	uint32_t length;
}
;

#endif
using compact_cellSpan = compactHashSpan;

#define LIST_ALWAYS_FALSE 0b00
#define LIST_ALWAYS_TRUE 0b01
#define LIST_COMPACT 0b10
#define LIST_ITERATE 0b11

struct compactListEntry {
	struct hashEntry {
		uint32_t kind : 2;
		uint32_t beginning : 25;
		uint32_t length : 5;
	};
	struct cellEntry {
		uint32_t kind : 2;
		uint32_t hash : 30;
	};
	union {
		hashEntry hash;
		cellEntry cell;
	};
};



#define cudaAllocateMemory cudaMallocManaged


//#define DEBUG_INVALID_PARITLCES
#define _VEC(vec) vec.x, vec.y, vec.z, vec.w
#define _VECSTR "[%+.8e %+.8e %+.8e %+.8e]"

#ifndef __CUDACC__
template <typename T> std::string type_name() {
	if constexpr (std::is_same<T, float4>::value)
		return "float4";
	if constexpr (std::is_same<T, float3>::value)
		return "float3";
	if constexpr (std::is_same<T, float2>::value)
		return "float2";
	if constexpr (std::is_same<T, uint4>::value)
		return "uint4";
	if constexpr (std::is_same<T, uint3>::value)
		return "uint3";
	if constexpr (std::is_same<T, uint2>::value)
		return "uint2";
	if constexpr (std::is_same<T, float>::value)
		return "float";
	if constexpr (std::is_same<T, uint32_t>::value)
		return "uint";
	if constexpr (std::is_same<T, int32_t>::value)
		return "int";
	return typeid(T).name();
}
#endif

#include <math/nonesuch.h>

template <class T> using rear_ptr_t = decltype(T::rear_ptr);                                          
template <class Ptr> using rear_ptr_type_template = typename SI::detail::detector<std::ptrdiff_t, void, rear_ptr_t, Ptr>::type; 
template <typename T>                                                                       
constexpr bool has_rear_ptr = !std::is_same<rear_ptr_type_template<T>, std::ptrdiff_t>::value;

//#ifndef __CUDACC__
//inline VolumeGVDB gvdbInstance;
//#endif
#include <texture_types.h>
#include <surface_types.h>

namespace gvdb{
#define ALIGN_GVDB(x)	__align__(x)
#define ID_UNDEFI	0xFFFF
#define ID_UNDEFL	0xFFFFFFFF
//#define ID_UNDEF64	0xFFFFFFFFFFFFFFFFll
#define CHAN_UNDEF	255
#define MAX_CHANNEL  32

	struct ALIGN_GVDB(16) VDBNode {
		uchar		mLev;			// Level		Max = 255			1 byte
		uchar		mFlags;
		uchar		mPriority;
		uchar		pad;
		int3		mPos;			// Pos			Max = +/- 4 mil (linear space/range)	12 bytes
		int3		mValue;			// Value		Max = +8 mil		4 bytes
		float3		mVRange;
		uint64		mParent;		// Parent ID						8 bytes
		uint64		mChildList;		// Child List						8 bytes
		uint64		mMask;			// Bitmask starts - Must keep here, even if not USE_BITMASKS
	};


	struct ALIGN_GVDB(16) VDBAtlasNode {
		int3		mPos;
		int			mLeafID;
	};
	struct ALIGN_GVDB(16) VDBInfo {
		int			dim[10];
		int			res[10];
		float3		vdel[10];
		int3		noderange[10];
		int			nodecnt[10];
		int			nodewid[10];
		int			childwid[10];
		char* nodelist[10];
		char* childlist[10];
		VDBAtlasNode* atlas_map;
		int3		atlas_cnt;
		int3		atlas_res;
		int			atlas_apron;
		int			brick_res;
		int			apron_table[8];
		int			top_lev;
		int			max_iter;
		float		epsilon;
		bool		update;
		uchar		clr_chan;
		float3		bmin;
		float3		bmax;
		cudaTextureObject_t		volIn[MAX_CHANNEL];
		cudaSurfaceObject_t		volOut[MAX_CHANNEL];
	};

	inline __host__ __device__  float3 make_float3(float x, float y, float z)
	{
		float3 t; t.x = x; t.y = y; t.z = z; return t;
	}
	inline __host__ __device__ float3 make_float3(int3 a)
	{
		return make_float3(float(a.x), float(a.y), float(a.z));
	}
	inline __host__ __device__ float3 make_float3(nvdb::Vector3DI a)
	{
		return make_float3(float(a.x), float(a.y), float(a.z));
	}
	inline __host__ __device__ float3 make_float3(float s)
	{
		return make_float3(s, s, s);
	}
	inline __host__ __device__  int3 make_int3(int x, int y, int z)
	{
		int3 t; t.x = x; t.y = y; t.z = z; return t;
	}
	inline __host__ __device__ int3 make_int3(float3 a)
	{
		return make_int3(int(a.x), int(a.y), int(a.z));
	}

	// get node at a specific level and pool index
	inline __device__ gvdb::VDBNode* getNode(gvdb::VDBInfo* gvdb, int lev, int n)
	{
		gvdb::VDBNode* node = (gvdb::VDBNode*)(gvdb->nodelist[lev] + n * gvdb->nodewid[lev]);
		return node;
	}

	// get node at a specific level and pool index
	inline __device__ gvdb::VDBNode* getNode(gvdb::VDBInfo* gvdb, int lev, int n, float3* vmin)
	{
		gvdb::VDBNode* node = (gvdb::VDBNode*)(gvdb->nodelist[lev] + n * gvdb->nodewid[lev]);
		*vmin = make_float3(node->mPos);
		return node;
	}
	inline __device__ int getChild(gvdb::VDBInfo* gvdb, gvdb::VDBNode* node, int b)
	{
		uint64 listid = node->mChildList;
#ifndef WIN32
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Woverflow"
#endif
#ifdef __CUDACC__
#pragma nv_diag_suppress = integer_truncated	
#endif
		if (listid == ID_UNDEFL) return ID_UNDEF64;
#ifndef WIN32
#pragma GCC diagnostic pop
#endif
		uchar clev = uchar((listid >> 8) & 0xFF);
		int cndx = listid >> 16;
		uint64* clist = (uint64*)(gvdb->childlist[clev] + cndx * gvdb->childwid[clev]);
		int c = (*(clist + b)) >> 16;
		return c;
	}

	inline __device__ bool isBitOn(VDBInfo* gvdb, gvdb::VDBNode* node, int b)
	{
		return getChild(gvdb, node, b) != ID_UNDEF64;
	}

	// iteratively find the leaf node at the given position
	inline __device__ gvdb::VDBNode* getNode(gvdb::VDBInfo* gvdb, int lev, int start_id, float3 pos, uint64* node_id)
	{
		float3 vmin, vmax;
		int3 p;
		int b;
		*node_id = ID_UNDEFL;

		gvdb::VDBNode* node = getNode(gvdb, lev, start_id, &vmin);		// get starting node
		while (lev > 0 && node != 0x0) {
			// is point inside node? if no, exit
			vmax = vmin + make_float3(gvdb->noderange[lev]);
			if (pos.x < vmin.x || pos.y < vmin.y || pos.z < vmin.z || pos.x >= vmax.x || pos.y >= vmax.y || pos.z >= vmax.z) {
				*node_id = ID_UNDEFL;
				return 0x0;
			}
			p = make_int3((pos - vmin) / gvdb->vdel[lev]);		// check child bit
			b = (((int(p.z) << gvdb->dim[lev]) + int(p.y)) << gvdb->dim[lev]) + int(p.x);
			lev--;
			if (isBitOn(gvdb, node, b)) {						// child exists, recurse down tree
				*node_id = getChild(gvdb, node, b);				// get next node_id
				node = getNode(gvdb, lev, *node_id, &vmin);
			}
			else {
				*node_id = ID_UNDEFL;
				return 0x0;										// no child, exit
			}
		}
		return node;
	}
	inline __device__ gvdb::VDBNode* getNodeAtPoint(gvdb::VDBInfo* gvdb, float3 pos, float3* offs, float3* vmin, float3* vdel, uint64* node_id)
	{
		// iteratively get node at world point
		gvdb::VDBNode* node = getNode(gvdb, gvdb->top_lev, 0, pos, node_id);
		if (node == 0x0) return 0x0;

		// compute node bounding box
		*vmin = make_float3(node->mPos);
		*vdel = gvdb->vdel[node->mLev];
		*offs = make_float3(node->mValue);
		return node;
	}
}
