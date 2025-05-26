#define NO_QT
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_surface_types.h>
#include <surface_functions.h>
//#if defined(__clang__) && defined(__CUDA__)
//#endif
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>
#include <render/vrtxRender/vrtxRender.h>
//#include <render\rigidRender\rigid_render.h>
#include <sstream>
#include <texture_types.h>
#include <math/math.h>
#include <vector_functions.h>
#include <vector_types.h>

//#define BRIDSON_05
//#define SMALL_SCALE_DETAIL
//#define ISO_DENSITY
//#define ANISOTROPIC
#pragma region MACROS
//#ifdef __INTELLISENSE__
//#define gridDim \
//  (int3 { 32, 1, 1 })
//#define blockDim \
//  (int3 { 32, 1, 1 })
//#define threadIdx \
//  (int3 { 0, 0, 0 })
//#define blockIdx \
//  (int3 { 0, 0, 0 })
//#endif
#ifdef __INTELLISENSE__
#define CPSYMBOL(symbol, var)
#define LAUNCH(kernel, blocks, tpb, sm, stream) kernel
#else
#define CPSYMBOL(symbol, var) cudaMemcpyToSymbol(symbol, &var, sizeof(symbol))
#define LAUNCH(kernel, blocks, tpb, sm, stream) kernel<<<blocks, tpb, sm, stream>>>
#endif
#pragma endregion

#define TWO_PI 6.2831853071795864769252867665590057683943f
#define NUDGE_FACTOR 1e-3f // epsilon
#define samps 1            // samples
#define BVH_STACK_SIZE 32

namespace vrtx {

__device__ bool RayIntersectsBox(const gpuBVH &bvh, const float3 &originInWorldSpace, const float3 &rayInWorldSpace,
                                 int boxIdx) {
  float Tnear, Tfar;
  Tnear = -FLT_MAX;
  Tfar = FLT_MAX;

  float2 limits;

#define CHECK_NEAR_AND_FAR_INTERSECTION(c)                                                                             \
  if (rayInWorldSpace.c == 0.f) {                                                                                      \
    if (originInWorldSpace.c < limits.x)                                                                               \
      return false;                                                                                                    \
    if (originInWorldSpace.c > limits.y)                                                                               \
      return false;                                                                                                    \
  } else {                                                                                                             \
    float T1 = (limits.x - originInWorldSpace.c) / rayInWorldSpace.c;                                                  \
    float T2 = (limits.y - originInWorldSpace.c) / rayInWorldSpace.c;                                                  \
    if (T1 > T2) {                                                                                                     \
      float tmp = T1;                                                                                                  \
      T1 = T2;                                                                                                         \
      T2 = tmp;                                                                                                        \
    }                                                                                                                  \
    if (T1 > Tnear)                                                                                                    \
      Tnear = T1;                                                                                                      \
    if (T2 < Tfar)                                                                                                     \
      Tfar = T2;                                                                                                       \
    if (Tnear > Tfar)                                                                                                  \
      return false;                                                                                                    \
    if (Tfar < 0.f)                                                                                                    \
      return false;                                                                                                    \
  }
  auto lim = bvh.cudaBVHlimits[boxIdx];
  limits = float2{lim.bottom.x, lim.top.x};
  CHECK_NEAR_AND_FAR_INTERSECTION(x)
  limits = float2{lim.bottom.y, lim.top.y};
  CHECK_NEAR_AND_FAR_INTERSECTION(y)
  limits = float2{lim.bottom.z, lim.top.z};
  CHECK_NEAR_AND_FAR_INTERSECTION(z)
  return true;
}

__device__ bool BVH_IntersectTriangles(gpuBVH &bvh, const float3 &origin, const float3 &ray, unsigned avoidSelf,
                                       int &pBestTriIdx, float3 &pointHitInWorldSpace, float &kAB, float &kBC,
                                       float &kCA, float &hitdist, float3 &boxnormal) {
  pBestTriIdx = -1;
  float bestTriDist;
  bestTriDist = FLT_MAX;
  int32_t stack[BVH_STACK_SIZE];
  int32_t stackIdx = 0;
  stack[stackIdx++] = 0;
  while (stackIdx) {
    int32_t boxIdx = stack[stackIdx - 1];
    stackIdx--;
    uint4 data = bvh.cudaBVHindexesOrTrilists[boxIdx];
    if (!(data.x & 0x80000000)) { // INNER NODE
      if (RayIntersectsBox(bvh, origin, ray, boxIdx)) {
        stack[stackIdx++] = data.y;
        stack[stackIdx++] = data.z;
        if (stackIdx > BVH_STACK_SIZE) {
          return false;
        }
      }
    } else {
      for (uint32_t i = data.w; i < data.w + (data.x & 0x7fffffff); i++) {
        int32_t idx = bvh.cudaTriIdxList[i];
        if (avoidSelf == idx)
          continue;
        float4 normal = bvh.cudaTriangleIntersectionData[idx].normal;
        float d = math::sqlength3(normal);
        float k = math::dot3(normal, ray);
        if (k == 0.0f)
          continue;
        float s = (normal.w - math::dot3(normal, origin)) / k;
        if (s <= 0.0f)
          continue;
        if (s <= NUDGE_FACTOR)
          continue;
        float3 hit = ray * s;
        hit += origin;

        float4 ee1 = bvh.cudaTriangleIntersectionData[idx].e1d1;
        float kt1 = math::dot3(ee1, hit) - ee1.w;
        if (kt1 < 0.0f)
          continue;
        float4 ee2 = bvh.cudaTriangleIntersectionData[idx].e2d2;
        float kt2 = math::dot3(ee2, hit) - ee2.w;
        if (kt2 < 0.0f)
          continue;
        float4 ee3 = bvh.cudaTriangleIntersectionData[idx].e3d3;
        float kt3 = math::dot3(ee3, hit) - ee3.w;
        if (kt3 < 0.0f)
          continue;
        {
          float hitZ = math::sqdistance(origin, hit);
          if (hitZ < bestTriDist) {
            bestTriDist = hitZ;
            hitdist = sqrtf(bestTriDist);
            pBestTriIdx = idx;
            pointHitInWorldSpace = hit;
            kAB = kt1;
            kBC = kt2;
            kCA = kt3;
          }
        }
      }
    }
  }

  return pBestTriIdx != -1;
}
__shared__ extern float sm_data[];
struct rayState {
  uint32_t rayDone : 1;
  uint32_t threadDone : 1;
  uint32_t rayBounced : 1;
  uint32_t rayHitFluidAABB : 1;
  uint32_t rayHitFluidSurface : 1;
  int32_t index : 27;
};
#pragma region types
#pragma endregion
#pragma region globals
Pixel *cuImage;
worldRay *cuCurrentRays;
worldRay *cuCompactedRays;
// int32_t *cuInternalFlag;
int32_t *rayCounter;
uint32_t *cRNGSeeds;
int32_t *cuResortIndex;
int32_t *cuResortKey;
float *cufluidDepth;
float4 *cufluidIntersection;
float4 *cuFluidColor;
Box *cuBoxes = nullptr;
Sphere *cuSpheres = nullptr;

__device__ __constant__ SceneInformation cScene;
__device__ __constant__ vrtxFluidMemory fluidMemory;
//__device__ __constant__ Box boxes[] = {
//	//{{-25.f, -25.f, 96.f},{25.f,25.f, 132.f},{1.f,1.f,1.f}, {0.f,0.f,0.f}, DIFF},
//	{{190.f, -192.f, -192.f},{192.f,192.f, 192.f},{1.f,1.f,1.f}, {0.f,0.f,0.f}, DIFF}
//	,{ {-521, -FLT_MAX, -FLT_MAX},{-51, FLT_MAX, FLT_MAX},{0.f,0.f,0.f}, {1.f, 1.f, 1.f}, DIFF}
//	//,{ {-FLT_MAX, -25.f, -FLT_MAX},{32, FLT_MAX, FLT_MAX},{0.f,0.f,0.f}, {1.f, 1.f, 1.f}, DIFF}
//};
//__device__ __constant__ Sphere spheres[] = {
//    //{16, {192.0f, 192, 192}, {1.f, 1.f, 1.f}, {0.f, 0.f, 0.f}, DIFF},
//	{32, {-96, 0, 16}, {0, 0, 0}, {1.f, 1.f, 1.f}, SPEC},
//	{32, {-96, -64, 16}, {0, 0, 0}, {0.5f, 0.f, 0.f}, DIFF},
//	{32, {-96, 64, 64}, {0, 0, 0}, {1.0f, 1.f, 1.f}, REFR},
//	{10000, {50.0f, 40.8f, -1060}, {0.55f, 0.55f, 0.55f}, {0.075f, 0.075f, 0.075f}, DIFF},
//    //{10000, {50.0f, 40.8f, -1060}, {0.55, 0.55, 0.55}, {0.175f, 0.175f, 0.175f}, DIFF},
//	//{10000, {50.0f, 40.8f, -1060}, {0.f,0.f,0.f}, {0.f,0.f,0.f}, DIFF},
//
//    {100000, {0.0f, 0, -100000.}, {0, 0, 0}, {0.2f, 0.2f, 0.2f}, DIFF},
//    {100000, {0.0f, 0, -100000.1}, {0, 0, 0}, {0.3f, 0.3f, 0.3f}, DIFF}};
__device__ __constant__ Box *cBoxes;
__device__ __constant__ Sphere *cSpheres;
__device__ __constant__ int32_t cNumBoxes;
__device__ __constant__ int32_t cNumSpheres;

__device__ __constant__ int32_t cNumRays;
__device__ __constant__ worldRay *cRaysDepth;
__device__ __constant__ worldRay *cCompactRays;
//__device__ __constant__ int32_t *cInternalFlag;
__device__ __constant__ int32_t *cRayCounter;
__device__ __constant__ Pixel *cImage;
__device__ __constant__ uint32_t *cuSeeds;
__device__ __constant__ float *cfluidDepth;
__device__ __constant__ int32_t *cResortIndex;
__device__ __constant__ int32_t *cResortKey;
__device__ __constant__ float4 *cfluidIntersection;
__device__ __constant__ float4 *cFluidColor;
__device__ __constant__ vrtxFluidArrays arrays;

__device__ auto radiusFromVolume(float volume) { return powf(volume * PI4O3_1, 1.f / 3.f); }
__device__ auto radiusFromVolume(int32_t i) { return powf(arrays.auxIsoDensity[i] * PI4O3_1, 1.f / 3.f); }
#define G(i, x, y) (arrays.anisotropicMatrices[i * 1 + arrays.num_ptcls * (x * 3 + y)])

surface<void, cudaSurfaceType2D> surfaceWriteOut;
#pragma endregion
#pragma region helper_functions
__device__ __host__ __inline__ int8_t sgn(float x) { return x > 0.f ? 1 : (x < 0.f ? -1 : 0); }
__device__ auto randf(int32_t index) {
  auto x = cuSeeds[index];
  x ^= x >> 13;
  x ^= x << 17;
  x ^= x >> 5;
  cuSeeds[index] = x;
  auto r = (x & 0x007FFFFF) | 0x3F800000;
  return *reinterpret_cast<float *>(&r) - 1.f;
}
hostDeviceInline int32_t splitInt(uint32_t a) {
  uint32_t x = a;
  x = (x | (x << 16u)) & 0x030000FFu;
  x = (x | (x << 8u)) & 0x0300F00Fu;
  x = (x | (x << 4u)) & 0x030C30C3u;
  x = (x | (x << 2u)) & 0x09249249u;
  return x;
}
hostDeviceInline int32_t splitInt(int32_t a) {
  int32_t x = a;
  x = (x | (x << 16)) & 0x030000FF;
  x = (x | (x << 8)) & 0x0300F00F;
  x = (x | (x << 4)) & 0x030C30C3;
  x = (x | (x << 2)) & 0x09249249;
  return x;
}
hostDeviceInline uint32_t zEncode(uint32_t x, uint32_t y, uint32_t z) {
  uint32_t answer = 0u;
  answer |= splitInt(x) | splitInt(y) << 1u | splitInt(z) << 2u;
  return answer;
}
hostDeviceInline int32_t zEncode(int32_t x, int32_t y, int32_t z) {
  int32_t answer = 0;
  answer |= splitInt(x) | splitInt(y) << 1 | splitInt(z) << 2;
  return answer;
}
hostDeviceInline auto i3toZ(int3 idx) { return zEncode(idx.x, idx.y, idx.z); }
hostDeviceInline auto ui3toZ(uint3 idx) { return zEncode(idx.x, idx.y, idx.z); }
hostDeviceInline auto getVoxel(float4 p, float3 min, float3 d) {
  int32_t i = static_cast<int32_t>(floorf((p.x - min.x) / d.x));
  int32_t j = static_cast<int32_t>(floorf((p.y - min.y) / d.y));
  int32_t k = static_cast<int32_t>(floorf((p.z - min.z) / d.z));
  return int3{i, j, k};
}
hostDeviceInline auto getVoxel(float3 p, float3 min, float3 d) {
  int32_t i = static_cast<int32_t>(floorf((p.x - min.x) / d.x));
  int32_t j = static_cast<int32_t>(floorf((p.y - min.y) / d.y));
  int32_t k = static_cast<int32_t>(floorf((p.z - min.z) / d.z));
  return int3{i, j, k};
}
hostDeviceInline auto iCoord(float4 p) { return getVoxel(p, arrays.min_coord, arrays.cell_size); }
hostDeviceInline auto iCoord(float3 p) { return getVoxel(p, arrays.min_coord, arrays.cell_size); }
// hostDeviceInline auto iCoord(float4 p, float3 m, float3 d) { return getVoxel(p, m, d); }
// hostDeviceInline auto iCoord(float3 p, float3 m, float3 d) { return getVoxel(p, m, d); }
hostDeviceInline auto zCoord(int3 idx) { return i3toZ(idx); }
hostDeviceInline auto zCoord(uint3 idx) { return ui3toZ(idx); }
hostDeviceInline auto zCoord(float4 p) { return zCoord(iCoord(p)); }
hostDeviceInline auto zCoord(float3 p) { return zCoord(iCoord(p)); }
// hostDeviceInline auto zCoord(float4 p, float3 m, float3 d) { return zCoord(iCoord(p, m, d)); }
// hostDeviceInline auto zCoord(float3 p, float3 m, float3 d) { return zCoord(iCoord(p, m, d)); }
hostDeviceInline auto hCoord(uint3 idx) { return zCoord(idx) % arrays.hash_entries; }
hostDeviceInline auto hCoord(int3 idx) { return zCoord(idx) % arrays.hash_entries; }
hostDeviceInline auto hCoord(float4 p, float3 m, float3 d) { return zCoord(iCoord(p)) % arrays.hash_entries; }
hostDeviceInline auto hCoord(float3 p, float3 m, float3 d) { return zCoord(iCoord(p)) % arrays.hash_entries; }
// hostDeviceInline auto hCoord(float4 p, float3 m, float3 d) { return zCoord(iCoord(p, m, d)) % arrays.hash_entries; }
// hostDeviceInline auto hCoord(float3 p, float3 m, float3 d) { return zCoord(iCoord(p, m, d)) % arrays.hash_entries; }

namespace common {
__device__ auto generateCameraRay(int32_t x, int32_t y, curandState &randState, int32_t i) {
  float3 rendercampos = float3{cScene.m_camera.position.x, cScene.m_camera.position.y, cScene.m_camera.position.z};

  int32_t pixelx = x;
  int32_t pixely = cScene.height - y - 1;

  // float3 finalcol = float3{ 0.0f, 0.0f, 0.0f };
  float3 rendercamview =
      math::normalize(float3{cScene.m_camera.view.x, cScene.m_camera.view.y, cScene.m_camera.view.z});
  float3 rendercamup = math::normalize(float3{cScene.m_camera.up.x, cScene.m_camera.up.y, cScene.m_camera.up.z});
  float3 horizontalAxis = math::normalize(math::cross(rendercamview, rendercamup));
  float3 verticalAxis = math::normalize(math::cross(horizontalAxis, rendercamview));

  float3 middle = rendercampos + rendercamview;
  float3 horizontal = horizontalAxis * tanf(cScene.m_camera.fov.x * 0.5f * (CUDART_PI_F / 180));
  float3 vertical = -verticalAxis * tanf(-cScene.m_camera.fov.y * 0.5f * (CUDART_PI_F / 180));

  float jitterValueX = curand_uniform(&randState) - 0.5f;
  float jitterValueY = curand_uniform(&randState) - 0.5f;
  float sx = (jitterValueX + pixelx) / (cScene.width - 1);
  float sy = (jitterValueY + pixely) / (cScene.height - 1);

  // compute pixel on screen
  float3 pointOnPlaneOneUnitAwayFromEye = middle + (horizontal * ((2 * sx) - 1)) + (vertical * ((2 * sy) - 1));
  float3 pointOnImagePlane =
      rendercampos + ((pointOnPlaneOneUnitAwayFromEye - rendercampos) * cScene.m_camera.focalDistance);

  float3 aperturePoint;
  if (cScene.m_camera.apertureRadius > 0.00001f) {
    float random1 = curand_uniform(&randState);
    float random2 = curand_uniform(&randState);
    float angle = 2.f * CUDART_PI_F * random1;
    float distance = cScene.m_camera.apertureRadius * sqrtf(random2);
    float apertureX = cos(angle) * distance;
    float apertureY = sin(angle) * distance;

    aperturePoint = rendercampos + (horizontalAxis * apertureX) + (verticalAxis * apertureY);
  } else {
    aperturePoint = rendercampos;
  }
  float3 apertureToImagePlane = pointOnImagePlane - aperturePoint;
  apertureToImagePlane = math::normalize(apertureToImagePlane);
  float3 rayInWorldSpace = math::normalize(apertureToImagePlane);
  float3 originInWorldSpace = aperturePoint;

  return worldRay{originInWorldSpace, 1e21f, rayInWorldSpace, 1.f, 0u, 0u, (uint32_t)i};
}
__global__ void generateBlockedRays(int32_t seed, Pixel *image, worldRay *rays, worldRay *oldRays,
                                    int32_t msaa_factor) {
  int32_t x = blockIdx.x * blockDim.y + threadIdx.y;
  int32_t y = blockIdx.y * blockDim.z + threadIdx.z;
  if (x >= cScene.width)
    return;
  if (y >= cScene.height)
    return;
  int32_t i = (cScene.height - y - 1) * cScene.width + x;

  int32_t blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int32_t threadId = blockId * (blockDim.x * blockDim.y * blockDim.z) + (threadIdx.z * (blockDim.x * blockDim.y)) +
                     (threadIdx.y * blockDim.x) + threadIdx.x;

  curandState randState;
  curand_init(seed + threadId, 0, 0, &randState);

  image[i] = Pixel{float3{0.f, 0.f, 0.f}, float3{1.f, 1.f, 1.f}};
  auto worldRay = generateCameraRay(x, y, randState, i);
  rays[i * msaa_factor + threadIdx.x] = worldRay;
  cfluidIntersection[i] = float4{0.f, 0.f, 0.f, FLT_MAX};
  // cInternalFlag[i] = 0;
}
//#define DEBUG_NORMALS

__global__ void toneMap(int32_t frameNumber, float3 *accumBuffer, Pixel *image, float rate) {
  int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  int32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= cScene.width)
    return;
  if (y >= cScene.height)
    return;
  int32_t i = (cScene.height - y - 1) * cScene.width + x;

  accumBuffer[i] += image[i].color / rate;
  float3 tempcol = (accumBuffer[i] / frameNumber);
  // tempcol = tempcol / 2.f + 0.5f;
  float3 colour = float3{math::clamp(tempcol.x, 0.0f, 1.0f), math::clamp(tempcol.y, 0.0f, 1.0f),
                         math::clamp(tempcol.z, 0.0f, 1.0f)};
#ifdef DEBUG_NORMALS
  float4 out{colour.x, colour.y, colour.z, 1.f};
#else
  float4 out{colour.x, colour.y, colour.z, 1.f};
  // float4 out{(powf(colour.x, 1 / 2.2f)), (powf(colour.y, 1 / 2.2f)), (powf(colour.z, 1 / 2.2f)), 1.f};
#endif
// out = float4{ colour.x, colour.y, colour.z, 1.f };
#if defined(__CUDA_ARCH__)
  surf2Dwrite(out, surfaceWriteOut, x * sizeof(float4), y, cudaBoundaryModeClamp);
#endif
}
__global__ void toneMapNormals(int32_t frameNumber, float3 *accumBuffer, Pixel *image, float rate) {
  int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  int32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= cScene.width)
    return;
  if (y >= cScene.height)
    return;
  int32_t i = (cScene.height - y - 1) * cScene.width + x;

  accumBuffer[i] += image[i].color / rate;
  float3 tempcol = (accumBuffer[i] / frameNumber);
  tempcol = tempcol / 2.f + 0.5f;
  float3 colour = float3{math::clamp(tempcol.x, 0.0f, 1.0f), math::clamp(tempcol.y, 0.0f, 1.0f),
                         math::clamp(tempcol.z, 0.0f, 1.0f)};
  float4 out{colour.x, colour.y, colour.z, 1.f};
  //float4 out = float4{ (float)x / (float)cScene.width, (float)y / (float)cScene.height,0.f,1.f };
#if defined(__CUDA_ARCH__)
  surf2Dwrite(out, surfaceWriteOut, x * sizeof(float4), y, cudaBoundaryModeClamp);
#endif
}
} // namespace common
namespace aabb {
__device__ auto rayIntersectAABB(Ray r, float3 b_min, float3 b_max) {
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define MAX(x, y) ((x) > (y) ? (x) : (y))
  float t1 = (b_min.x - r.orig.x) / r.dir.x;
  float t2 = (b_max.x - r.orig.x) / r.dir.x;

  float tmin = MIN(t1, t2);
  float tmax = MAX(t1, t2);

  t1 = (b_min.y - r.orig.y) / r.dir.y;
  t2 = (b_max.y - r.orig.y) / r.dir.y;

  tmin = MAX(tmin, MIN(t1, t2));
  tmax = MIN(tmax, MAX(t1, t2));

  t1 = (b_min.z - r.orig.z) / r.dir.z;
  t2 = (b_max.z - r.orig.z) / r.dir.z;

  tmin = MAX(tmin, MIN(t1, t2));
  tmax = MIN(tmax, MAX(t1, t2));

  return AABBHit{tmax > MAX(tmin, 0.f), tmin, tmax};
}
} // namespace aabb
namespace traversal {

__device__ auto lookup_cell(const int3 &idx) {
  // if (idx.x >= fluidMemory.grid_size.x || idx.y >= fluidMemory.grid_size.y || idx.z >= fluidMemory.grid_size.z)
  //  return INT_MAX;
  if (idx.x < 0 || idx.y < 0 || idx.z < 0)
    return INT_MAX;
  auto morton = zCoord(idx);
  auto s = fluidMemory.hashMap[hCoord(idx)];
  if (s.hash.kind == LIST_ALWAYS_FALSE)
    return INT_MAX;
  if (s.hash.kind == LIST_ALWAYS_TRUE)
    return 1;
  if (s.hash.kind == LIST_COMPACT)
    if (morton == s.cell.hash)
      return 1;
  if (s.hash.kind == LIST_ITERATE)
    for (int32_t ii = s.hash.beginning; ii < s.hash.beginning + s.hash.length; ++ii)
      if (fluidMemory.cellSpan[ii].cell.hash == morton)
        return 1;
  return INT_MAX;
}
__device__ __host__ float mod(float a, float N) { return a - N * floorf(a / N); }
__device__ __host__ float intBound2_s(float s, float ds) {
  if (s == floorf(s) && ds < 0.f)
    return 0.f;
  if (ds < 0.f)
    return intBound2_s(-s, -ds);
  float s2 = mod(s, 1.f);
  return (1.f - s2) / ds;
}
__device__ float3 intBound(const float3 &s, const float3 &ds) {
  // return (ds > 0 ? Math.ceil(s) - s : s - Math.floor(s)) / Math.abs(ds);
  // return float3{
  //	(ds.x > 0.f ? ceilf(s.x) - s.x : s.x - floorf(s.x)) / fabsf(ds.x),
  //	(ds.y > 0.f ? ceilf(s.y) - s.y : s.y - floorf(s.y)) / fabsf(ds.y),
  //	(ds.z > 0.f ? ceilf(s.z) - s.z : s.z - floorf(s.z)) / fabsf(ds.z)
  //};
  return float3{intBound2_s(s.x, ds.x), intBound2_s(s.y, ds.y), intBound2_s(s.z, ds.z)};
}
__device__ float3 intBoundRay(const Ray &r) {
  return intBound((r.orig - fluidMemory.min_coord) / fluidMemory.cell_size, r.dir);
}
__device__ float3 intBoundRay(const Ray &r, float t) {
  return intBound((r(t) - fluidMemory.min_coord) / fluidMemory.cell_size, r.dir);
}
} // namespace traversal
#pragma endregion
namespace render {
namespace util {
__device__ int lanemask_lt(int lane) { return (1 << (lane)) - 1; }
// increment the value at ptr by 1 and return the old value
__device__ int atomicAggInc(int *p) {
  unsigned int writemask = __activemask();
  unsigned int total = __popc(writemask);
  unsigned int prefix = __popc(writemask & lanemask_lt(threadIdx.x & (warpSize - 1)));
  // Find the lowest-numbered active lane
  int elected_lane = __ffs(writemask) - 1;
  int base_offset = 0;
  if (prefix == 0) {
    base_offset = atomicAdd(p, total);
  }
  base_offset = __shfl_sync(writemask, base_offset, elected_lane);
  int thread_offset = prefix + base_offset;
  return thread_offset;
  // int mask = __match_any_sync(__activemask(), (unsigned long long)ptr);
  // int leader = __ffs(mask) - 1; // select a leader
  // int res;
  // unsigned lane_id = threadIdx.x % warpSize;
  // if (lane_id == leader) // leader does the update
  // res = atomicAdd(ptr, __popc(mask));
  // res = __shfl_sync(mask, res, leader);             // get leader’s old value
  // return res + __popc(mask & ((1 << lane_id) - 1)); // compute old value
}
struct compactIterator {
  int3 central_idx;
  bool full_loop;

  hostDevice compactIterator(int3 position) : central_idx(position) {}
  struct cell_iterator {
    int3 idx;
    int32_t i = -1, j = -1, k = -1;
    uint32_t ii = 0;
    uint32_t jj = 0;

    uint32_t neighbor_idx;

    compactHashSpan s{0, INVALID_BEG, 0};
    uint32_t cs_beg = 0;
    uint32_t cs_len = 0;

    hostDevice int32_t cs_loop() {
      if (cs_beg != INVALID_BEG && jj < (cs_beg + cs_len)) {
        neighbor_idx = jj;
        ++jj;
        return neighbor_idx;
      }
      return -1;
    }

    hostDevice int32_t s_loop() {
      if (s.beginning != INVALID_BEG) {
        uint3 cell =
            uint3{static_cast<uint32_t>(idx.x + i), static_cast<uint32_t>(idx.y + j), static_cast<uint32_t>(idx.z + k)};
        if (cell.x == UINT32_MAX || cell.y == UINT32_MAX || cell.z == UINT32_MAX)
          return -1;
        auto morton = zCoord(cell);
        if (s.compacted && ii < s.beginning + s.length) {
          cs_beg = s.beginning;
          cs_len = s.length;
          // cs = compact_cellSpan{0, s.beginning, s.length};
          jj = cs_beg;
          ii = s.beginning + s.length;
          if (zCoord(arrays.position[cs_beg]) == morton) {
            if (cs_loop() != -1) {
              return neighbor_idx;
            }
          }
        } else
          for (; ii < s.beginning + s.length;) {
            auto cst = arrays.compactCellSpan[ii];
            cs_beg = cst.beginning;
#ifdef BITFIELD_STRUCTURES
            cs_len = cst.length == INVALID_LEN ? arrays.auxLength[ii] : cst.length;
#else
            cs_len = cst.length;
#endif
            // cs = arrays.compactCellSpan[ii];
            ++ii;
            jj = cs_beg;
            if (zCoord(arrays.position[cs_beg]) == morton) {
              if (cs_loop() != -1) {
                return neighbor_idx;
              }
            }
          }
        ++k;
      }
      return -1;
    }

    hostDevice void increment() {
      if (cs_loop() != -1)
        return;
      if (s_loop() != -1)
        return;

      for (; i <= 1; ++i) {
        for (; j <= 1; ++j) {
          for (; k <= 1;) {
            uint3 cell = uint3{static_cast<uint32_t>(idx.x + i), static_cast<uint32_t>(idx.y + j),
                               static_cast<uint32_t>(idx.z + k)};
            if (cell.x == UINT32_MAX || cell.y == UINT32_MAX || cell.z == UINT32_MAX) {
              ++k;
              continue;
            }
            auto morton = zCoord(cell);

            s = arrays.compactHashMap[hCoord(cell)];
            ii = s.beginning;
            if (s.beginning == INVALID_BEG) {
              ++k;
              continue;
            }
            if (s_loop() != -1)
              return;
          }
          k = -1;
        }
        j = -1;
      }
    }

    hostDevice cell_iterator(int3 c_idx, int32_t _i = -1, int32_t _j = -1, int32_t _k = -1)
        : idx(c_idx), i(_i), j(_j), k(_k) {
      int32_t thread_idx = getThreadIdx();
      increment();
    }

    hostDeviceInline int32_t operator*() { return neighbor_idx; };
    hostDeviceInline bool operator==(const cell_iterator &rawIterator) const { return (i == rawIterator.i); }
    hostDeviceInline bool operator!=(const cell_iterator &rawIterator) const { return (i != rawIterator.i); }

    hostDeviceInline cell_iterator &operator++() {
      increment();
      return (*this);
    }
    hostDeviceInline cell_iterator operator++(int) {
      auto temp(*this);
      increment();
      return temp;
    }
  };

  hostDeviceInline cell_iterator begin() const { return cell_iterator(central_idx); }
  hostDeviceInline cell_iterator end() const { return cell_iterator(central_idx, 2, 2, 2); }
  hostDeviceInline cell_iterator cbegin() const { return cell_iterator(central_idx); }
  hostDeviceInline cell_iterator cend() const { return cell_iterator(central_idx, 2, 2, 2); }
};
template <typename T> __device__ __host__ __inline__ auto square(T &&x) { return x * x; }
template <typename T> __device__ __host__ __inline__ auto cube(T &&x) { return x * x * x; }
__device__ __host__ __inline__ float k(float4 x_i, float4 x_j) {
  auto h = (x_i.w + x_j.w) * 0.5f * kernelSize();
  auto d = math::distance3(x_i, x_j);
  auto s = d / h;
  return math::max(0.f, cube(1.f - square(s)));
}
__device__ __inline__ float turkAnisotropic(float4 position, int32_t j) {
  float4 xBar_j = arrays.centerPosition[j];
  // float det = xBar_j.w;
  auto hjoj = xBar_j.w;
  float3 diff = math::castTo<float3>(position - xBar_j);
  auto dtr = math::length3(diff);
  if (dtr > hjoj)
    return 0.f;
  float a1 = G(j, 0, 0) * diff.x + G(j, 0, 1) * diff.y + G(j, 0, 2) * diff.z;
  float a2 = G(j, 0, 1) * diff.x + G(j, 1, 1) * diff.y + G(j, 1, 2) * diff.z;
  float a3 = G(j, 0, 2) * diff.x + G(j, 1, 2) * diff.y + G(j, 2, 2) * diff.z;
  float det = G(j, 1, 0);
  float q = sqrtf(a1 * a1 + a2 * a2 + a3 * a3);
  float sigma = 315.f / (64.f * CUDART_PI_F);
  float W = 0.f;
  auto q2 = 1.f - q * q;
  if (q <= 1.f) {
    W = q2 * q2 * q2;
  }
  return sigma * det * W;
}
__device__ __inline__ float3 turkAnisotropicGradient(float4 position, int32_t j) {
  float4 xBar_j = arrays.centerPosition[j];
  // float det = xBar_j.w;
  auto hjoj = xBar_j.w;
  float3 diff = math::castTo<float3>(position - xBar_j);
  auto dtr = math::length3(diff);
  if (dtr > hjoj)
    return float3{0.f, 0.f, 0.f};

  float a1 = G(j, 0, 0) * diff.x + G(j, 0, 1) * diff.y + G(j, 0, 2) * diff.z;
  float a2 = G(j, 0, 1) * diff.x + G(j, 1, 1) * diff.y + G(j, 1, 2) * diff.z;
  float a3 = G(j, 0, 2) * diff.x + G(j, 1, 2) * diff.y + G(j, 2, 2) * diff.z;
  float det = G(j, 1, 0);
  float3 dir{a1, a2, a3};
  float q = math::length3(dir);
  float sigma = 315.f / (64.f * CUDART_PI_F);
  float W = 0.f;
  auto q2 = 1.f - q * q;
  if (q <= 1.f) {
    W = -6.f * q * q2 * q2 * q2;
  }
  return q >= 1e-12f && q <= 1.f ? sigma * dir / q * det * W : float3{0.f, 0.f, 0.f};
}
__device__ __inline__ float fresnel(const float3 &I, const float3 &N, float ior) {
  float kr;
  float cosi = math::clamp(-1.f, 1.f, math::dot(I, N));
  float etai = 1.f, etat = ior;
  if (cosi > 0.f) {
    auto t = etat;
    etat = etai;
    etai = t;
  }
  // Compute sini using Snell's law
  float sint = etai / etat * sqrtf(math::max(0.f, 1.f - cosi * cosi));
  // Total internal reflection
  if (sint >= 1.f) {
    kr = 1;
  } else {
    float cost = sqrtf(math::max(0.f, 1.f - sint * sint));
    cosi = fabsf(cosi);
    float Rs = ((etat * cosi) - (etai * cost)) / ((etat * cosi) + (etai * cost));
    float Rp = ((etai * cosi) - (etat * cost)) / ((etai * cosi) + (etat * cost));
    kr = (Rs * Rs + Rp * Rp) / 2.f;
  }
  // As a consequence of the conservation of energy, transmittance is given by:
  // kt = 1 - kr;
  return kr;
}
__device__ float3 refract(const float3 &I, const float3 &N, const float &ior) {
  float cosi = math::clamp(-1.f, 1.f, math::dot(I, N));
  float etai = 1.f, etat = ior;
  float3 n = N;
  if (cosi < 0.f) {
    cosi = -cosi;
  } else {
    etai = ior;
    etat = 1.f;
    n = -N;
  }
  float eta = etai / etat;
  float k = 1.f - eta * eta * (1.f - cosi * cosi);
  return k < 0.f ? float3{0.f, 0.f, 0.f} : eta * I + (eta * cosi - sqrtf(k)) * n;
}
__device__ float3 reflect(const float3 &I, const float3 &N) { return I - 2.f * math::dot(I, N) * N; }
} // namespace util
__global__ void intersectAABB(int32_t numRays) {
  int32_t i = static_cast<int32_t>(threadIdx.x + blockIdx.x * blockDim.x);
  if (i >= numRays)
    return;

  auto idRay = cRaysDepth[i];
  auto aabb = aabb::rayIntersectAABB(idRay, fluidMemory.min_coord, fluidMemory.max_coord);
  if (aabb.hit && idRay.bounces < fluidMemory.bounces) {
    cResortKey[i] = idRay.index;
    cRaysDepth[i].depth = aabb.tmin;
  } else if (!aabb.hit && idRay.bounces < fluidMemory.bounces) {
    cResortKey[i] = cNumRays + i;
    cRaysDepth[i].depth = FLT_MAX;
  } else {
    cResortKey[i] = 2 * cNumRays + i;
    cRaysDepth[i].depth = FLT_MAX;
  }
  cResortIndex[i] = i;
  cfluidIntersection[i] = float4{1.f, 0., 0.f, FLT_MAX};
  cfluidDepth[i] = FLT_MAX;
}
__global__ void sort(int32_t numRays) {
  int32_t i = static_cast<int32_t>(threadIdx.x + blockIdx.x * blockDim.x);
  if (i >= numRays)
    return;
  if (i != cResortIndex[i])
    cCompactRays[i] = cRaysDepth[cResortIndex[i]];
  cfluidDepth[i] = FLT_MAX;
  // if (blockIdx.x == 0 && blockIdx.y == 0) {
  //	printf("[%d %d] -> %d -> %d\n",x ,y, i, cResortIndex[i]);
  //}
}

__device__ __host__ auto degToRad(float deg) { return deg / 360.f * 2.f * CUDART_PI_F; }

struct AABB {
  float3 min, max;
};

deviceInline void CoordinateSystem(const float3 &v1, float3 *v2, float3 *v3) {
  if (fabsf(v1.x) > fabsf(v1.y))
    *v2 = float3{-v1.z, 0, v1.x} / sqrtf(v1.x * v1.x + v1.z * v1.z);
  else
    *v2 = float3{0, v1.z, -v1.y} / sqrtf(v1.y * v1.y + v1.z * v1.z);
  *v3 = math::cross(v1, *v2);
}
deviceInline float3 SphericalDirection(float sinTheta, float cosTheta, float phi) {
  return float3{sinTheta * cosf(phi), sinTheta * sinf(phi), cosTheta};
}

deviceInline float3 SphericalDirection(float sinTheta, float cosTheta, float phi, const float3 &x, const float3 &y,
                                       const float3 &z) {
  return sinTheta * cosf(phi) * x + sinTheta * sinf(phi) * y + cosTheta * z;
}

deviceInline auto pToWorld(const Matrix4x4 &mat, float3 x) {
  return math::castTo<float3>(mat * float4{x.x, x.y, x.z, 1.f});
}
deviceInline auto vToWorld(const Matrix4x4 &mat, float3 x) {
  return math::castTo<float3>(mat * float4{x.x, x.y, x.z, 0.f});
}
deviceInline auto nToWorld(const Matrix4x4 &mat, float3 x) {
  return math::castTo<float3>(mat * float4{x.x, x.y, x.z, 0.f});
}
deviceInline auto UniformConePdf(float cosThetaMax) { return 1 / (2 * CUDART_PI_F * (1 - cosThetaMax)); }

struct Interaction {
  float3 p, n;
};

#define f3_0                                                                                                           \
  float3 { 0.f, 0.f, 0.f }
#define f4_0                                                                                                           \
  float4 { 0.f, 0.f, 0.f, 0.f }
struct hitInformation {
  bool hit = false;
  float3 k_d = f3_0;
  float3 k_s = f3_0;
  float rough = 1.f;
  float depth = 1e21f;
  float3 normal = f3_0;
  Refl_t material = Refl_t::Lambertian;
  float3 emission = f3_0;
  float3 rayDirection;
};

namespace pbrt {
// Global Inline Functions
deviceInline uint32_t FloatToBits(float f) { return __float_as_uint(f); }
deviceInline float BitsToFloat(uint32_t ui) { return __uint_as_float(ui); }
deviceInline float NextFloatDown(float v) {
  // Handle infinity and positive zero for _NextFloatDown()_
  // if (std::isinf(v) && v < 0.)
  //  return v;
  if (v == 0.f)
    v = -0.f;
  uint32_t ui = FloatToBits(v);
  if (v > 0)
    --ui;
  else
    ++ui;
  return BitsToFloat(ui);
}
deviceInline float NextFloatUp(float v) {
  // Handle infinity and negative zero for _NextFloatUp()_
  // if (std::isinf(v) && v > 0.)
  //  return v;
  if (v == -0.f)
    v = 0.f;

  // Advance _v_ to next higher float
  uint32_t ui = FloatToBits(v);
  if (v >= 0)
    ++ui;
  else
    --ui;
  return BitsToFloat(ui);
}
#define MaxFloat std::numeric_limits<float>::max()
#define Infinity std::numeric_limits<float>::infinity()
#define MachineEpsilon (std::numeric_limits<float>::epsilon() * 0.5)
// EFloat Declarations
class EFloat {
public:
  // EFloat Public Methods
  deviceInline EFloat() {}
  deviceInline EFloat(float v, float err = 0.f) : v(v) {
    if (err == 0.)
      low = high = v;
    else {
      // Compute conservative bounds by rounding the endpoints away
      // from the middle. Note that this will be over-conservative in
      // cases where v-err or v+err are exactly representable in
      // floating-point, but it's probably not worth the trouble of
      // checking this case.
      low = NextFloatDown(v - err);
      high = NextFloatUp(v + err);
    }
// Store high precision reference value in _EFloat_
#ifndef NDEBUG
    vPrecise = v;
    Check();
#endif // NDEBUG
  }
#ifndef NDEBUG
  EFloat(float v, long double lD, float err) : EFloat(v, err) {
    vPrecise = lD;
    Check();
  }
#endif // DEBUG
  deviceInline EFloat operator+(EFloat ef) const {
    EFloat r;
    r.v = v + ef.v;
#ifndef NDEBUG
    r.vPrecise = vPrecise + ef.vPrecise;
#endif // DEBUG
       // Interval arithemetic addition, with the result rounded away from
       // the value r.v in order to be conservative.
    r.low = NextFloatDown(LowerBound() + ef.LowerBound());
    r.high = NextFloatUp(UpperBound() + ef.UpperBound());
    return r;
  }
  deviceInline explicit operator float() const { return v; }
  deviceInline explicit operator double() const { return v; }
  deviceInline float GetAbsoluteError() const { return NextFloatUp(std::max(fabsf(high - v), fabsf(v - low))); }
  deviceInline float UpperBound() const { return high; }
  deviceInline float LowerBound() const { return low; }
#ifndef NDEBUG
  float GetRelativeError() const { return fabsf((vPrecise - v) / vPrecise); }
  long double PreciseValue() const { return vPrecise; }
#endif
  deviceInline EFloat operator-(EFloat ef) const {
    EFloat r;
    r.v = v - ef.v;
#ifndef NDEBUG
    r.vPrecise = vPrecise - ef.vPrecise;
#endif
    r.low = NextFloatDown(LowerBound() - ef.UpperBound());
    r.high = NextFloatUp(UpperBound() - ef.LowerBound());
    return r;
  }
  deviceInline EFloat operator*(EFloat ef) const {
    EFloat r;
    r.v = v * ef.v;
#ifndef NDEBUG
    r.vPrecise = vPrecise * ef.vPrecise;
#endif
    float prod[4] = {LowerBound() * ef.LowerBound(), UpperBound() * ef.LowerBound(), LowerBound() * ef.UpperBound(),
                     UpperBound() * ef.UpperBound()};
    r.low = NextFloatDown(std::min(std::min(prod[0], prod[1]), std::min(prod[2], prod[3])));
    r.high = NextFloatUp(std::max(std::max(prod[0], prod[1]), std::max(prod[2], prod[3])));
    return r;
  }
  deviceInline EFloat operator/(EFloat ef) const {
    EFloat r;
    r.v = v / ef.v;
#ifndef NDEBUG
    r.vPrecise = vPrecise / ef.vPrecise;
#endif
    if (ef.low < 0 && ef.high > 0) {
      // Bah. The interval we're dividing by straddles zero, so just
      // return an interval of everything.
      r.low = -Infinity;
      r.high = Infinity;
    } else {
      float div[4] = {LowerBound() / ef.LowerBound(), UpperBound() / ef.LowerBound(), LowerBound() / ef.UpperBound(),
                      UpperBound() / ef.UpperBound()};
      r.low = NextFloatDown(std::min(std::min(div[0], div[1]), std::min(div[2], div[3])));
      r.high = NextFloatUp(std::max(std::max(div[0], div[1]), std::max(div[2], div[3])));
    }
    return r;
  }
  deviceInline EFloat operator-() const {
    EFloat r;
    r.v = -v;
#ifndef NDEBUG
    r.vPrecise = -vPrecise;
#endif
    r.low = -high;
    r.high = -low;
    return r;
  }
  deviceInline bool operator==(EFloat fe) const { return v == fe.v; }
  deviceInline EFloat(const EFloat &ef) {
    v = ef.v;
    low = ef.low;
    high = ef.high;
#ifndef NDEBUG
    vPrecise = ef.vPrecise;
#endif
  }
  deviceInline EFloat &operator=(const EFloat &ef) {
    if (&ef != this) {
      v = ef.v;
      low = ef.low;
      high = ef.high;
#ifndef NDEBUG
      vPrecise = ef.vPrecise;
#endif
    }
    return *this;
  }

private:
  // EFloat Private Data
  float v, low, high;
#ifndef NDEBUG
  long double vPrecise;
#endif // NDEBUG
  deviceInline friend EFloat sqrt(EFloat fe);
  deviceInline friend EFloat abs(EFloat fe);
  deviceInline friend bool Quadratic(EFloat A, EFloat B, EFloat C, EFloat *t0, EFloat *t1);
};
// EFloat Inline Functions
// deviceInline EFloat operator*(float f, EFloat fe) { return EFloat(f) * fe; }
//
// deviceInline EFloat operator/(float f, EFloat fe) { return EFloat(f) / fe; }
//
// deviceInline EFloat operator+(float f, EFloat fe) { return EFloat(f) + fe; }
//
// deviceInline EFloat operator-(float f, EFloat fe) { return EFloat(f) - fe; }
deviceInline EFloat sqrt(EFloat fe) {
  EFloat r;
  r.v = sqrtf(fe.v);
#ifndef NDEBUG
  r.vPrecise = sqrtf(fe.vPrecise);
#endif
  r.low = NextFloatDown(sqrtf(fe.low));
  r.high = NextFloatUp(sqrtf(fe.high));
  return r;
}
deviceInline EFloat abs(EFloat fe) {
  if (fe.low >= 0)
    // The entire interval is greater than zero, so we're all set.
    return fe;
  else if (fe.high <= 0) {
    // The entire interval is less than zero.
    EFloat r;
    r.v = -fe.v;
#ifndef NDEBUG
    r.vPrecise = -fe.vPrecise;
#endif
    r.low = -fe.high;
    r.high = -fe.low;
    return r;
  } else {
    // The interval straddles zero.
    EFloat r;
    r.v = fabsf(fe.v);
#ifndef NDEBUG
    r.vPrecise = fabsf(fe.vPrecise);
#endif
    r.low = 0;
    r.high = std::max(-fe.low, fe.high);
    return r;
  }
}
deviceInline bool Quadratic(EFloat A, EFloat B, EFloat C, EFloat *t0, EFloat *t1);
deviceInline bool Quadratic(EFloat A, EFloat B, EFloat C, EFloat *t0, EFloat *t1) {
  // Find quadratic discriminant
  double discrim = (double)B.v * (double)B.v - 4. * (double)A.v * (double)C.v;
  if (discrim < 0.)
    return false;
  double rootDiscrim = sqrtf(discrim);

  EFloat floatRootDiscrim(rootDiscrim, MachineEpsilon * rootDiscrim);

  // Compute quadratic _t_ values
  EFloat q;
  if ((float)B < 0)
    q = -EFloat(.5f) * (B - floatRootDiscrim);
  else
    q = -EFloat(.5f) * (B + floatRootDiscrim);
  *t0 = q / A;
  *t1 = C / q;
  if ((float)*t0 > (float)*t1) {
    auto tt = *t0;
    *t0 = *t1;
    *t1 = tt;
    // std::swap(*t0, *t1);
  }
  return true;
}
deviceInline float gamma(int n) { return (n * MachineEpsilon) / (1 - n * MachineEpsilon); }
deviceInline float3 OffsetRayOrigin(const float3 &p, const float3 &n, const float3 &w) {
  float d = 1e-1f;
  float3 offset = d * n;
  if (math::dot(w, n) < 0)
    offset = -offset;
  float3 po = p + offset;
  if (offset.x > 0)
    po.x = NextFloatUp(po.x);
  else if (offset.x < 0)
    po.x = NextFloatDown(po.x);

  if (offset.y > 0)
    po.y = NextFloatUp(po.y);
  else if (offset.y < 0)
    po.y = NextFloatDown(po.y);

  if (offset.z > 0)
    po.z = NextFloatUp(po.z);
  else if (offset.z < 0)
    po.z = NextFloatDown(po.z);

  return po;
}
// Reflection Declarations
deviceInline float FrDielectric(float cosThetaI, float etaI, float etaT);
// BSDF Inline Functions
deviceInline float CosTheta(const float3 &w) { return w.z; }
deviceInline float Cos2Theta(const float3 &w) { return w.z * w.z; }
deviceInline float AbsCosTheta(const float3 &w) { return fabsf(w.z); }
deviceInline float Sin2Theta(const float3 &w) { return std::max((float)0, (float)1.f - Cos2Theta(w)); }
deviceInline float SinTheta(const float3 &w) { return sqrtf(Sin2Theta(w)); }
deviceInline float TanTheta(const float3 &w) { return SinTheta(w) / CosTheta(w); }
deviceInline float Tan2Theta(const float3 &w) { return Sin2Theta(w) / Cos2Theta(w); }
deviceInline float CosPhi(const float3 &w) {
  float sinTheta = SinTheta(w);
  return (sinTheta == 0) ? 1 : math::clamp(w.x / sinTheta, -1, 1);
}
deviceInline float SinPhi(const float3 &w) {
  float sinTheta = SinTheta(w);
  return (sinTheta == 0) ? 0 : math::clamp(w.y / sinTheta, -1, 1);
}
deviceInline float Cos2Phi(const float3 &w) { return CosPhi(w) * CosPhi(w); }
deviceInline float Sin2Phi(const float3 &w) { return SinPhi(w) * SinPhi(w); }
deviceInline float CosDPhi(const float3 &wa, const float3 &wb) {
  return math::clamp((wa.x * wb.x + wa.y * wb.y) / sqrtf((wa.x * wa.x + wa.y * wa.y) * (wb.x * wb.x + wb.y * wb.y)), -1,
                     1);
}
deviceInline float3 Reflect(const float3 &wo, const float3 &n) { return -wo + 2.f * math::dot3(wo, n) * n; }
deviceInline bool Refract(const float3 &wi, const float3 &n, float eta, float3 *wt) {
  // Compute $\cos \theta_\roman{t}$ using Snell's law
  float cosThetaI = math::dot3(n, wi);
  float sin2ThetaI = std::max(float(0), float(1.f - cosThetaI * cosThetaI));
  float sin2ThetaT = eta * eta * sin2ThetaI;

  // Handle total internal reflection for transmission
  if (sin2ThetaT >= 1.f)
    return false;
  float cosThetaT = sqrtf(1 - sin2ThetaT);
  *wt = eta * -wi + (eta * cosThetaI - cosThetaT) * float3(n);
  return true;
}
deviceInline bool SameHemisphere(const float3 &w, const float3 &wp) { return w.z * wp.z > 0.f; }
#undef G
// MicrofacetDistribution Declarations
class MicrofacetDistribution {
public:
  // MicrofacetDistribution Public Methods
  __device__ virtual ~MicrofacetDistribution();
  __device__ virtual float D(const float3 &wh) const = 0;
  __device__ virtual float Lambda(const float3 &w) const = 0;
  __device__ float G1(const float3 &w) const {
    //    if (Dot(w, wh) * CosTheta(w) < 0.) return 0.;
    return 1.f / (1.f + Lambda(w));
  }
  __device__ virtual float G(const float3 &wo, const float3 &wi) const { return 1.f / (1.f + Lambda(wo) + Lambda(wi)); }
  __device__ virtual float3 Sample_wh(const float3 &wo, const float2 &u) const = 0;
  __device__ float Pdf(const float3 &wo, const float3 &wh) const;

protected:
  // MicrofacetDistribution Protected Methods
  __device__ MicrofacetDistribution(bool sampleVisibleArea) : sampleVisibleArea(sampleVisibleArea) {}

  // MicrofacetDistribution Protected Data
  const bool sampleVisibleArea;
};
class BeckmannDistribution : public MicrofacetDistribution {
public:
  // BeckmannDistribution Public Methods
  __device__ static float RoughnessToAlpha(float roughness) {
    roughness = std::max(roughness, (float)1e-3);
    float x = logf(roughness);
    return 1.62142f + 0.819955f * x + 0.1734f * x * x + 0.0171201f * x * x * x + 0.000640711f * x * x * x * x;
  }
  __device__ BeckmannDistribution(float alphax, float alphay, bool samplevis = true)
      : MicrofacetDistribution(samplevis), alphax(alphax), alphay(alphay) {}
  __device__ float D(const float3 &wh) const;
  __device__ float3 Sample_wh(const float3 &wo, const float2 &u) const;

private:
  // BeckmannDistribution Private Methods
  __device__ float Lambda(const float3 &w) const;

  // BeckmannDistribution Private Data
  const float alphax, alphay;
};
class TrowbridgeReitzDistribution : public MicrofacetDistribution {
public:
  // TrowbridgeReitzDistribution Public Methods
  __device__ static inline float RoughnessToAlpha(float roughness);
  __device__ TrowbridgeReitzDistribution(float alphax, float alphay, bool samplevis = true)
      : MicrofacetDistribution(samplevis), alphax(alphax), alphay(alphay) {}
  __device__ float D(const float3 &wh) const;
  __device__ float3 Sample_wh(const float3 &wo, const float2 &u) const;

private:
  // TrowbridgeReitzDistribution Private Methods
  __device__ float Lambda(const float3 &w) const;

  // TrowbridgeReitzDistribution Private Data
  const float alphax, alphay;
};
// MicrofacetDistribution Inline Methods
__device__ inline float TrowbridgeReitzDistribution::RoughnessToAlpha(float roughness) {
  roughness = std::max(roughness, (float)1e-3);
  float x = logf(roughness);
  return 1.62142f + 0.819955f * x + 0.1734f * x * x + 0.0171201f * x * x * x + 0.000640711f * x * x * x * x;
}
__device__ inline float ErfInv(float x) {
  float w, p;
  x = math::clamp(x, -.99999f, .99999f);
  w = -logf((1.f - x) * (1.f + x));
  if (w < 5.f) {
    w = w - 2.5f;
    p = 2.81022636e-08f;
    p = 3.43273939e-07f + p * w;
    p = -3.5233877e-06f + p * w;
    p = -4.39150654e-06f + p * w;
    p = 0.00021858087f + p * w;
    p = -0.00125372503f + p * w;
    p = -0.00417768164f + p * w;
    p = 0.246640727f + p * w;
    p = 1.50140941f + p * w;
  } else {
    w = sqrtf(w) - 3.f;
    p = -0.000200214257f;
    p = 0.000100950558f + p * w;
    p = 0.00134934322f + p * w;
    p = -0.00367342844f + p * w;
    p = 0.00573950773f + p * w;
    p = -0.0076224613f + p * w;
    p = 0.00943887047f + p * w;
    p = 1.00167406f + p * w;
    p = 2.83297682f + p * w;
  }
  return p * x;
}
__device__ inline float Erf(float x) {
  // constants
  float a1 = 0.254829592f;
  float a2 = -0.284496736f;
  float a3 = 1.421413741f;
  float a4 = -1.453152027f;
  float a5 = 1.061405429f;
  float p = 0.3275911f;

  // Save the sign of x
  int sign = 1;
  if (x < 0.f)
    sign = -1;
  x = fabsf(x);

  // A&S formula 7.1.26
  float t = 1.f / (1.f + p * x);
  float y = 1.f - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * expf(-x * x);

  return sign * y;
}
// Microfacet Utility Functions
__device__ void BeckmannSample11(float cosThetaI, float U1, float U2, float *slope_x, float *slope_y) {
  /* Special case (normal incidence) */
  if (cosThetaI > .9999f) {
    float r = sqrtf(-logf(1.0f - U1));
    float sinPhi = sinf(2.f * CUDART_PI_F * U2);
    float cosPhi = cosf(2.f * CUDART_PI_F * U2);
    *slope_x = r * cosPhi;
    *slope_y = r * sinPhi;
    return;
  }

  /* The original inversion routine from the paper contained
     discontinuities, which causes issues for QMC integration
     and techniques like Kelemen-style MLT. The following code
     performs a numerical inversion with better behavior */
  float sinThetaI = sqrtf(std::max((float)0, (float)1 - cosThetaI * cosThetaI));
  float tanThetaI = sinThetaI / cosThetaI;
  float cotThetaI = 1.f / tanThetaI;

  /* Search interval -- everything is parameterized
     in the Erf() domain */
  float a = -1.f, c = Erf(cotThetaI);
  float sample_x = std::max(U1, (float)1e-6f);

  /* Start with a good initial guess */
  // float b = (1-sample_x) * a + sample_x * c;

  /* We can do better (inverse of an approximation computed in
   * Mathematica) */
  float thetaI = acosf(cosThetaI);
  float fit = 1.f + thetaI * (-0.876f + thetaI * (0.4265f - 0.0594f * thetaI));
  float b = c - (1.f + c) * std::pow(1.f - sample_x, fit);

  /* Normalization factor for the CDF */
  constexpr float SQRT_PI_INV = 0.56418958354775628694807945156079f;
  float normalization = 1.f / (1.f + c + SQRT_PI_INV * tanThetaI * expf(-cotThetaI * cotThetaI));

  int it = 0;
  while (++it < 10) {
    /* Bisection criterion -- the oddly-looking
       Boolean expression are intentional to check
       for NaNs at little additional cost */
    if (!(b >= a && b <= c))
      b = 0.5f * (a + c);

    /* Evaluate the CDF and its derivative
       (i.e. the density function) */
    float invErf = ErfInv(b);
    float value = normalization * (1.f + b + SQRT_PI_INV * tanThetaI * expf(-invErf * invErf)) - sample_x;
    float derivative = normalization * (1.f - invErf * tanThetaI);

    if (fabsf(value) < 1e-5f)
      break;

    /* Update bisection intervals */
    if (value > 0.f)
      c = b;
    else
      a = b;

    b -= value / derivative;
  }

  /* Now convert back into a slope value */
  *slope_x = ErfInv(b);

  /* Simulate Y component */
  *slope_y = ErfInv(2.0f * std::max(U2, (float)1e-6f) - 1.0f);
}
__device__ float3 BeckmannSample(const float3 &wi, float alpha_x, float alpha_y, float U1, float U2) {
  // 1. stretch wi
  float3 wiStretched = math::normalize3(float3{alpha_x * wi.x, alpha_y * wi.y, wi.z});

  // 2. simulate P22_{wi}(x_slope, y_slope, 1, 1)
  float slope_x, slope_y;
  BeckmannSample11(CosTheta(wiStretched), U1, U2, &slope_x, &slope_y);

  // 3. rotate
  float tmp = CosPhi(wiStretched) * slope_x - SinPhi(wiStretched) * slope_y;
  slope_y = SinPhi(wiStretched) * slope_x + CosPhi(wiStretched) * slope_y;
  slope_x = tmp;

  // 4. unstretch
  slope_x = alpha_x * slope_x;
  slope_y = alpha_y * slope_y;

  // 5. compute normal
  return math::normalize3(float3{-slope_x, -slope_y, 1.f});
}
// MicrofacetDistribution Method Definitions
__device__ MicrofacetDistribution::~MicrofacetDistribution() {}
__device__ float BeckmannDistribution::D(const float3 &wh) const {
  float tan2Theta = Tan2Theta(wh);
  if (isinf(tan2Theta))
    return 0.f;
  float cos4Theta = Cos2Theta(wh) * Cos2Theta(wh);
  return expf(-tan2Theta * (Cos2Phi(wh) / (alphax * alphax) + Sin2Phi(wh) / (alphay * alphay))) /
         (CUDART_PI_F * alphax * alphay * cos4Theta);
}
// bool isinf(float s) {
//  return (2 * s == s) && (s != 0);
//}
__device__ float TrowbridgeReitzDistribution::D(const float3 &wh) const {
  float tan2Theta = Tan2Theta(wh);
  if (isinf(tan2Theta))
    return 0.f;
  const float cos4Theta = Cos2Theta(wh) * Cos2Theta(wh);
  float e = (Cos2Phi(wh) / (alphax * alphax) + Sin2Phi(wh) / (alphay * alphay)) * tan2Theta;
  return 1.f / (CUDART_PI_F * alphax * alphay * cos4Theta * (1.f + e) * (1.f + e));
}
__device__ float BeckmannDistribution::Lambda(const float3 &w) const {
  float absTanTheta = fabsf(TanTheta(w));
  if (isinf(absTanTheta))
    return 0.f;
  // Compute _alpha_ for direction _w_
  float alpha = sqrtf(Cos2Phi(w) * alphax * alphax + Sin2Phi(w) * alphay * alphay);
  float a = 1.f / (alpha * absTanTheta);
  if (a >= 1.6f)
    return 0.f;
  return (1.f - 1.259f * a + 0.396f * a * a) / (3.535f * a + 2.181f * a * a);
}
__device__ float TrowbridgeReitzDistribution::Lambda(const float3 &w) const {
  float absTanTheta = fabsf(TanTheta(w));
  if (isinf(absTanTheta))
    return 0.f;
  // Compute _alpha_ for direction _w_
  float alpha = sqrtf(Cos2Phi(w) * alphax * alphax + Sin2Phi(w) * alphay * alphay);
  float alpha2Tan2Theta = (alpha * absTanTheta) * (alpha * absTanTheta);
  return (-1.f + sqrtf(1.f + alpha2Tan2Theta)) / 2.f;
}
__device__ float3 BeckmannDistribution::Sample_wh(const float3 &wo, const float2 &u) const {
  if (!sampleVisibleArea) {
    // Sample full distribution of normals for Beckmann distribution

    // Compute $\tan^2 \theta$ and $\phi$ for Beckmann distribution sample
    float tan2Theta, phi;
    if (alphax == alphay) {
      float logSample = logf(1.f - u.x);
      tan2Theta = -alphax * alphax * logSample;
      phi = u.y * 2.f * CUDART_PI_F;
    } else {
      // Compute _tan2Theta_ and _phi_ for anisotropic Beckmann
      // distribution
      float logSample = logf(1.f - u.x);
      phi = std::atan(alphay / alphax * std::tan(2.f * CUDART_PI_F * u.y + 0.5f * CUDART_PI_F));
      if (u.y > 0.5f)
        phi += CUDART_PI_F;
      float sinPhi = sinf(phi), cosPhi = cosf(phi);
      float alphax2 = alphax * alphax, alphay2 = alphay * alphay;
      tan2Theta = -logSample / (cosPhi * cosPhi / alphax2 + sinPhi * sinPhi / alphay2);
    }

    // Map sampled Beckmann angles to normal direction _wh_
    float cosTheta = 1.f / sqrtf(1.f + tan2Theta);
    float sinTheta = sqrtf(std::max((float)0.f, 1.f - cosTheta * cosTheta));
    float3 wh = SphericalDirection(sinTheta, cosTheta, phi);
    if (!SameHemisphere(wo, wh))
      wh = -wh;
    return wh;
  } else {
    // Sample visible area of normals for Beckmann distribution
    float3 wh;
    bool flip = wo.z < 0.f;
    wh = BeckmannSample(flip ? -wo : wo, alphax, alphay, u.x, u.y);
    if (flip)
      wh = -wh;
    return wh;
  }
}
__device__ void TrowbridgeReitzSample11(float cosTheta, float U1, float U2, float *slope_x, float *slope_y) {
  // special case (normal incidence)
  if (cosTheta > .9999f) {
    float r = sqrtf(U1 / (1 - U1));
    float phi = 6.28318530718f * U2;
    *slope_x = r * cos(phi);
    *slope_y = r * sin(phi);
    return;
  }

  float sinTheta = sqrtf(std::max((float)0, (float)1 - cosTheta * cosTheta));
  float tanTheta = sinTheta / cosTheta;
  float a = 1.f / tanTheta;
  float G1 = 2.f / (1.f + sqrtf(1.f + 1.f / (a * a)));

  // sample slope_x
  float A = 2.f * U1 / G1 - 1.f;
  float tmp = 1.f / (A * A - 1.f);
  if (tmp > 1e10f)
    tmp = 1e10f;
  float B = tanTheta;
  float D = sqrtf(std::max(float(B * B * tmp * tmp - (A * A - B * B) * tmp), float(0)));
  float slope_x_1 = B * tmp - D;
  float slope_x_2 = B * tmp + D;
  *slope_x = (A < 0.f || slope_x_2 > 1.f / tanTheta) ? slope_x_1 : slope_x_2;

  // sample slope_y
  float S;
  if (U2 > 0.5f) {
    S = 1.f;
    U2 = 2.f * (U2 - .5f);
  } else {
    S = -1.f;
    U2 = 2.f * (.5f - U2);
  }
  float z = (U2 * (U2 * (U2 * 0.27385f - 0.73369f) + 0.46341f)) /
            (U2 * (U2 * (U2 * 0.093073f + 0.309420f) - 1.000000f) + 0.597999f);
  *slope_y = S * z * sqrtf(1.f + *slope_x * *slope_x);
}
__device__ float3 TrowbridgeReitzSample(const float3 &wi, float alpha_x, float alpha_y, float U1, float U2) {
  // 1. stretch wi
  float3 wiStretched = math::normalize3(float3{alpha_x * wi.x, alpha_y * wi.y, wi.z});

  // 2. simulate P22_{wi}(x_slope, y_slope, 1, 1)
  float slope_x, slope_y;
  TrowbridgeReitzSample11(CosTheta(wiStretched), U1, U2, &slope_x, &slope_y);

  // 3. rotate
  float tmp = CosPhi(wiStretched) * slope_x - SinPhi(wiStretched) * slope_y;
  slope_y = SinPhi(wiStretched) * slope_x + CosPhi(wiStretched) * slope_y;
  slope_x = tmp;

  // 4. unstretch
  slope_x = alpha_x * slope_x;
  slope_y = alpha_y * slope_y;

  // 5. compute normal
  return math::normalize3(float3{-slope_x, -slope_y, 1.f});
}
__device__ float3 TrowbridgeReitzDistribution::Sample_wh(const float3 &wo, const float2 &u) const {
  float3 wh;
  if (!sampleVisibleArea) {
    float cosTheta = 0.f, phi = (2.f * CUDART_PI_F) * u.y;
    if (alphax == alphay) {
      float tanTheta2 = alphax * alphax * u.x / (1.0f - u.y);
      cosTheta = 1.f / sqrtf(1.f + tanTheta2);
    } else {
      phi = std::atan(alphay / alphax * std::tan(2.f * CUDART_PI_F * u.y + .5f * CUDART_PI_F));
      if (u.y > .5f)
        phi += CUDART_PI_F;
      float sinPhi = sinf(phi), cosPhi = cosf(phi);
      const float alphax2 = alphax * alphax, alphay2 = alphay * alphay;
      const float alpha2 = 1.f / (cosPhi * cosPhi / alphax2 + sinPhi * sinPhi / alphay2);
      float tanTheta2 = alpha2 * u.x / (1.f - u.x);
      cosTheta = 1.f / sqrtf(1.f + tanTheta2);
    }
    float sinTheta = sqrtf(std::max((float)0., (float)1. - cosTheta * cosTheta));
    wh = SphericalDirection(sinTheta, cosTheta, phi);
    if (!SameHemisphere(wo, wh))
      wh = -wh;
  } else {
    bool flip = wo.z < 0.f;
    wh = TrowbridgeReitzSample(flip ? -wo : wo, alphax, alphay, u.x, u.y);
    if (flip)
      wh = -wh;
  }
  return wh;
}
__device__ float MicrofacetDistribution::Pdf(const float3 &wo, const float3 &wh) const {
  if (sampleVisibleArea)
    return D(wh) * G1(wo) * fabsf(math::dot3(wo, wh)) / AbsCosTheta(wo);
  else
    return D(wh) * AbsCosTheta(wh);
}
// BSDF Declarations
enum BxDFType {
  BSDF_REFLECTION = 1 << 0,
  BSDF_TRANSMISSION = 1 << 1,
  BSDF_DIFFUSE = 1 << 2,
  BSDF_GLOSSY = 1 << 3,
  BSDF_SPECULAR = 1 << 4,
  BSDF_ALL = BSDF_DIFFUSE | BSDF_GLOSSY | BSDF_SPECULAR | BSDF_REFLECTION | BSDF_TRANSMISSION,
};
enum class TransportMode { Radiance, Importance };
class BxDF;
class BSDF {
public:
  // BSDF Public Methods
  deviceInline BSDF(const hitInformation &si, float eta = 1)
      : eta(eta), ns(si.normal), ng(si.normal), ss(math::normalize(math::cross(si.normal, si.rayDirection))),
        ts(math::normalize(math::cross(si.normal, ss))) {}
  deviceInline void Add(BxDF *b) { bxdfs[nBxDFs++] = b; }
  deviceInline int NumComponents(BxDFType flags = BSDF_ALL) const;
  deviceInline float3 WorldToLocal(const float3 &v) const {
    return float3{math::dot3(v, ss), math::dot3(v, ts), math::dot3(v, ns)};
  }
  deviceInline float3 LocalToWorld(const float3 &v) const {
    return float3{ss.x * v.x + ts.x * v.y + ns.x * v.z, ss.y * v.x + ts.y * v.y + ns.y * v.z,
                  ss.z * v.x + ts.z * v.y + ns.z * v.z};
  }
  deviceInline float3 f(const float3 &woW, const float3 &wiW, BxDFType flags = BSDF_ALL) const;
  deviceInline float3 rho(int nSamples, const float2 *samples1, const float2 *samples2,
                          BxDFType flags = BSDF_ALL) const;
  deviceInline float3 rho(const float3 &wo, int nSamples, const float2 *samples, BxDFType flags = BSDF_ALL) const;
  deviceInline float3 Sample_f(const float3 &wo, float3 *wi, const float2 &u, float *pdf, BxDFType type = BSDF_ALL,
                               BxDFType *sampledType = nullptr) const;
  deviceInline float Pdf(const float3 &wo, const float3 &wi, BxDFType flags = BSDF_ALL) const;
  // BSDF Public Data
  const float eta;

private:
  // BSDF Private Data
  const float3 ns, ng;
  const float3 ss, ts;
  int nBxDFs = 0;
  static constexpr int MaxBxDFs = 8;
  BxDF *bxdfs[MaxBxDFs];
  friend class MixMaterial;
};
// BxDF Declarations
class BxDF {
public:
  // BxDF Interface
  deviceInline virtual ~BxDF() {}
  deviceInline BxDF(BxDFType type) : type(type) {}
  deviceInline bool MatchesFlags(BxDFType t) const { return (type & t) == type; }
  deviceInline virtual float3 f(const float3 &wo, const float3 &wi) const = 0;
  deviceInline virtual float3 Sample_f(const float3 &wo, float3 *wi, const float2 &sample, float *pdf,
                                       BxDFType *sampledType = nullptr) const;
  deviceInline virtual float3 rho(const float3 &wo, int nSamples, const float2 *samples) const;
  deviceInline virtual float3 rho(int nSamples, const float2 *samples1, const float2 *samples2) const;
  deviceInline virtual float Pdf(const float3 &wo, const float3 &wi) const;
  // BxDF Public Data
  const BxDFType type;
};
class ScaledBxDF : public BxDF {
public:
  // ScaledBxDF Public Methods
  deviceInline ScaledBxDF(BxDF *bxdf, const float3 &scale) : BxDF(BxDFType(bxdf->type)), bxdf(bxdf), scale(scale) {}
  deviceInline float3 rho(const float3 &w, int nSamples, const float2 *samples) const {
    return scale * bxdf->rho(w, nSamples, samples);
  }
  deviceInline float3 rho(int nSamples, const float2 *samples1, const float2 *samples2) const {
    return scale * bxdf->rho(nSamples, samples1, samples2);
  }
  deviceInline float3 f(const float3 &wo, const float3 &wi) const;
  deviceInline float3 Sample_f(const float3 &wo, float3 *wi, const float2 &sample, float *pdf,
                               BxDFType *sampledType) const;
  deviceInline float Pdf(const float3 &wo, const float3 &wi) const;

private:
  BxDF *bxdf;
  float3 scale;
};
class Fresnel {
public:
  // Fresnel Interface
  deviceInline virtual ~Fresnel();
  deviceInline virtual float Evaluate(float cosI) const = 0;
};
class FresnelConductor : public Fresnel {
public:
  // FresnelConductor Public Methods
  deviceInline float Evaluate(float cosThetaI) const;
  deviceInline FresnelConductor(const float &etaI, const float &etaT, const float &k) : etaI(etaI), etaT(etaT), k(k) {}

private:
  float etaI, etaT, k;
};
class FresnelDielectric : public Fresnel {
public:
  // FresnelDielectric Public Methods
  deviceInline float Evaluate(float cosThetaI) const;
  deviceInline FresnelDielectric(float etaI, float etaT) : etaI(etaI), etaT(etaT) {}

private:
  float etaI, etaT;
};
class FresnelNoOp : public Fresnel {
public:
  deviceInline float Evaluate(float) const { return 1.f; }
};
class SpecularReflection : public BxDF {
public:
  // SpecularReflection Public Methods
  deviceInline SpecularReflection(const float3 &R, Fresnel *fresnel)
      : BxDF(BxDFType(BSDF_REFLECTION | BSDF_SPECULAR)), R(R), fresnel(fresnel) {}
  deviceInline float3 f(const float3 &wo, const float3 &wi) const { return float3{0.f, 0.f, 0.f}; }
  deviceInline float3 Sample_f(const float3 &wo, float3 *wi, const float2 &sample, float *pdf,
                               BxDFType *sampledType) const;
  deviceInline float Pdf(const float3 &wo, const float3 &wi) const { return 0; }

private:
  // SpecularReflection Private Data
  const float3 R;
  const Fresnel *fresnel;
};
class SpecularTransmission : public BxDF {
public:
  // SpecularTransmission Public Methods
  deviceInline SpecularTransmission(const float3 &T, float etaA, float etaB, TransportMode mode)
      : BxDF(BxDFType(BSDF_TRANSMISSION | BSDF_SPECULAR)), T(T), etaA(etaA), etaB(etaB), fresnel(etaA, etaB),
        mode(mode) {}
  deviceInline float3 f(const float3 &wo, const float3 &wi) const { return float3{0.f, 0.f, 0.f}; }
  deviceInline float3 Sample_f(const float3 &wo, float3 *wi, const float2 &sample, float *pdf,
                               BxDFType *sampledType) const;
  deviceInline float Pdf(const float3 &wo, const float3 &wi) const { return 0; }

private:
  // SpecularTransmission Private Data
  const float3 T;
  const float etaA, etaB;
  const FresnelDielectric fresnel;
  const TransportMode mode;
};
class FresnelSpecular : public BxDF {
public:
  // FresnelSpecular Public Methods
  deviceInline FresnelSpecular(const float3 &R, const float3 &T, float etaA, float etaB, TransportMode mode)
      : BxDF(BxDFType(BSDF_REFLECTION | BSDF_TRANSMISSION | BSDF_SPECULAR)), R(R), T(T), etaA(etaA), etaB(etaB),
        mode(mode) {}
  deviceInline float3 f(const float3 &wo, const float3 &wi) const { return f3_0; }
  deviceInline float3 Sample_f(const float3 &wo, float3 *wi, const float2 &u, float *pdf, BxDFType *sampledType) const;
  deviceInline float Pdf(const float3 &wo, const float3 &wi) const { return 0; }

private:
  // FresnelSpecular Private Data
  const float3 R, T;
  const float etaA, etaB;
  const TransportMode mode;
};
class LambertianReflection : public BxDF {
public:
  // LambertianReflection Public Methods
  deviceInline LambertianReflection(const float3 &R) : BxDF(BxDFType(BSDF_REFLECTION | BSDF_DIFFUSE)), R(R) {}
  deviceInline float3 f(const float3 &wo, const float3 &wi) const;
  deviceInline float3 rho(const float3 &, int, const float2 *) const { return R; }
  deviceInline float3 rho(int, const float2 *, const float2 *) const { return R; }

private:
  // LambertianReflection Private Data
  const float3 R;
};
class LambertianTransmission : public BxDF {
public:
  // LambertianTransmission Public Methods
  deviceInline LambertianTransmission(const float3 &T) : BxDF(BxDFType(BSDF_TRANSMISSION | BSDF_DIFFUSE)), T(T) {}
  deviceInline float3 f(const float3 &wo, const float3 &wi) const;
  deviceInline float3 rho(const float3 &, int, const float2 *) const { return T; }
  deviceInline float3 rho(int, const float2 *, const float2 *) const { return T; }
  deviceInline float3 Sample_f(const float3 &wo, float3 *wi, const float2 &u, float *pdf, BxDFType *sampledType) const;
  deviceInline float Pdf(const float3 &wo, const float3 &wi) const;

private:
  // LambertianTransmission Private Data
  float3 T;
};
class OrenNayar : public BxDF {
public:
  // OrenNayar Public Methods
  deviceInline float3 f(const float3 &wo, const float3 &wi) const;
  deviceInline OrenNayar(const float3 &R, float sigma) : BxDF(BxDFType(BSDF_REFLECTION | BSDF_DIFFUSE)), R(R) {
    sigma = degToRad(sigma);
    float sigma2 = sigma * sigma;
    A = 1.f - (sigma2 / (2.f * (sigma2 + 0.33f)));
    B = 0.45f * sigma2 / (sigma2 + 0.09f);
  }

private:
  // OrenNayar Private Data
  const float3 R;
  float A, B;
};
class MicrofacetReflection : public BxDF {
public:
  // MicrofacetReflection Public Methods
  deviceInline MicrofacetReflection(const float3 &R, MicrofacetDistribution *distribution, Fresnel *fresnel)
      : BxDF(BxDFType(BSDF_REFLECTION | BSDF_GLOSSY)), R(R), distribution(distribution), fresnel(fresnel) {}
  deviceInline float3 f(const float3 &wo, const float3 &wi) const;
  deviceInline float3 Sample_f(const float3 &wo, float3 *wi, const float2 &u, float *pdf, BxDFType *sampledType) const;
  deviceInline float Pdf(const float3 &wo, const float3 &wi) const;

private:
  // MicrofacetReflection Private Data
  const float3 R;
  const MicrofacetDistribution *distribution;
  const Fresnel *fresnel;
};
class MicrofacetTransmission : public BxDF {
public:
  // MicrofacetTransmission Public Methods
  deviceInline MicrofacetTransmission(const float3 &T, MicrofacetDistribution *distribution, float etaA, float etaB,
                                      TransportMode mode)
      : BxDF(BxDFType(BSDF_TRANSMISSION | BSDF_GLOSSY)), T(T), distribution(distribution), etaA(etaA), etaB(etaB),
        fresnel(etaA, etaB), mode(mode) {}
  deviceInline float3 f(const float3 &wo, const float3 &wi) const;
  deviceInline float3 Sample_f(const float3 &wo, float3 *wi, const float2 &u, float *pdf, BxDFType *sampledType) const;
  deviceInline float Pdf(const float3 &wo, const float3 &wi) const;

private:
  // MicrofacetTransmission Private Data
  const float3 T;
  const MicrofacetDistribution *distribution;
  const float etaA, etaB;
  const FresnelDielectric fresnel;
  const TransportMode mode;
};
class FresnelBlend : public BxDF {
public:
  // FresnelBlend Public Methods
  deviceInline FresnelBlend(const float3 &Rd, const float3 &Rs, MicrofacetDistribution *distrib);
  deviceInline float3 f(const float3 &wo, const float3 &wi) const;
  deviceInline float3 SchlickFresnel(float cosTheta) const {
    auto pow5 = [](float v) { return (v * v) * (v * v) * v; };
    return Rs + pow5(1.f - cosTheta) * (float3{1.f, 1.f, 1.f} - Rs);
  }
  deviceInline float3 Sample_f(const float3 &wi, float3 *sampled_f, const float2 &u, float *pdf,
                               BxDFType *sampledType) const;
  deviceInline float Pdf(const float3 &wo, const float3 &wi) const;

private:
  // FresnelBlend Private Data
  const float3 Rd, Rs;
  MicrofacetDistribution *distribution;
};
// BSDF Inline Method Definitions
deviceInline int BSDF::NumComponents(BxDFType flags) const {
  int num = 0;
  for (int i = 0; i < nBxDFs; ++i)
    if (bxdfs[i]->MatchesFlags(flags))
      ++num;
  return num;
}
// BxDF Utility Functions
deviceInline float FrDielectric(float cosThetaI, float etaI, float etaT) {
  cosThetaI = math::clamp(cosThetaI, -1.f, 1.f);
  // Potentially swap indices of refraction
  bool entering = cosThetaI > 0.f;
  if (!entering) {
    float eta = etaI;
    etaI = etaT;
    etaT = eta;
    // std::swap(etaI, etaT);
    cosThetaI = fabsf(cosThetaI);
  }

  // Compute _cosThetaT_ using Snell's law
  float sinThetaI = sqrtf(std::max((float)0, 1.f - cosThetaI * cosThetaI));
  float sinThetaT = etaI / etaT * sinThetaI;

  // Handle total internal reflection
  if (sinThetaT >= 1.f)
    return 1.f;
  float cosThetaT = sqrtf(std::max((float)0, 1 - sinThetaT * sinThetaT));
  float Rparl = ((etaT * cosThetaI) - (etaI * cosThetaT)) / ((etaT * cosThetaI) + (etaI * cosThetaT));
  float Rperp = ((etaI * cosThetaI) - (etaT * cosThetaT)) / ((etaI * cosThetaI) + (etaT * cosThetaT));
  return (Rparl * Rparl + Rperp * Rperp) / 2.f;
}
// https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/
deviceInline float FrConductor(float cosThetaI, const float &etai, const float &etat, const float &k) {
  cosThetaI = math::clamp(cosThetaI, -1.f, 1.f);
  float eta = etat / etai;
  float etak = k / etai;

  float cosThetaI2 = cosThetaI * cosThetaI;
  float sinThetaI2 = 1.f - cosThetaI2;
  float eta2 = eta * eta;
  float etak2 = etak * etak;

  float t0 = eta2 - etak2 - sinThetaI2;
  float a2plusb2 = math::sqrt(t0 * t0 + 4.f * eta2 * etak2);
  float t1 = a2plusb2 + cosThetaI2;
  float a = math::sqrt(0.5f * (a2plusb2 + t0));
  float t2 = (float)2 * cosThetaI * a;
  float Rs = (t1 - t2) / (t1 + t2);

  float t3 = cosThetaI2 * a2plusb2 + sinThetaI2 * sinThetaI2;
  float t4 = t2 * sinThetaI2;
  float Rp = Rs * (t3 - t4) / (t3 + t4);

  return 0.5f * (Rp + Rs);
}
// BxDF Method Definitions
deviceInline float3 ScaledBxDF::f(const float3 &wo, const float3 &wi) const { return scale * bxdf->f(wo, wi); }
deviceInline float3 ScaledBxDF::Sample_f(const float3 &wo, float3 *wi, const float2 &sample, float *pdf,
                                         BxDFType *sampledType) const {
  float3 f = bxdf->Sample_f(wo, wi, sample, pdf, sampledType);
  return scale * f;
}
deviceInline float ScaledBxDF::Pdf(const float3 &wo, const float3 &wi) const { return bxdf->Pdf(wo, wi); }
deviceInline Fresnel::~Fresnel() {}
deviceInline float FresnelConductor::Evaluate(float cosThetaI) const {
  return FrConductor(fabsf(cosThetaI), etaI, etaT, k);
}
deviceInline float FresnelDielectric::Evaluate(float cosThetaI) const { return FrDielectric(cosThetaI, etaI, etaT); }
deviceInline float3 Faceforward(const float3 &v, const float3 &v2) { return (math::dot3(v, v2) < 0.f) ? -v : v; }
deviceInline float3 SpecularReflection::Sample_f(const float3 &wo, float3 *wi, const float2 &sample, float *pdf,
                                                 BxDFType *sampledType) const {
  // Compute perfect specular reflection direction
  *wi = float3{-wo.x, -wo.y, wo.z};
  *pdf = 1.f;
  return fresnel->Evaluate(CosTheta(*wi)) * R / AbsCosTheta(*wi);
}
deviceInline float3 SpecularTransmission::Sample_f(const float3 &wo, float3 *wi, const float2 &sample, float *pdf,
                                                   BxDFType *sampledType) const {
  // Figure out which $\eta$ is incident and which is transmitted
  bool entering = CosTheta(wo) > 0.f;
  float etaI = entering ? etaA : etaB;
  float etaT = entering ? etaB : etaA;

  // Compute ray direction for specular transmission
  if (!Refract(wo, Faceforward(float3{0, 0, 1}, wo), etaI / etaT, wi))
    return f3_0;
  *pdf = 1;
  float3 ft = T * (float3{1.f, 1.f, 1.f} - fresnel.Evaluate(CosTheta(*wi)));
  // Account for non-symmetry with transmission to different medium
  if (mode == TransportMode::Radiance)
    ft *= (etaI * etaI) / (etaT * etaT);
  return ft / AbsCosTheta(*wi);
}
deviceInline float3 LambertianReflection::f(const float3 &wo, const float3 &wi) const { return R / CUDART_PI_F; }
deviceInline float3 LambertianTransmission::f(const float3 &wo, const float3 &wi) const { return T / CUDART_PI_F; }
deviceInline float3 OrenNayar::f(const float3 &wo, const float3 &wi) const {
  float sinThetaI = SinTheta(wi);
  float sinThetaO = SinTheta(wo);
  // Compute cosine term of Oren-Nayar model
  float maxCos = 0;
  if (sinThetaI > 1e-4f && sinThetaO > 1e-4f) {
    float sinPhiI = SinPhi(wi), cosPhiI = CosPhi(wi);
    float sinPhiO = SinPhi(wo), cosPhiO = CosPhi(wo);
    float dCos = cosPhiI * cosPhiO + sinPhiI * sinPhiO;
    maxCos = std::max((float)0, dCos);
  }

  // Compute sine and tangent terms of Oren-Nayar model
  float sinAlpha, tanBeta;
  if (AbsCosTheta(wi) > AbsCosTheta(wo)) {
    sinAlpha = sinThetaO;
    tanBeta = sinThetaI / AbsCosTheta(wi);
  } else {
    sinAlpha = sinThetaI;
    tanBeta = sinThetaO / AbsCosTheta(wo);
  }
  return R / CUDART_PI_F * (A + B * maxCos * sinAlpha * tanBeta);
}
deviceInline float3 MicrofacetReflection::f(const float3 &wo, const float3 &wi) const {
  if (!SameHemisphere(wo, wi))
    return f3_0; // transmission only
  float cosThetaO = AbsCosTheta(wo), cosThetaI = AbsCosTheta(wi);
  float3 wh = wi + wo;
  // Handle degenerate cases for microfacet reflection
  if (cosThetaI == 0 || cosThetaO == 0)
    return f3_0;
  if (wh.x == 0 && wh.y == 0 && wh.z == 0)
    return f3_0;
  wh = math::normalize(wh);
  // For the Fresnel call, make sure that wh is in the same hemisphere
  // as the surface normal, so that TIR is handled correctly.
  float F = fresnel->Evaluate(math::dot3(wi, Faceforward(wh, float3{0, 0, 1})));
  return R * distribution->D(wh) * distribution->G(wo, wi) * F / (4.f * cosThetaI * cosThetaO);
}
deviceInline float3 MicrofacetTransmission::f(const float3 &wo, const float3 &wi) const {
  if (SameHemisphere(wo, wi))
    return f3_0; // transmission only

  float cosThetaO = CosTheta(wo);
  float cosThetaI = CosTheta(wi);
  if (cosThetaI == 0.f || cosThetaO == 0.f)
    return f3_0;

  // Compute $\wh$ from $\wo$ and $\wi$ for microfacet transmission
  float eta = CosTheta(wo) > 0.f ? (etaB / etaA) : (etaA / etaB);
  float3 wh = math::normalize(wo + wi * eta);
  if (wh.z < 0.f)
    wh = -wh;

  float F = fresnel.Evaluate(math::dot3(wo, wh));

  float sqrtDenom = math::dot3(wo, wh) + eta * math::dot3(wi, wh);
  float factor = (mode == TransportMode::Radiance) ? (1.f / eta) : 1.f;

  return (float3{1.f, 1.f, 1.f} - F) * T *
         fabsf(distribution->D(wh) * distribution->G(wo, wi) * eta * eta * fabsf(math::dot3(wi, wh)) *
               fabsf(math::dot3(wo, wh)) * factor * factor / (cosThetaI * cosThetaO * sqrtDenom * sqrtDenom));
}
deviceInline FresnelBlend::FresnelBlend(const float3 &Rd, const float3 &Rs, MicrofacetDistribution *distribution)
    : BxDF(BxDFType(BSDF_REFLECTION | BSDF_GLOSSY)), Rd(Rd), Rs(Rs), distribution(distribution) {}
deviceInline float3 FresnelBlend::f(const float3 &wo, const float3 &wi) const {
  auto pow5 = [](float v) { return (v * v) * (v * v) * v; };
  float3 diffuse = (28.f / (23.f * CUDART_PI_F)) * Rd * (float3{1.f, 1.f, 1.f} - Rs) *
                   (1.f - pow5(1.f - .5f * AbsCosTheta(wi))) * (1.f - pow5(1.f - .5f * AbsCosTheta(wo)));
  float3 wh = wi + wo;
  if (wh.x == 0.f && wh.y == 0.f && wh.z == 0.f)
    return f3_0;
  wh = math::normalize(wh);
  float3 specular = distribution->D(wh) /
                    (4.f * fabsf(math::dot3(wi, wh)) * std::max(AbsCosTheta(wi), AbsCosTheta(wo))) *
                    SchlickFresnel(math::dot3(wi, wh));
  return diffuse + specular;
}
deviceInline float2 ConcentricSampleDisk(const float2 &u) {
  // Map uniform random numbers to $[-1,1]^2$
  float2 uOffset = 2.f * u - float2{1, 1};

  // Handle degeneracy at the origin
  if (uOffset.x == 0 && uOffset.y == 0)
    return float2{0, 0};

  // Apply concentric mapping to point
  float theta, r;
  if (fabsf(uOffset.x) > fabsf(uOffset.y)) {
    r = uOffset.x;
    theta = CUDART_PIO4_F * (uOffset.y / uOffset.x);
  } else {
    r = uOffset.y;
    theta = CUDART_PIO2_F - CUDART_PIO4_F * (uOffset.x / uOffset.y);
  }
  return r * float2{cosf(theta), sinf(theta)};
}
deviceInline float3 CosineSampleHemisphere(const float2 &u) {
  float2 d = ConcentricSampleDisk(u);
  float z = sqrtf(std::max((float)0, 1 - d.x * d.x - d.y * d.y));
  return float3{d.x, d.y, z};
}
deviceInline float3 BxDF::Sample_f(const float3 &wo, float3 *wi, const float2 &u, float *pdf,
                                   BxDFType *sampledType) const {
  // Cosine-sample the hemisphere, flipping the direction if necessary
  *wi = CosineSampleHemisphere(u);
  if (wo.z < 0.f)
    wi->z *= -1.f;
  *pdf = Pdf(wo, *wi);
  return f(wo, *wi);
}
deviceInline float BxDF::Pdf(const float3 &wo, const float3 &wi) const {
  return SameHemisphere(wo, wi) ? AbsCosTheta(wi) / CUDART_PI_F : 0.f;
}
deviceInline float3 LambertianTransmission::Sample_f(const float3 &wo, float3 *wi, const float2 &u, float *pdf,
                                                     BxDFType *sampledType) const {
  *wi = CosineSampleHemisphere(u);
  if (wo.z > 0.f)
    wi->z *= -1.f;
  *pdf = Pdf(wo, *wi);
  return f(wo, *wi);
}
deviceInline float LambertianTransmission::Pdf(const float3 &wo, const float3 &wi) const {
  return !SameHemisphere(wo, wi) ? AbsCosTheta(wi) / CUDART_PI_F : 0;
}
deviceInline float3 MicrofacetReflection::Sample_f(const float3 &wo, float3 *wi, const float2 &u, float *pdf,
                                                   BxDFType *sampledType) const {
  // Sample microfacet orientation $\wh$ and reflected direction $\wi$
  if (wo.z == 0.f)
    return f3_0;
  float3 wh = distribution->Sample_wh(wo, u);
  if (math::dot3(wo, wh) < 0.f)
    return f3_0; // Should be rare
  *wi = Reflect(wo, wh);
  if (!SameHemisphere(wo, *wi))
    return f3_0;

  // Compute PDF of _wi_ for microfacet reflection
  *pdf = distribution->Pdf(wo, wh) / (4.f * math::dot3(wo, wh));
  return f(wo, *wi);
}
deviceInline float MicrofacetReflection::Pdf(const float3 &wo, const float3 &wi) const {
  if (!SameHemisphere(wo, wi))
    return 0;
  float3 wh = math::normalize(wo + wi);
  return distribution->Pdf(wo, wh) / (4.f * math::dot3(wo, wh));
}
deviceInline float3 MicrofacetTransmission::Sample_f(const float3 &wo, float3 *wi, const float2 &u, float *pdf,
                                                     BxDFType *sampledType) const {
  if (wo.z == 0.f)
    return f3_0;
  float3 wh = distribution->Sample_wh(wo, u);
  if (math::dot3(wo, wh) < 0.f)
    return f3_0; // Should be rare

  float eta = CosTheta(wo) > 0.f ? (etaA / etaB) : (etaB / etaA);
  if (!Refract(wo, (float3)wh, eta, wi))
    return f3_0;
  *pdf = Pdf(wo, *wi);
  return f(wo, *wi);
}
deviceInline float MicrofacetTransmission::Pdf(const float3 &wo, const float3 &wi) const {
  if (SameHemisphere(wo, wi))
    return 0;
  // Compute $\wh$ from $\wo$ and $\wi$ for microfacet transmission
  float eta = CosTheta(wo) > 0.f ? (etaB / etaA) : (etaA / etaB);
  float3 wh = math::normalize(wo + wi * eta);

  // Compute change of variables _dwh\_dwi_ for microfacet transmission
  float sqrtDenom = math::dot3(wo, wh) + eta * math::dot3(wi, wh);
  float dwh_dwi = fabsf((eta * eta * math::dot3(wi, wh)) / (sqrtDenom * sqrtDenom));
  return distribution->Pdf(wo, wh) * dwh_dwi;
}
__constant__ double DoubleOneMinusEpsilon = 0.99999999999999989;
__constant__ float FloatOneMinusEpsilon = 0.99999994f;
__constant__ float OneMinusEpsilon = 0.99999994f;
deviceInline float3 FresnelBlend::Sample_f(const float3 &wo, float3 *wi, const float2 &uOrig, float *pdf,
                                           BxDFType *sampledType) const {
  float2 u = uOrig;
  if (u.x < .5f) {
    u.x = std::min(2.f * u.x, OneMinusEpsilon);
    // Cosine-sample the hemisphere, flipping the direction if necessary
    *wi = CosineSampleHemisphere(u);
    if (wo.z < 0.f)
      wi->z *= -1.f;
  } else {
    u.x = std::min(2.f * (u.x - .5f), OneMinusEpsilon);
    // Sample microfacet orientation $\wh$ and reflected direction $\wi$
    float3 wh = distribution->Sample_wh(wo, u);
    *wi = Reflect(wo, wh);
    if (!SameHemisphere(wo, *wi))
      return f3_0;
  }
  *pdf = Pdf(wo, *wi);
  return f(wo, *wi);
}
deviceInline float FresnelBlend::Pdf(const float3 &wo, const float3 &wi) const {
  if (!SameHemisphere(wo, wi))
    return 0;
  float3 wh = math::normalize(wo + wi);
  float pdf_wh = distribution->Pdf(wo, wh);
  return .5f * (AbsCosTheta(wi) / CUDART_PI_F + pdf_wh / (4 * math::dot3(wo, wh)));
}
deviceInline float3 FresnelSpecular::Sample_f(const float3 &wo, float3 *wi, const float2 &u, float *pdf,
                                              BxDFType *sampledType) const {
  float F = FrDielectric(CosTheta(wo), etaA, etaB);
  if (u.x < F) {
    // Compute specular reflection for _FresnelSpecular_

    // Compute perfect specular reflection direction
    *wi = float3{-wo.x, -wo.y, wo.z};
    if (sampledType)
      *sampledType = BxDFType(BSDF_SPECULAR | BSDF_REFLECTION);
    *pdf = F;
    return F * R / AbsCosTheta(*wi);
  } else {
    // Compute specular transmission for _FresnelSpecular_

    // Figure out which $\eta$ is incident and which is transmitted
    bool entering = CosTheta(wo) > 0.f;
    float etaI = entering ? etaA : etaB;
    float etaT = entering ? etaB : etaA;

    // Compute ray direction for specular transmission
    if (!Refract(wo, Faceforward(float3{0, 0, 1}, wo), etaI / etaT, wi))
      return f3_0;
    float3 ft = T * (1 - F);

    // Account for non-symmetry with transmission to different medium
    if (mode == TransportMode::Radiance)
      ft *= (etaI * etaI) / (etaT * etaT);
    if (sampledType)
      *sampledType = BxDFType(BSDF_SPECULAR | BSDF_TRANSMISSION);
    *pdf = 1.f - F;
    return ft / AbsCosTheta(*wi);
  }
}
deviceInline float3 BxDF::rho(const float3 &w, int nSamples, const float2 *u) const {
  float3 r = f3_0;
  for (int i = 0; i < nSamples; ++i) {
    // Estimate one term of $\rho_\roman{hd}$
    float3 wi;
    float pdf = 0.f;
    float3 f = Sample_f(w, &wi, u[i], &pdf);
    if (pdf > 0.f)
      r += f * AbsCosTheta(wi) / pdf;
  }
  return r / nSamples;
}
deviceInline float3 UniformSampleHemisphere(const float2 &u) {
  float z = u.x;
  float r = sqrtf(std::max((float)0, (float)1. - z * z));
  float phi = 2.f * CUDART_PI_F * u.y;
  return float3{r * cosf(phi), r * sinf(phi), z};
}
deviceInline float UniformHemispherePdf() { return 2.f / CUDART_PI_F; }
deviceInline float3 BxDF::rho(int nSamples, const float2 *u1, const float2 *u2) const {
  float3 r = f3_0;
  for (int i = 0; i < nSamples; ++i) {
    // Estimate one term of $\rho_\roman{hh}$
    float3 wo, wi;
    wo = UniformSampleHemisphere(u1[i]);
    float pdfo = UniformHemispherePdf(), pdfi = 0.f;
    float3 f = Sample_f(wo, &wi, u2[i], &pdfi);
    if (pdfi > 0.f)
      r += f * AbsCosTheta(wi) * AbsCosTheta(wo) / (pdfo * pdfi);
  }
  return r / (CUDART_PI_F * nSamples);
}
// BSDF Method Definitions
deviceInline float3 BSDF::f(const float3 &woW, const float3 &wiW, BxDFType flags) const {
  float3 wi = WorldToLocal(wiW), wo = WorldToLocal(woW);
  if (wo.z == 0.f)
    return f3_0;
  bool reflect = math::dot3(wiW, ng) * math::dot3(woW, ng) > 0.f;
  float3 f = f3_0;
  for (int i = 0; i < nBxDFs; ++i)
    if (bxdfs[i]->MatchesFlags(flags) &&
        ((reflect && (bxdfs[i]->type & BSDF_REFLECTION)) || (!reflect && (bxdfs[i]->type & BSDF_TRANSMISSION))))
      f += bxdfs[i]->f(wo, wi);
  return f;
}
deviceInline float3 BSDF::rho(int nSamples, const float2 *samples1, const float2 *samples2, BxDFType flags) const {
  float3 ret = f3_0;
  for (int i = 0; i < nBxDFs; ++i)
    if (bxdfs[i]->MatchesFlags(flags))
      ret += bxdfs[i]->rho(nSamples, samples1, samples2);
  return ret;
}
deviceInline float3 BSDF::rho(const float3 &woWorld, int nSamples, const float2 *samples, BxDFType flags) const {
  float3 wo = WorldToLocal(woWorld);
  float3 ret = f3_0;
  for (int i = 0; i < nBxDFs; ++i)
    if (bxdfs[i]->MatchesFlags(flags))
      ret += bxdfs[i]->rho(wo, nSamples, samples);
  return ret;
}
deviceInline float3 BSDF::Sample_f(const float3 &woWorld, float3 *wiWorld, const float2 &u, float *pdf, BxDFType type,
                                   BxDFType *sampledType) const {
  // Choose which _BxDF_ to sample
  int matchingComps = NumComponents(type);
  if (matchingComps == 0) {
    *pdf = 0;
    if (sampledType)
      *sampledType = BxDFType(0);
    return f3_0;
  }
  int comp = std::min((int)std::floor(u.x * matchingComps), matchingComps - 1);

  // Get _BxDF_ pointer for chosen component
  BxDF *bxdf = nullptr;
  int count = comp;
  for (int i = 0; i < nBxDFs; ++i)
    if (bxdfs[i]->MatchesFlags(type) && count-- == 0) {
      bxdf = bxdfs[i];
      break;
    }
  // Remap _BxDF_ sample _u_ to $[0,1)^2$
  float2 uRemapped{std::min(u.x * (float)matchingComps - (float)comp, OneMinusEpsilon), u.y};

  // Sample chosen _BxDF_
  float3 wi, wo = WorldToLocal(woWorld);
  if (wo.z == 0)
    return f3_0;
  *pdf = 0;
  if (sampledType)
    *sampledType = bxdf->type;
  float3 f = bxdf->Sample_f(wo, &wi, uRemapped, pdf, sampledType);
  if (*pdf == 0) {
    if (sampledType)
      *sampledType = BxDFType(0);
    return f3_0;
  }
  *wiWorld = LocalToWorld(wi);

  // Compute overall PDF with all matching _BxDF_s
  if (!(bxdf->type & BSDF_SPECULAR) && matchingComps > 1)
    for (int i = 0; i < nBxDFs; ++i)
      if (bxdfs[i] != bxdf && bxdfs[i]->MatchesFlags(type))
        *pdf += bxdfs[i]->Pdf(wo, wi);

  if (matchingComps > 1)
    *pdf /= (float)matchingComps;

  // Compute value of BSDF for sampled direction
  if (!(bxdf->type & BSDF_SPECULAR)) {
    bool reflect = math::dot3(*wiWorld, ng) * math::dot3(woWorld, ng) > 0;
    f = f3_0;
    for (int i = 0; i < nBxDFs; ++i)
      if (bxdfs[i]->MatchesFlags(type) &&
          ((reflect && (bxdfs[i]->type & BSDF_REFLECTION)) || (!reflect && (bxdfs[i]->type & BSDF_TRANSMISSION))))
        f += bxdfs[i]->f(wo, wi);
  }
  // if (math::length3(f) > 10.f && (bxdf->MatchesFlags(BSDF_REFLECTION) || bxdf->MatchesFlags(BSDF_TRANSMISSION)))
  //  printf("[%f %f %f] -> [%f %f %f] -> [%f %f %f] @ %f / %d\n", wo.x, wo.y, wo.z, wi.x, wi.y, wi.z, f.x, f.y, f.z,
  //         *pdf, matchingComps);
  return f;
}
deviceInline float BSDF::Pdf(const float3 &woWorld, const float3 &wiWorld, BxDFType flags) const {
  if (nBxDFs == 0.f)
    return 0.f;
  float3 wo = WorldToLocal(woWorld), wi = WorldToLocal(wiWorld);
  if (wo.z == 0)
    return 0.;
  float pdf = 0.f;
  int matchingComps = 0;
  for (int i = 0; i < nBxDFs; ++i)
    if (bxdfs[i]->MatchesFlags(flags)) {
      ++matchingComps;
      pdf += bxdfs[i]->Pdf(wo, wi);
    }
  float v = matchingComps > 0 ? pdf / matchingComps : 0.f;
  return v;
}
} // namespace pbrt

namespace shading {
__device__ bool intersectFluid(int32_t tIdx, hitInformation &hit, const worldRay &ray) {
  auto fluidIntersection = cfluidIntersection[tIdx];
  auto depth = fluidIntersection.w;
  bool fluidHit = depth < hit.depth;
  if (fluidHit) {
    float3 normal = math::castTo<float3>(fluidIntersection);
    normal = math::normalize(normal);
    float3 color = !fluidMemory.vrtxSurface ? math::castTo<float3>(cFluidColor[tIdx]) : fluidMemory.vrtxFluidColor;

    hit = hitInformation{fluidHit, color,  float3{1.f, 1.f, 1.f},     fluidMemory.IOR,
                         depth,    normal, fluidMemory.fluidMaterial, f3_0,
                         ray.dir};
    return true;
  }
  return false;
}
__device__ bool intersectBVH(int32_t tIdx, hitInformation &hit, const worldRay &idRay, int32_t numBVHs,
                             gpuBVH *sceneBVH) {
  bool bvhhit = false;
  int32_t idx = -1;
  float kAB = 0.f, kBC = 0.f, kCA = 0.f;
  int32_t bvh_idx = -1;
  int32_t pBestTriIdx = -1;
  for (int32_t i = 0; i < numBVHs; ++i) {
    if (!sceneBVH[i].active)
      continue;
    float ktAB = 0.f, ktBC = 0.f, ktCA = 0.f;
    float hitdistance = FLT_MAX;
    float3 boxnormal = float3{0, 0, 0};
    float3 point;
    BVH_IntersectTriangles(sceneBVH[i], idRay.orig, idRay.dir, UINT32_MAX, pBestTriIdx, point, ktAB, ktBC, ktCA,
                           hitdistance, boxnormal);

    if (pBestTriIdx != -1 &&
        (hitdistance < hit.depth /*|| (fluidHit && hitdistance < depth + fluidMemory.fluidBias)*/) &&
        hitdistance > 0.002f) // EPSILON
    {
      hit.depth = hitdistance;
      bvh_idx = i;
      kAB = ktAB;
      kBC = ktBC;
      kCA = ktCA;
      auto pBestTri = &sceneBVH[bvh_idx].pTriangles[pBestTriIdx];
      hit.normal = math::normalize(math::castTo<float3>(pBestTri->normal));
      auto i0 = pBestTri->i0;
      auto i1 = pBestTri->i1;
      auto i2 = pBestTri->i2;
      // printf("%p [%d @ %d] -> %d %d %d => %.2f [%.2f %.2f %.2f]\n", pBestTri, pBestTriIdx, bvh_idx, i0, i1, i2,
      // hitdistance, kAB, kBC, kCA);
      auto v0 = math::castTo<float3>(sceneBVH[bvh_idx].vertices[i0].position);
      auto v1 = math::castTo<float3>(sceneBVH[bvh_idx].vertices[i1].position);
      auto v2 = math::castTo<float3>(sceneBVH[bvh_idx].vertices[i2].position);
      auto n0 = math::castTo<float3>(sceneBVH[bvh_idx].vertices[i0].normal);
      auto n1 = math::castTo<float3>(sceneBVH[bvh_idx].vertices[i1].normal);
      auto n2 = math::castTo<float3>(sceneBVH[bvh_idx].vertices[i2].normal);

      auto ab = v1 - v0;
      auto bc = v2 - v1;
      auto cross_ab_bc = math::cross(ab, bc);
      auto area = math::length(cross_ab_bc);

      auto ABx = kAB * math::distance(v0, v1);
      auto BCx = kBC * math::distance(v1, v2);
      auto CAx = kCA * math::distance(v2, v0);

      n0 *= BCx / area;
      n1 *= CAx / area;
      n2 *= ABx / area;

      hit.normal = math::normalize(n0 + n1 + n2);
      // return n;

      // n = math::normalize(math::castTo<float3>(kBC * n0 + kCA * n1 + kAB * n2));

      // hit.normal = math::dot(hit.normal, idRay.dir) < 0 ? hit.normal : hit.normal * -1;
      hit.k_d = fluidMemory.bvhColor;
      hit.k_s = float3{1.f, 1.f, 1.f};
      hit.material = fluidMemory.bvhMaterial;
      hit.emission = f3_0;
      hit.rough = fluidMemory.IOR;
      hit.hit = true;
      bvhhit = true;
    }
  }
  return bvhhit;
}

using rng = curandState;
__device__ deviceInline float sphericalTheta(const float3 &Wl) { return acosf(math::clamp(Wl.y, -1.f, 1.f)); }
__device__ deviceInline float sphericalPhi(const float3 &Wl) {
  float p = atan2f(Wl.z, Wl.x);
  return (p < 0.f) ? p + 2.f * CUDART_PI_F : p;
}
__device__ deviceInline float cosTheta(const float3 &Ws) { return Ws.z; }
__device__ deviceInline float absCosTheta(const float3 &Ws) { return fabsf(cosTheta(Ws)); }
__device__ deviceInline bool sameHemisphere(const float3 &Ww1, const float3 &Ww2) { return Ww1.z * Ww2.z > 0.0f; }
__device__ deviceInline float2 concentric_diskSample(const float2 &U) {
  float r, theta;
  float sx = 2 * U.x - 1;
  float sy = 2 * U.y - 1;

  if (sx == 0.0 && sy == 0.0) {
    return float2{0.f, .0f};
  }

  if (sx >= -sy) {
    if (sx > sy) {
      r = sx;
      if (sy > 0.0)
        theta = sy / r;
      else
        theta = 8.0f + sy / r;
    } else {
      r = sy;
      theta = 2.0f - sx / r;
    }
  } else {
    if (sx <= sy) {
      r = -sx;
      theta = 4.0f - sy / r;
    } else {
      r = -sy;
      theta = 6.0f + sx / r;
    }
  }

  theta *= CUDART_PI_F / 4.f;

  return float2{r * cosf(theta), r * sinf(theta)};
}
__device__ deviceInline float3 cosineWeighted_hemisphere(const float2 &U) {
  const float2 ret = concentric_diskSample(U);
  return float3{ret.x, ret.y, sqrtf(math::max(0.f, 1.f - ret.x * ret.x - ret.y * ret.y))};
}
__device__ deviceInline float3 sphericalDirection(const float &SinTheta, const float &cosTheta, const float &Phi) {
  return float3{SinTheta * cosf(Phi), SinTheta * sinf(Phi), cosTheta};
}
__device__ deviceInline float3 uniform_sampleSphere(const float2 &U) {
  float z = 1.f - 2.f * U.x;
  float r = sqrtf(math::max(0.f, 1.f - z * z));
  float phi = 2.f * CUDART_PI_F * U.y;
  float x = r * cosf(phi);
  float y = r * sinf(phi);
  return float3{x, y, z};
}
__device__ deviceInline float PowerHeuristic(int nf, float fPdf, int ng, float gPdf) {
  float f = nf * fPdf, g = ng * gPdf;
  return (f * f) / (f * f + g * g);
}
enum struct Shader { BRDF, Lambert, Microfacet };
struct ShadingInformation {
  Shader type; ///< Type of the shader

  float3 normal;       ///< Surface normal for shading
  float3 position;     ///< World space coordinates of the shaded point
  float3 rayDirection; ///< Direction of viewer ray (towards viewer)

  float3 k_d;            ///< Constant term for diffuse shading
  float3 k_s;            ///< Constant term for specular shading
  float IOR = 4.5f;      ///< Index of refraction
  float roughness = 1.f; ///< Surface roughness
};
struct Shading {
  float3 color{0.f, 0.f, 0.f}; ///< Color value returned by the shader
  float pdf{0.f};              ///< PDF of the sampled ray

  ////////////////////////////////////////////////////////////////////////////////////////////////////
  /// Addition assignment operator, only functionality required for this POD
  ///
  /// @param	rhs	The value to add
  ///
  /// @return	The result of the operation.
  ////////////////////////////////////////////////////////////////////////////////////////////////////

  __device__ Shading &operator+=(const Shading &rhs) {
    color = color + rhs.color;
    pdf += rhs.pdf;
    return *this;
  }
};
__device__ __inline__ auto lambertian(const float3 &Wo, const float3 &Wi, const ShadingInformation &l_info) {
  return Shading{l_info.k_d / CUDART_PI_F, sameHemisphere(Wo, Wi) ? absCosTheta(Wi) / CUDART_PI_F : 0.0f};
}
__device__ __inline__ auto fresnelDielectric(float cosi, float cost, const float3 &etai, const float3 &etat) {
  float3 Rparl = ((etat * cosi) - (etai * cost)) / ((etat * cosi) + (etai * cost));
  float3 Rperp = ((etai * cosi) - (etat * cost)) / ((etai * cosi) + (etat * cost));
  return (Rparl * Rparl + Rperp * Rperp) / 2.f;
}
__device__ __inline__ float3 fresnelEvaluate(float cosi, const ShadingInformation &l_info) {
  cosi = math::clamp(cosi, -1.0f, 1.0f);
  float ei = 1.f, et = l_info.IOR;
  // swap refractive indices
  if (cosi > 0.0f) {
    auto temp = et;
    et = ei;
    et = temp;
  }
  // Snell's law
  float sint = ei / et * sqrtf(math::max(0.f, 1.f - cosi * cosi));
  if (sint >= 1.0f) { // total internal reflection
    return float3{1.f, 1.f, 1.0f};
  } else {
    float cost = sqrtf(math::max(0.f, 1.0f - sint * sint));
    return fresnelDielectric(fabsf(cosi), cost, float3{ei, ei, ei}, float3{et, et, et});
  }
}
__device__ __inline__ float blinnPdf(const float3 &Wo, const float3 &Wi, const ShadingInformation &l_info) {
  float3 wh = math::normalize(Wo + Wi);

  float costheta = absCosTheta(wh);
  float blinn_pdf =
      ((l_info.roughness + 1.f) * powf(costheta, l_info.roughness)) / (2.f * CUDART_PI_F * 4.f * math::dot(Wo, wh));

  if (math::dot(Wo, wh) <= 0.0f)
    blinn_pdf = 0.0f;

  return blinn_pdf;
}
__device__ __inline__ float blinnD(const float3 &wh, const ShadingInformation &l_info) {
  float costhetah = absCosTheta(wh);
  return (l_info.roughness + 2) / (2.f * CUDART_PI_F) * powf(costhetah, l_info.roughness);
}

__device__ __inline__ float microFacetG(const float3 &wo, const float3 &wi, const float3 &wh,
                                        const ShadingInformation &l_info) {
  float NdotWh = absCosTheta(wh);
  float NdotWo = absCosTheta(wo);
  float NdotWi = absCosTheta(wi);
  float WOdotWh = fabsf(math::dot(wo, wh));

  return math::min(1.f, math::min((2.f * NdotWh * NdotWo / WOdotWh), (2.f * NdotWh * NdotWi / WOdotWh)));
}
__device__ __inline__ Shading microFacet(const float3 &wo, const float3 &wi, const ShadingInformation &l_info) {
  if (!sameHemisphere(wo, wi))
    return Shading{{0.f, 0.f, 0.f}, 0.f};
  float cosThetaO = absCosTheta(wo);
  float cosThetaI = absCosTheta(wi);

  if (cosThetaI == 0.f || cosThetaO == 0.f)
    return Shading{{0.f, 0.f, 0.f}, 0.f};

  float3 wh = wi + wo;

  if (wh.x == 0. && wh.y == 0. && wh.z == 0.)
    return Shading{{0.f, 0.f, 0.f}, 0.f};

  wh = math::normalize(wh);
  float cosThetaH = math::dot(wi, wh);

  float3 F = fresnelEvaluate(cosThetaH, l_info);

  return Shading{l_info.k_s * blinnD(wh, l_info) * microFacetG(wo, wi, wh, l_info) * F / (4.f * cosThetaI * cosThetaO),
                 blinnPdf(wo, wi, l_info)};
}
__device__ __inline__ Shading isotropicPhase(const float3 &Wo, const float3 &Wi, const ShadingInformation &l_info) {
  return Shading{l_info.k_d / CUDART_PI_F, 1.f / (4.0f * CUDART_PI_F)};
}
__device__ __inline__ float3 LocalToWorld(const float3 &W, const ShadingInformation &l_info) {
  float3 m_Nu = (math::normalize(math::cross(l_info.normal, l_info.rayDirection)));
  float3 m_Nv = (math::normalize(math::cross(l_info.normal, m_Nu)));

  return float3{m_Nu.x * W.x + m_Nv.x * W.y + l_info.normal.x * W.z,
                m_Nu.y * W.x + m_Nv.y * W.y + l_info.normal.y * W.z,
                m_Nu.z * W.x + m_Nv.z * W.y + l_info.normal.z * W.z};
}
__device__ __inline__ float3 WorldToLocal(const float3 &W, const ShadingInformation &l_info) {
  float3 m_Nu = (math::normalize(math::cross(l_info.normal, l_info.rayDirection)));
  float3 m_Nv = (math::normalize(math::cross(l_info.normal, m_Nu)));

  return float3{math::dot(W, m_Nu), math::dot(W, m_Nv), math::dot(W, l_info.normal)};
}
__device__ __inline__ Shading BRDF(const float3 &Wo, const float3 &Wi, const ShadingInformation &l_info) {
  const float3 Wol = WorldToLocal(Wo, l_info);
  const float3 Wil = WorldToLocal(Wi, l_info);

  Shading R{{0.f, 0.f, 0.f}, 0.f};
  auto lam = lambertian(Wol, Wil, l_info);
  auto mic = microFacet(Wol, Wil, l_info);
  // lam.color = math::max(lam.color, float3{0.f, 0.f, 0.f});
  // lam.pdf = math::max(lam.pdf, 0.f);
  // mic.color = math::max(mic.color, float3{0.f, 0.f, 0.f});
  // mic.pdf = math::max(mic.pdf, 0.f);
  if (l_info.type == Shader::BRDF || l_info.type == Shader::Lambert)
    R += lam;
  if (l_info.type == Shader::BRDF || l_info.type == Shader::Microfacet)
    R += mic;
  // R.color = l_info.k_d / CUDART_PI_F;

  R.color = R.color * math::clamp(math::dot(Wi, l_info.normal), 0.f, 1.f);

  return R;
}
__device__ __inline__ Shading shade(const ShadingInformation &l_info, float3 Wi) {
  return BRDF(l_info.rayDirection, Wi, l_info);
}

__device__ __inline__ float4 sampleBRDF(const ShadingInformation &l_info, rng *RNG) {
  if (l_info.type == Shader::BRDF || l_info.type == Shader::Lambert) {
    const float3 Wol = WorldToLocal(l_info.rayDirection, l_info);
    float3 Wi = cosineWeighted_hemisphere(float2{curand_uniform(RNG), curand_uniform(RNG)});
    if (Wol.z < 0.0f)
      Wi.z *= -1.0f;
    auto f3 = LocalToWorld(Wi, l_info);
    auto pdf = math::dot3(f3, l_info.normal) / CUDART_PI_F;

    return float4{f3.x, f3.y, f3.z, pdf};
  } else {
    const float3 Wol = WorldToLocal(l_info.rayDirection, l_info);
    float3 Wi = cosineWeighted_hemisphere(float2{curand_uniform(RNG), curand_uniform(RNG)});
    if (Wol.z < 0.0f)
      Wi.z *= -1.0f;
    auto f3 = LocalToWorld(Wi, l_info);
    auto pdf = math::dot3(f3, l_info.normal) / CUDART_PI_F;

    return float4{f3.x, f3.y, f3.z, pdf};
  }
  //} else {
  //  float costheta = powf(curand_uniform(RNG), 1.f / (l_info.roughness + 1));
  //  float sintheta = sqrtf(math::max(0.f, 1.f - costheta * costheta));
  //  float phi = curand_uniform(RNG) * 2.f * CUDART_PI_F;

  //  float3 wh = sphericalDirection(sintheta, costheta, phi);

  //  if (!sameHemisphere(l_info.rayDirection, wh))
  //    wh = -wh;

  //  return -l_info.rayDirection + 2.f * math::dot(l_info.rayDirection, wh) * wh;
  //}
}

// Sphere Declarations
class Sphere {
  Matrix4x4 WorldToObject, ObjectToWorld;
  float3 emissivity, color;
  Refl_t material;
  float roughness;

public:
  // Sphere Public Methods
  deviceInline Sphere(const Matrix4x4 ObjectToWorld, const Matrix4x4 WorldToObject, float3 emis, float3 col,
                      float rough, Refl_t mat, float radius, float zMin, float zMax, float phiMax)
      : ObjectToWorld(ObjectToWorld), WorldToObject(WorldToObject), radius(radius), roughness(rough),
        zMin(math::clamp(std::min(zMin, zMax), -radius, radius)),
        zMax(math::clamp(std::max(zMin, zMax), -radius, radius)),
        thetaMin(acosf(math::clamp(std::min(zMin, zMax) / radius, -1, 1))),
        thetaMax(acosf(math::clamp(std::max(zMin, zMax) / radius, -1, 1))),
        phiMax(degToRad(math::clamp(phiMax, 0, 360))), emissivity(emis), color(col), material(mat) {}
  deviceInline AABB ObjectBound() const { return AABB{float3{-radius, -radius, zMin}, float3{radius, radius, zMax}}; }
  deviceInline bool Intersect(const Ray &idRay, hitInformation &isect) const {
    float phi;
    float3 pHit;
    // Transform _Ray_ to object space
    float3 oErr{0.f, 0.f, 0.f}, dErr{0.f, 0.f, 0.f};
    Ray ray{pToWorld(WorldToObject, idRay.orig), vToWorld(WorldToObject, idRay.dir)};
    // Compute quadratic sphere coefficients

    // Initialize _EFloat_ ray coordinate values
    pbrt::EFloat ox(ray.orig.x, oErr.x), oy(ray.orig.y, oErr.y), oz(ray.orig.z, oErr.z);
    pbrt::EFloat dx(ray.dir.x, dErr.x), dy(ray.dir.y, dErr.y), dz(ray.dir.z, dErr.z);
    pbrt::EFloat a = dx * dx + dy * dy + dz * dz;
    pbrt::EFloat b = pbrt::EFloat(2.f) * (dx * ox + dy * oy + dz * oz);
    pbrt::EFloat c = ox * ox + oy * oy + oz * oz - pbrt::EFloat(radius) * pbrt::EFloat(radius);

    // Solve quadratic equation for _t_ values
    pbrt::EFloat t0, t1;
    if (!Quadratic(a, b, c, &t0, &t1))
      return false;

    // Check quadric shape _t0_ and _t1_ for nearest intersection
    if (t0.UpperBound() > isect.depth || t1.LowerBound() <= 0)
      return false;
    pbrt::EFloat tShapeHit = t0;
    if (tShapeHit.LowerBound() <= 0) {
      tShapeHit = t1;
      if (tShapeHit.UpperBound() > isect.depth)
        return false;
    }

    // Compute sphere hit position and $\phi$
    pHit = ray((float)tShapeHit);

    // Refine sphere intersection point
    pHit *= radius / math::distance(pHit, float3{0, 0, 0});
    if (pHit.x == 0 && pHit.y == 0)
      pHit.x = 1e-5f * radius;
    phi = std::atan2(pHit.y, pHit.x);
    if (phi < 0)
      phi += 2 * CUDART_PI_F;

    // Test sphere intersection against clipping parameters
    if ((zMin > -radius && pHit.z < zMin) || (zMax < radius && pHit.z > zMax) || phi > phiMax) {
      if (tShapeHit == t1)
        return false;
      if (t1.UpperBound() > isect.depth)
        return false;
      tShapeHit = t1;
      // Compute sphere hit position and $\phi$
      pHit = ray((float)tShapeHit);

      // Refine sphere intersection point
      pHit *= radius / math::distance(pHit, float3{0, 0, 0});
      if (pHit.x == 0 && pHit.y == 0)
        pHit.x = 1e-5f * radius;
      phi = std::atan2(pHit.y, pHit.x);
      if (phi < 0)
        phi += 2 * CUDART_PI_F;
      if ((zMin > -radius && pHit.z < zMin) || (zMax < radius && pHit.z > zMax) || phi > phiMax)
        return false;
    }

    // Find parametric representation of sphere hit
    float u = phi / phiMax;
    float theta = acosf(math::clamp(pHit.z / radius, -1.f, 1.f));
    float v = (theta - thetaMin) / (thetaMax - thetaMin);

    // Compute sphere $\dpdu$ and $\dpdv$
    float zRadius = sqrtf(pHit.x * pHit.x + pHit.y * pHit.y);
    float invZRadius = 1 / zRadius;
    float cosPhi = pHit.x * invZRadius;
    float sinPhi = pHit.y * invZRadius;
    float3 dpdu{-phiMax * pHit.y, phiMax * pHit.x, 0};
    float3 dpdv = (thetaMax - thetaMin) * float3{pHit.z * cosPhi, pHit.z * sinPhi, -radius * sinf(theta)};

    // Compute sphere $\dndu$ and $\dndv$
    float3 d2Pduu = -phiMax * phiMax * float3{pHit.x, pHit.y, 0};
    float3 d2Pduv = (thetaMax - thetaMin) * pHit.z * phiMax * float3{-sinPhi, cosPhi, 0.f};
    float3 d2Pdvv = -(thetaMax - thetaMin) * (thetaMax - thetaMin) * float3{pHit.x, pHit.y, pHit.z};

    // Compute coefficients for fundamental forms
    float E = math::dot3(dpdu, dpdu);
    float F = math::dot3(dpdu, dpdv);
    float G = math::dot3(dpdv, dpdv);
    float3 N = math::normalize(math::cross(dpdu, dpdv));
    float e = math::dot3(N, d2Pduu);
    float f = math::dot3(N, d2Pduv);
    float g = math::dot3(N, d2Pdvv);

    // Compute $\dndu$ and $\dndv$ from fundamental form coefficients
    float invEGF2 = 1 / (E * G - F * F);
    float3 dndu = float3{(f * F - e * G) * invEGF2 * dpdu + (e * F - f * E) * invEGF2 * dpdv};
    float3 dndv = float3{(g * F - f * G) * invEGF2 * dpdu + (f * F - g * E) * invEGF2 * dpdv};

    // Compute error bounds for sphere intersection
    float3 pError = pbrt::gamma(5) * math::abs((float3)pHit);

    // Initialize _SurfaceInteraction_ from parametric information
    isect.hit = true;
    isect.k_d = color;
    isect.k_s = float3{1.f, 1.f, 1.f};
    isect.rough = roughness;
    isect.depth = (float)tShapeHit;
    isect.normal = N;
    isect.emission = emissivity;
    isect.material = material;
    isect.rayDirection = idRay.dir;

    //*isect = (*ObjectToWorld)(
    //    SurfaceInteraction(pHit, pError, float2(u, v), -ray.d, dpdu, dpdv, dndu, dndv, ray.time, this));

    //// Update _tHit_ for quadric intersection
    //*tHit = (float)tShapeHit;
    return true;
  }
  deviceInline float Area() const { return phiMax * radius * (zMax - zMin); }
  deviceInline Interaction Sample(const float2 &u, float *pdf) const {
    float3 pObj = float3{0, 0, 0} + radius * uniform_sampleSphere(u);
    Interaction it;
    it.n = math::castTo<float3>(math::normalize3(ObjectToWorld * float4{pObj.x, pObj.y, pObj.z, 0.f}));
    // Reproject _pObj_ to sphere surface and compute _pObjError_
    pObj *= radius / math::distance(pObj, float3{0, 0, 0});
    it.p = math::castTo<float3>(ObjectToWorld * float4{pObj.x, pObj.y, pObj.z});
    *pdf = 1 / Area();
    return it;
  }
  deviceInline Interaction Sample(const Interaction &ref, const float2 &u, float *pdf) const {
    float3 pCenter = math::castTo<float3>((ObjectToWorld)*float4{0.f, 0.f, 0.f, 1.f});

    // Sample uniformly on sphere if $\pt{}$ is inside it
    float3 pOrigin = pbrt::OffsetRayOrigin(ref.p, ref.n, pCenter - ref.p);
    if (math::distance(pOrigin, pCenter) <= radius * radius) {
      Interaction intr = Sample(u, pdf);
      float3 wi = intr.p - ref.p;
      if (math::sqlength3(wi) == 0)
        *pdf = 0;
      else {
        // Convert from area measure returned by Sample() call above to
        // solid angle measure.
        wi = math::normalize(wi);
        *pdf *= math::sqdistance(ref.p, intr.p) / fabsf(math::dot(intr.n, -wi));
      }
      // if ((*pdf) == FLT_INF)
      //  *pdf = 0.f;
      //*pdf = 0.f;
      return intr;
    }

    // Sample sphere uniformly inside subtended cone

    // Compute coordinate system for sphere sampling
    float dc = math::distance(ref.p, pCenter);
    float invDc = 1.f / dc;
    float3 wc = (pCenter - ref.p) * invDc;
    float3 wcX, wcY;
    CoordinateSystem(wc, &wcX, &wcY);

    // Compute $\theta$ and $\phi$ values for sample in cone
    float sinThetaMax = radius * invDc;
    float sinThetaMax2 = sinThetaMax * sinThetaMax;
    float invSinThetaMax = 1 / sinThetaMax;
    float cosThetaMax = sqrtf(std::max((float)0.f, 1.f - sinThetaMax2));

    float cosTheta = (cosThetaMax - 1) * u.x + 1;
    float sinTheta2 = 1 - cosTheta * cosTheta;

    if (sinThetaMax2 < 0.00068523f /* sin^2(1.5 deg) */) {
      /* Fall back to a Taylor series expansion for small angles, where
         the standard approach suffers from severe cancellation errors */
      sinTheta2 = sinThetaMax2 * u.x;
      cosTheta = sqrtf(1 - sinTheta2);
    }

    // Compute angle $\alpha$ from center of sphere to sampled point on surface
    float cosAlpha = sinTheta2 * invSinThetaMax +
                     cosTheta * sqrtf(std::max((float)0.f, 1.f - sinTheta2 * invSinThetaMax * invSinThetaMax));
    float sinAlpha = sqrtf(std::max((float)0.f, 1.f - cosAlpha * cosAlpha));
    float phi = u.y * 2 * CUDART_PI_F;

    // Compute surface normal and sampled point on sphere
    float3 nWorld = SphericalDirection(sinAlpha, cosAlpha, phi, -wcX, -wcY, -wc);
    float3 pWorld = pCenter + radius * float3{nWorld.x, nWorld.y, nWorld.z};

    // Return _Interaction_ for sampled point on sphere
    Interaction it;
    it.p = pWorld;
    it.n = nWorld;

    // Uniform cone PDF.
    *pdf = 1.f / (2.f * CUDART_PI_F * (1.f - cosThetaMax));

    return it;
  }
  deviceInline float shapePdf(const Interaction &ref, const float3 &wi) const {
    // Intersect sample ray with area light geometry
    Ray ray{ref.p + ref.n * 1e-4f, wi};
    float tHit;
    hitInformation isectLight;
    // Ignore any alpha textures used for trimming the shape when performing
    // this intersection. Hack for the "San Miguel" scene, where this is used
    // to make an invisible area light.
    if (!Intersect(ray, isectLight))
      return 0;

    // Convert light sample weight to solid angle measure
    float pdf = math::sqdistance(ref.p, ray(isectLight.depth)) / (fabsf(math::dot(isectLight.normal, -wi)) * Area());
    //    if (std::isinf(pdf))
    //    pdf = 0.f;
    return pdf;
  }
  deviceInline float Pdf(const Interaction &ref, const float3 &wi) const {
    float3 pCenter = pToWorld(ObjectToWorld, f3_0);
    // Return uniform PDF if point is inside sphere
    float3 pOrigin = pbrt::OffsetRayOrigin(ref.p, ref.n, pCenter - ref.p);
    if (math::sqdistance(pOrigin, pCenter) <= radius * radius)
      return shapePdf(ref, wi);

    // Compute general sphere PDF
    float sinThetaMax2 = radius * radius / math::sqdistance(ref.p, pCenter);
    float cosThetaMax = sqrtf(std::max((float)0, 1 - sinThetaMax2));
    return UniformConePdf(cosThetaMax);
  }
  deviceInline float SolidAngle(const float3 &p, int nSamples) const {
    float3 pCenter = pToWorld(ObjectToWorld, f3_0);
    if (math::sqdistance(p, pCenter) <= radius * radius)
      return 4 * CUDART_PI_F;
    float sinTheta2 = radius * radius / math::sqdistance(p, pCenter);
    float cosTheta = sqrtf(std::max((float)0, 1 - sinTheta2));
    return (2 * CUDART_PI_F * (1 - cosTheta));
  }

private:
  // Sphere Private Data
  const float radius;
  const float zMin, zMax;
  const float thetaMin, thetaMax, phiMax;
};

class PointLight {
  Matrix4x4 m1, m1_1;
  Sphere S;
  float3 ctr;

public:
  deviceInline PointLight(float3 center, float3 emi, float epsilonSize = 1.f)
      : m1(Matrix4x4::fromTranspose(center)), m1_1(m1.inverse()), ctr(center),
        S(Sphere(m1, m1_1, emi, float3{1.f, 1.f, 1.f}, 0.f, Refl_t::Lambertian, epsilonSize, -epsilonSize, epsilonSize,
                 360.f)) {}
  deviceInline bool Intersect(const Ray &idRay, hitInformation &isect) { return S.Intersect(idRay, isect); }
  deviceInline Interaction Sample(const Interaction &ref, float *pdf) const {
    *pdf = 1.f;
    Interaction itr;
    itr.p = ctr;
    itr.n = math::normalize3(ref.p - ctr);
    return itr;
  }
};

class Box {
  float3 min, max, emi, col;
  Refl_t refl;
  float roughness;

public:
  __device__ Box(float3 c, float3 e, float3 co, float rough, Refl_t mat, float3 em)
      : emi(em), col(co), refl(mat), min(c - e * 0.5f), max(c + e * 0.5f), roughness(rough) {}
  deviceInline bool Intersect(const Ray &worldRay, hitInformation &isect) {
    float tmin = ((worldRay.dir.x < 0.f ? max.x : min.x) - worldRay.orig.x) / worldRay.dir.x;
    float tmax = ((worldRay.dir.x < 0.f ? min.x : max.x) - worldRay.orig.x) / worldRay.dir.x;
    float tymin = ((worldRay.dir.y < 0.f ? max.y : min.y) - worldRay.orig.y) / worldRay.dir.y;
    float tymax = ((worldRay.dir.y < 0.f ? min.y : max.y) - worldRay.orig.y) / worldRay.dir.y;

    if ((tmin > tymax) || (tymin > tmax))
      return false;
    if (tymin > tmin)
      tmin = tymin;
    if (tymax < tmax)
      tmax = tymax;

    float tzmin = ((worldRay.dir.z < 0.f ? max.z : min.z) - worldRay.orig.z) / worldRay.dir.z;
    float tzmax = ((worldRay.dir.z < 0.f ? min.z : max.z) - worldRay.orig.z) / worldRay.dir.z;

    if ((tmin > tzmax) || (tzmin > tmax))
      return false;
    if (tzmin > tmin)
      tmin = tzmin;
    if (tzmax < tmax)
      tmax = tzmax;
    // return (tmin < 0.f && tmax > 0.f) || (tmin > 0.f && tmax > 0.f) ? (tmin < 0.f ? tmax : tmin) : 0.f;

    bool hit = (tmin < 0.f && tmax > 0.f) || (tmin > 0.f && tmax > 0.f);
    if (!hit)
      return false;

    float t = (tmin < 0.f ? tmax : tmin);
    if (t > isect.depth)
      return false;
    isect.k_d = col;
    isect.k_s = float3{1.f, 1.f, 1.f};
    isect.rough = roughness;
    isect.depth = t;
    isect.emission = emi;
    isect.hit = true;
    isect.material = refl;
    isect.rayDirection = worldRay.dir;
    isect.normal = normal(worldRay(t));
    return true;
  }
  __device__ float3 normal(const float3 &hit) const {
    auto c = (min + max) * 0.5f;
    auto p = hit - c;
    auto d = (min - max) * 0.5f;
    auto bias = 1.0001f;
    auto result = math::normalize(float3{static_cast<float>(static_cast<int32_t>(p.x / fabsf(d.x) * bias)),
                                         static_cast<float>(static_cast<int32_t>(p.y / abs(d.y) * bias)),
                                         static_cast<float>(static_cast<int32_t>(p.z / fabsf(d.z) * bias))});
    return result;
  }
};

class Plane {
  float3 n, p0;
  Refl_t material;
  float3 color, emission;
  float roughness;

public:
  deviceInline Plane(float3 normal, float3 p, float3 col, float rough, Refl_t mat, float3 emi)
      : n(math::normalize3(normal)), p0(p), color(col), emission(emi), material(mat), roughness(rough) {}

  deviceInline bool Intersect(const Ray &worldRay, hitInformation &isect) {
    // assuming vectors are all normalized
    float denom = math::dot3(n, worldRay.dir);
    if (fabsf(denom) < 1e-6f)
      return false;
    float3 p0l0 = p0 - worldRay.orig;
    float t = math::dot3(p0l0, n) / denom;
    if (t < 0 || t >= isect.depth)
      return false;
    isect.k_d = color;
    isect.k_s = float3{1.f, 1.f, 1.f};
    isect.rough = roughness;
    isect.depth = t;
    isect.emission = emission;
    isect.hit = true;
    isect.material = material;
    isect.rayDirection = worldRay.dir;
    isect.normal = -n;
    return true;
  }
};
class TexturedPlane {
  float3 n, p0;
  Refl_t material;
  float3 color, emission;
  float2 period;
  float roughness;

public:
  deviceInline TexturedPlane(float3 normal, float3 p, float2 pe, float3 col, float rough, Refl_t mat, float3 emi)
      : n(math::normalize3(normal)), p0(p), color(col), emission(emi), material(mat), period(pe), roughness(rough) {}

  deviceInline bool Intersect(const Ray &worldRay, hitInformation &isect) {
    // assuming vectors are all normalized
    float denom = math::dot3(n, worldRay.dir);
    if (fabsf(denom) < 1e-6f)
      return false;
    float3 p0l0 = p0 - worldRay.orig;
    float t = math::dot3(p0l0, n) / denom;
    if (t < 0 || t >= isect.depth)
      return false;
    float3 hitPosition = worldRay(t);
    float2 planeHit;
    float3 nabs{fabsf(n.x), fabsf(n.y), fabsf(n.z)};
    if (nabs.x >= nabs.y && nabs.x >= nabs.z)
      planeHit = float2{hitPosition.y, hitPosition.z};
    else if (nabs.y >= nabs.x && nabs.y >= nabs.z)
      planeHit = float2{hitPosition.x, hitPosition.z};
    else if (nabs.z >= nabs.y && nabs.z >= nabs.x)
      planeHit = float2{hitPosition.x, hitPosition.y};

    float u = fmodf(planeHit.x, period.x) / period.x;
    float v = fmodf(planeHit.y, period.y) / period.y;

    u = planeHit.x < 0.f ? 1.f + u : u;
    v = planeHit.y < 0.f ? 1.f + v : v;

    float4 rgba = tex2D<float4>(fluidMemory.uvmap, u, v);

    isect.k_d = float3{rgba.x, rgba.y, rgba.z};
    isect.k_s = float3{1.f, 1.f, 1.f};
    isect.rough = roughness;
    isect.depth = t;
    isect.emission = emission;
    isect.hit = true;
    isect.material = material;
    isect.rayDirection = worldRay.dir;
    isect.normal = -n;
    return true;
  }
};

struct LightSample {
  float3 Li; ///< Lightcolor returned from this sample
  bool Hit;  ///< True if the provided ray intersected the light source
  float pdf; ///< PDF of the ray for MIS
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// Describes a light source, either environment or rectangular, in the scene.
/// Provides functionality to check for intersection of rays and generating
/// sample rays for MIS Sampling
////////////////////////////////////////////////////////////////////////////////////////////////////

class LightSource {
public:
  float width = 1.f;  ///< Width of the light source in world space units
  float height = 1.f; ///< Height of the light source in world space units

  float radius = 100.f;             ///< Radius of the light source if it is an environment
                                    ///< light else distance from center.
  float3 position{1.f, 1.f, 1.f};   ///< Position of the light source in world space units
  float3 N{1.f, 0.f, 0.f};          ///< Normal of the light source if it is rectangular
  float3 U{0.f, 1.f, 0.f};          ///< Direction of the width component
  float3 V{0.f, 0.f, 1.f};          ///< Direction of the height component
  float area = 1.f;                 ///< Area of the light source
  float3 lightColor{1.f, 1.f, 1.f}; ///< Color of the light source (uniform across the area)
  int type = 0;                     ///< Type of the light source, 0 = rectangular, 1 = environment

  constexpr __device__ LightSource() {}

  ////////////////////////////////////////////////////////////////////////////////////////////////////
  /// Generates a random sampled ray for MIS Sampling that hits the light
  /// source.
  ///
  /// @param 		   	P 	Position of the ray intersection
  /// @param [out]	Rl	Generated ray from the light source to the
  /// intersection
  /// @param [in,out]	LS	Pseudo random number generator used for random
  /// sampling
  ///
  /// @return	Returns the sampled Light for the Ray, always hits by defintion.
  ////////////////////////////////////////////////////////////////////////////////////////////////////

  __device__ __inline__ LightSample SampleL(const float3 &P, Ray &Rl, float2 LS) const {
    if (type == 0) {
      // randomly samples a position on the light source using a local
      // coordinate system.
      Rl.orig = position + ((-0.5f + LS.x) * width * U) + ((-0.5f + LS.y) * height * V);
      Rl.dir = math::normalize(P - Rl.orig);

      return LightSample{
          math::dot(Rl.dir, N) > 0.0f ? lightColor / area : float3{0.0f, 0.f, 0.f}, // Black if the ray and normal
                                                                                    // point in the same direction
          true,
          fabsf(math::dot(Rl.dir, N)) > 0.0f ? math::sqdistance(P, Rl.orig) / (fabsf(math::dot(Rl.dir, N)) * area)
                                             : 0.0f};
    } else {
      // For environmental light sources all directions are equally valid
      Rl.orig = radius * uniform_sampleSphere(LS);
      Rl.dir = math::normalize(P - Rl.orig);

      return LightSample{lightColor, true, powf(radius, 2.0f) / area};
    }
  }

  ////////////////////////////////////////////////////////////////////////////////////////////////////
  /// Calcualtes the intersection of a Ray with a LightSource
  ///
  /// @param 		   	R	The candidate ay used for intersection
  /// testing
  /// @param [out]	T	The distance from the ray origin to the light
  /// source
  ///
  /// @return	Returns the sampled light of the ray against this light source
  ////////////////////////////////////////////////////////////////////////////////////////////////////

  __device__ __inline__ bool Intersect(const Ray &R, hitInformation &isect) const {
    if (type == 0) {
      const float dotN = math::dot(R.dir, N);

      if (dotN >= 0.0f)
        return false;

      float T = (-radius - math::dot(R.orig, N)) / dotN;

      if (T < 0.f || T > isect.depth)
        return false;

      const float3 Pl = R(T);
      const float3 Wl = Pl - position;
      const float2 UV = float2{math::dot(Wl, U), math::dot(Wl, V)};

      if (UV.x > width * 0.5f || UV.x < -width * 0.5f || UV.y > height * 0.5f || UV.y < -height * 0.5f)
        return false;

      isect.k_d = float3{1.f, 1.f, 1.f};
      isect.k_s = float3{1.f, 1.f, 1.f};
      isect.rough = 1.f;
      isect.depth = T;
      isect.emission = lightColor;
      isect.hit = true;
      isect.material = Refl_t::Lambertian;
      isect.rayDirection = R.dir;
      isect.normal = N;

      return true;
    } else {
      float T = radius;
      if (T < 0.f || T > isect.depth)
        return false;
      isect.k_d = float3{1.f, 1.f, 1.f};
      isect.k_s = float3{1.f, 1.f, 1.f};
      isect.rough = 1.f;
      isect.depth = T;
      isect.emission = lightColor;
      isect.hit = true;
      isect.material = Refl_t::Lambertian;
      isect.rayDirection = R.dir;
      isect.normal = N;

      return true; /*
       return LightSample{lightColor, true, powf(radius, 2.0f) / area};*/
    }
  }
};

__global__ void shadeAndBounceRays(int32_t seed, int32_t numBVHs, vrtx::gpuBVH *sceneBVH) {
  int32_t tIdx = getThreadIdx();
  if (tIdx > cNumRays)
    return;
  bool firstPass = true;
  while (true) {
    hitInformation hit;
    bool hitFluidAABB = false;
    auto idRay = cRaysDepth[tIdx];
    if (idRay.bounces >= fluidMemory.bounces)
      return;
    bool fluidHit = false;
    bool floorHit = false;
    {
      auto m1 = Matrix4x4::fromTranspose(float4{0.f, 0.f, 50.f, 0.f});
      auto m1_1 = m1.inverse();

      auto m2 = Matrix4x4::fromTranspose(float4{50.f, 0.f, 50.f, 0.f});
      auto m2_1 = m2.inverse();

      auto m3 = Matrix4x4::fromTranspose(float4{-50.f, 0.f, 50.f, 0.f});
      auto m3_1 = m3.inverse();
      /*deviceInline Sphere(const Matrix4x4 ObjectToWorld, const Matrix4x4 WorldToObject, bool reverseOrientation,
                          float radius, float zMin, float zMax, float phiMax)*/
      //  Sphere testSphere1(m1, m1_1, float3{1.f, 1.f, 1.f}, float3{0.f, 0.f, 0.f}, 0.f, Refl_t::Plastic, 5.f,
      //  -50.f, 50.f, 360.f);
      // Sphere testSphere2(m2, m2_1, f3_0, float3{1.f, 1.f, 1.f}, 0.2f, Refl_t::Plastic, 5.f,
      // -25.f, 25.f, 360.f);
      // Sphere testSphere3(m3, m3_1, f3_0, float3{0.f, 0.8f, 0.8f}, 20.f, Refl_t::Plastic, 10.f,
      // -25.f, 25.f, 360.f);
      // Box testBox(float3{0.f, 50.f, 0.f}, float3{10.f, 10.f, 10.f}, float3{0.2f, 0.2f, 0.2f}, 0.2f,
      // Refl_t::Lambertian,
      //            float3{0.f, 0.f, 0.f});
      // Box testBox2(float3{0.f, -50.f, 0.f}, float3{25.f, 25.f, 25.f}, float3{0.8f, 0.8f, 0.f}, 0.1f, Refl_t::Plastic,
      //            float3{0.f, 0.f, 0.f});
      // PointLight testPoint{float3{-50.f, 0.f, 0.f}, float3{1.f, 1.f, 1.f}, 1.f};
      Plane testPlanex(float3{1.f, 0.f, 0.f}, float3{fluidMemory.vrtxRenderDomainMin.x, 0.f, 0.f}, float3{1.f, 1.f, 1.f},
                       0.f, Refl_t::Nayar,
                       float3{1.f, 1.f, 1.f});
      Plane testPlaney(float3{-1.f, 0.f, 0.f}, float3{fluidMemory.vrtxRenderDomainMax.x, 0.f, 0.f},
                       float3{1.f, 1.f, 1.f}, 0.f, Refl_t::Nayar,
                       float3{ 1.f, 1.f, 1.f });
      Plane testPlanez(float3{0.f, 1.f, 0.f}, float3{0.f, fluidMemory.vrtxRenderDomainMin.y, 0.f},
                       float3{1.f, 1.f, 1.f}, 0.f, Refl_t::Nayar,
                       float3{1.f, 1.f, 1.f});
      Plane testPlanez2(float3{0.f, -1.f, 0.f}, float3{0.f, fluidMemory.vrtxRenderDomainMax.y, 0.f},
                        float3{1.f, 1.f, 1.f}, 0.f,
                        Refl_t::Lambertian, float3{ 1.f, 1.f, 1.f });
      //TexturedPlane testPlanez2(float3{ 0.f, -1.f, 0.f }, float3{ 0.f, fluidMemory.vrtxRenderDomainMax.y,0.f },
      //    float2{ 100.f, 100.f },
      //    float3{ 1.f, 1.f, 1.f }, 0.f, Refl_t::Lambertian, float3{ 0.f, 0.f, 0.f });

      Plane testPlane(float3{0.f, 0.f, -1.f}, float3{0.f, 0.f, fluidMemory.vrtxRenderDomainMax.z},
                      float3{1.f, 1.f, 1.f}, 0.f, Refl_t::Nayar,
                      float3{1.f, 1.f, 1.f});
      LightSource Light_a;
      float polar = CUDART_PI_F / 2.f;
      float azimuth = CUDART_PI_F / 2.f;
      Light_a.radius = 50;
      Light_a.width = 100;
      Light_a.height = 100;
      Light_a.lightColor = float3{1.f, 1.f, 1.f};
      Light_a.type = 0;

      Light_a.position.x = Light_a.radius * cosf(azimuth) * sinf(polar);
      Light_a.position.z = Light_a.radius * cosf(azimuth) * cosf(polar);
      Light_a.position.y = Light_a.radius * sinf(azimuth);
      Light_a.position += float3{0.f, 0.f, 0.f};
      Light_a.area = Light_a.width * Light_a.height;
      Light_a.N = math::normalize(float3{0.f, 0.f, 0.f} - Light_a.position);
      Light_a.U = math::normalize(math::cross(Light_a.N, float3{0.0f, 1.0f, 0.0f}));
      Light_a.V = math::normalize(math::cross(Light_a.N, Light_a.U));

      LightSource Light_b;
      Light_b.radius = 2500.f;
      Light_b.area = 4.0f * CUDART_PI_F * powf(Light_b.radius, 2.0f);
      Light_b.type = 1;
      Light_b.lightColor = float3{0.8f, 0.8f, 0.8f};

      // testSphere1.Intersect(idRay, hit);
      //  testSphere2.Intersect(idRay, hit);
      // testPoint.Intersect(idRay, hit);
      // testSphere3.Intersect(idRay, hit);
      // testBox.Intersect(idRay, hit);
      // testBox2.Intersect(idRay, hit);

      testPlane.Intersect(idRay, hit);
      testPlanex.Intersect(idRay, hit);
      testPlaney.Intersect(idRay, hit);
      testPlanez.Intersect(idRay, hit);
      testPlanez2.Intersect(idRay, hit);
      // Light_a.Intersect(idRay, hit);
      // Light_b.Intersect(idRay, hit);

      // if (hit.hit)
      //  cImage[idRay.index].color = hit.normal;
      // idRay.bounces += 100;
      // cRaysDepth[tIdx] = idRay;
      // return;
      intersectBVH(tIdx, hit, idRay, numBVHs, sceneBVH);

      hitFluidAABB = !firstPass || aabb::rayIntersectAABB(idRay, fluidMemory.min_coord, fluidMemory.max_coord).hit;
      if (firstPass)
        fluidHit = intersectFluid(tIdx, hit, idRay);
      else if (hitFluidAABB)
        return;
      firstPass = false;

      if (fluidMemory.renderFloor) {
        TexturedPlane testPlane2(float3{0.f, 0.f, 1.f}, float3{0.f, 0.f, fluidMemory.vrtxRenderDomainMin.z},
                                 float2{100.f, 100.f},
                                 float3{1.f, 1.f, 1.f}, 0.f, Refl_t::Nayar, float3{0.f, 0.f, 0.f});
        floorHit = testPlane2.Intersect(idRay, hit);
      } else {
        Plane testPlane2(float3{0.f, 0.f, 1.f}, float3{0.f, 0.f, fluidMemory.vrtxRenderDomainMin.z},
                         float3{1.f, 1.f, 1.f}, 0.f, Refl_t::Nayar,
                         float3{0.f, 0.f, 0.f});
        floorHit = testPlane2.Intersect(idRay, hit);
      }
      // hitFluidAABB = firstPass;

      // auto sphereHit = intersectSpheres(tIdx, hit, idRay);
      // auto boxHit = intersectBoxes(tIdx, hit, idRay);
    }
    // auto [intersect, color, depth, normal, material, emission] = hit;
    auto intersect = hit.hit;
    auto depth = hit.depth;
    auto normal = hit.normal;
    auto material = hit.material;
    auto emission = hit.emission;

    if (!intersect) {
      cImage[idRay.index].mask = float3{0.f, 0.f, 0.f};
      idRay =
          worldRay{idRay.orig, 1e21f, idRay.dir, 1.f, idRay.internal, (uint32_t)idRay.bounces + 100u, (uint32_t)idRay.index};
      cfluidIntersection[tIdx] = float4{0.f, 0.f, 0.f, FLT_MAX};
      cRaysDepth[tIdx] = idRay;
      return;
    }
    cImage[idRay.index].color += cImage[idRay.index].mask * emission;
    if (fluidMemory.vrtxDepth || fluidMemory.vrtxNormal) {
      if (fluidMemory.vrtxDepth)
        cImage[idRay.index].color = float3{depth * fluidMemory.vrtxDepthScale, depth * fluidMemory.vrtxDepthScale,
                                           depth * fluidMemory.vrtxDepthScale};
      if (fluidMemory.vrtxNormal)
        cImage[idRay.index].color = normal;
      idRay.bounces += 100;
      cfluidIntersection[tIdx] = float4{0.f, 0.f, 0.f, FLT_MAX};
      cRaysDepth[tIdx] = idRay;
      return;
    }
    if (math::length(emission) > 1e-2f) {
        if (idRay.bounces == 0) {
            if(fluidMemory.vrtxSurface)
                cImage[idRay.index].color = float3{ 1.f, 1.f, 1.f };
            else
                cImage[idRay.index].color = fluidMemory.vrtxFluidColor;

        }
      idRay.bounces = fluidMemory.bounces + 100;
      cfluidIntersection[tIdx] = float4{0.f, 0.f, 0.f, FLT_MAX};
      cRaysDepth[tIdx] = idRay;
      return;
    }
    if (floorHit && !fluidMemory.renderFloor) {
      if (idRay.bounces == 0)
        cImage[idRay.index].color = float3{1.f, 1.f, 1.f};
      idRay.bounces = fluidMemory.bounces + 100;
      cfluidIntersection[tIdx] = float4{0.f, 0.f, 0.f, FLT_MAX};
      cRaysDepth[tIdx] = idRay;
      return;
    }

    cfluidIntersection[tIdx] = float4{0.f, 0.f, 0.f, FLT_MAX};
    // DeBeer Absorption
    if (idRay.internal) {
      float3 k = fluidMemory.vrtxDebeer;
      float scale = fluidMemory.vrtxDebeerScale;
      cImage[idRay.index].mask *=
          float3{expf(-k.x * depth * scale), expf(-k.y * depth * scale), expf(-k.z * depth * scale)};
    }
    curandState randState;
    curand_init(seed + tIdx, 0, 0, &randState);
    hit.rayDirection = -idRay.dir;
    auto x = idRay.orig + hit.depth * idRay.dir;
    ShadingInformation l_info{
        Shader::BRDF, math::normalize(hit.normal), x, -idRay.dir, hit.k_d, hit.k_s, fluidMemory.IOR, fluidMemory.IOR};

    auto shade = [&](pbrt::BSDF &bsdfMaterial) {
      float3 Wil;
      float pdf = 1.f;
      float2 rand2{curand_uniform(&randState), curand_uniform(&randState)};
      pbrt::BxDFType flags;
      float3 bsdfValue = bsdfMaterial.Sample_f(l_info.rayDirection, &Wil, rand2, &pdf, pbrt::BSDF_ALL, &flags);
      if (pdf == pdf && pdf > 0.f && math::length3(bsdfValue) < 1e6f)
        cImage[idRay.index].mask *= bsdfValue / pdf * fabsf(math::dot(Wil, hit.normal));
      else if (pdf != 0.f)
        idRay.bounces += 100;
      if (flags & pbrt::BSDF_TRANSMISSION)
        idRay.internal = !idRay.internal;
      cRaysDepth[tIdx] = worldRay{x + math::normalize(math::castTo<float3>(Wil)) * (fluidHit ? fluidMemory.fluidBias : 0.01f),
                                  1e21f,
                                  math::normalize(math::castTo<float3>(Wil)),
                                  idRay.importance,
                                  idRay.internal, // math::dot(Wil, hit.normal) < 0.f,
                                  (uint32_t)idRay.bounces + 1u,
                                  (uint32_t)idRay.index};
    };

    if (hit.material == Refl_t::Glass || hit.material == Refl_t::Water) {
      float eta = hit.material == Refl_t::Water ? fluidMemory.IOR : fluidMemory.IOR;
      pbrt::BSDF bsdfMaterial(hit, eta);
      float3 R = float3{1.f, 1.f, 1.f};
      float3 T = float3{1.f, 1.f, 1.f};
      pbrt::FresnelSpecular fresnelMultiLobe(R, T, 1.f, eta, pbrt::TransportMode::Radiance);
      bsdfMaterial.Add(&fresnelMultiLobe);
      shade(bsdfMaterial);
    } else if (hit.material == Refl_t::RoughGlass) {
      float eta = 1.58f;
      pbrt::BSDF bsdfMaterial(hit, eta);
      float3 R = float3{1.f, 1.f, 1.f};
      float3 T = float3{1.f, 1.f, 1.f};
      float urough = pbrt::TrowbridgeReitzDistribution::RoughnessToAlpha(1e-4f);
      float vrough = pbrt::TrowbridgeReitzDistribution::RoughnessToAlpha(1e-4f);
      pbrt::TrowbridgeReitzDistribution distribV(urough, vrough);
      pbrt::MicrofacetDistribution *distrib = &distribV;
      pbrt::FresnelDielectric dielectric(1.f, eta);
      pbrt::MicrofacetReflection microFacetReflection(R, distrib, &dielectric);
      pbrt::MicrofacetTransmission microFacetTransmission(T, distrib, 1.f, eta, pbrt::TransportMode::Radiance);
      bsdfMaterial.Add(&microFacetReflection);
      bsdfMaterial.Add(&microFacetTransmission);
      shade(bsdfMaterial);
    } else if (hit.material == Refl_t::Lambertian) {
      if (fluidMemory.vrtxSurface && fluidHit)
        if (math::dot3(hit.normal, l_info.rayDirection) < 0.f)
          l_info.rayDirection *= -1.f;
      // if (math::dot3(hit.normal, l_info.rayDirection) > 0.f)
      //  l_info.rayDirection *= -1.f;
      pbrt::BSDF bsdfMaterial(hit);
      pbrt::LambertianReflection lambertian(l_info.k_d);
      bsdfMaterial.Add(&lambertian);
      shade(bsdfMaterial);
    } else if (hit.material == Refl_t::Mirror) {
      // Mirror material
      pbrt::BSDF bsdfMaterial(hit, 1.f);
      pbrt::LambertianReflection lmb(l_info.k_d);
      pbrt::FresnelDielectric fresnel(1.f, fluidMemory.vrtxDepthScale);
      float rough = pbrt::BeckmannDistribution::RoughnessToAlpha(1e-4f);
      pbrt::BeckmannDistribution distrib(rough, rough);
      pbrt::MicrofacetReflection specular(l_info.k_s, &distrib, &fresnel);
      bsdfMaterial.Add(&specular);
      // pbrt::OrenNayar orenNayar(l_info.k_d, l_info.roughness);
      // bsdfMaterial.Add(&orenNayar);
      shade(bsdfMaterial);
    } else if (hit.material == Refl_t::Nayar) {
      // Oren-Nayar Material
      if (fluidMemory.vrtxSurface && fluidHit)
        if (math::dot3(hit.normal, l_info.rayDirection) < 0.f)
          l_info.rayDirection *= -1.f;

      pbrt::BSDF bsdfMaterial(hit, 1.f);
      pbrt::OrenNayar orenNayar(l_info.k_d, l_info.roughness);
      bsdfMaterial.Add(&orenNayar);
      shade(bsdfMaterial);
    } else if (hit.material == Refl_t::Plastic) {
      // l_info.rayDirection = -l_info.rayDirection;
      pbrt::BSDF bsdfMaterial(hit, 1.f);
      pbrt::FresnelNoOp fresnelNoOp;
      pbrt::SpecularReflection specular(l_info.k_s, &fresnelNoOp);
      // bsdfMaterial.Add(&lmb);
      bsdfMaterial.Add(&specular);
      shade(bsdfMaterial);
    }
  }
} // namespace shading
} // namespace shading

hostDeviceInline auto minElem(float3 v) { return math::min(v.x, math::min(v.y, v.z)); }
hostDeviceInline auto maxElem(float3 v) { return math::max(v.x, math::max(v.y, v.z)); }
namespace sphIntersection {
__device__ __constant__ int3 fullLoopLUT[]{
    int3{0, 0, 0},   int3{0, 0, -1},   int3{0, 0, 1},   int3{0, -1, 0}, int3{0, -1, -1}, int3{0, -1, 1},
    int3{0, 1, 0},   int3{0, 1, -1},   int3{0, 1, 1},   int3{-1, 0, 0}, int3{-1, 0, -1}, int3{-1, 0, 1},
    int3{-1, -1, 0}, int3{-1, -1, -1}, int3{-1, -1, 1}, int3{-1, 1, 0}, int3{-1, 1, -1}, int3{-1, 1, 1},
    int3{1, 0, 0},   int3{1, 0, -1},   int3{1, 0, 1},   int3{1, -1, 0}, int3{1, -1, -1}, int3{1, -1, 1},
    int3{1, 1, 0},   int3{1, 1, -1},   int3{1, 1, 1}};
struct compactRayState {
  uint32_t rayDone : 1;
  uint32_t rayBounced : 1;
  uint32_t rayHitFluidAABB : 1;
  uint32_t rayHitFluidSurface : 1;
  int32_t index : 23;
  uint32_t rayStart : 1;
  uint32_t internal : 1;
};
#define IDRAY(i)                                                                                                       \
  Ray {                                                                                                                \
    {smRayOrig_x[i], smRayOrig_y[i], smRayOrig_z[i]}, { smRayDir_x[i], smRayDir_y[i], smRayDir_z[i] }                  \
  }
#define VOXEL(i)                                                                                                       \
  int3 { smVoxel_x[i], smVoxel_y[i], smVoxel_z[i] }
__global__ void raySchedulerIntersectFluidStatic6(int32_t numRays) {
  __shared__ float smRayOrig_x[64];
  __shared__ float smRayOrig_y[64];
  __shared__ float smRayOrig_z[64];
  __shared__ float smRayDir_x[64];
  __shared__ float smRayDir_y[64];
  __shared__ float smRayDir_z[64];
  __shared__ uint32_t smBegin[64];
  __shared__ uint32_t smLength[64];
  __shared__ int32_t smVoxel_x[64];
  __shared__ int32_t smVoxel_y[64];
  __shared__ int32_t smVoxel_z[64];

  const int32_t wIdx = threadIdx.x % 32;
  const int32_t wOffset = (threadIdx.x / 32) * 32;
  bool internal = false;
  compactRayState rs = compactRayState{0, 0, 0, 0, static_cast<int32_t>(threadIdx.x + blockIdx.x * blockDim.x),1, 0};
  // bool init = true;
  bool bounced = false;
  // rs = compactRayState{0, 0, 0, 0, util::atomicAggInc(cRayCounter)};
  if (rs.index >= numRays) {
    rs.index = -1;
  } else {
    auto idRay = cRaysDepth[rs.index];
    internal = idRay.internal;
    smRayOrig_x[threadIdx.x] = idRay.orig.x;
    smRayOrig_y[threadIdx.x] = idRay.orig.y;
    smRayOrig_z[threadIdx.x] = idRay.orig.z;
    idRay.dir = math::normalize3(idRay.dir);
    smRayDir_x[threadIdx.x] = idRay.dir.x;
    smRayDir_y[threadIdx.x] = idRay.dir.y;
    smRayDir_z[threadIdx.x] = idRay.dir.z;
    if (idRay.bounces >= fluidMemory.bounces)
      rs.index = -1;
    if (idRay.bounces >= 1)
      bounced = true;
    rs.internal = idRay.internal;
  }
  __syncwarp();
  float4 tMax;
  while (__any_sync(__activemask(), rs.index >= 0)) {
    if (rs.index >= 0 && !rs.rayHitFluidAABB) {
      auto aabb = aabb::rayIntersectAABB(IDRAY(threadIdx.x), fluidMemory.min_coord, fluidMemory.max_coord);
      aabb.tmin = math::max(0.f, aabb.tmin);
      auto mi = IDRAY(threadIdx.x)(aabb.tmin);
      int3 voxelPosition = getVoxel(mi, fluidMemory.min_coord, fluidMemory.cell_size);
      smVoxel_x[threadIdx.x] = voxelPosition.x;
      smVoxel_y[threadIdx.x] = voxelPosition.y;
      smVoxel_z[threadIdx.x] = voxelPosition.z;
      if (aabb.hit) {
        // rs.rayHitFluidSurface = 0;
        // rs.rayBounced = 0;
        rs.rayHitFluidAABB = 1;
        auto nxt = voxelPosition + int3{IDRAY(threadIdx.x).dir.x > 0 ? 1 : 0, IDRAY(threadIdx.x).dir.y > 0 ? 1 : 0,
                                        IDRAY(threadIdx.x).dir.z > 0 ? 1 : 0};
        auto nxtB = fluidMemory.min_coord + math::castTo<float3>(nxt) * fluidMemory.cell_size;
        tMax = float4{fabsf(IDRAY(threadIdx.x).dir.x) <= 1e-5f ? FLT_MAX : (nxtB.x - mi.x) / IDRAY(threadIdx.x).dir.x,
                      fabsf(IDRAY(threadIdx.x).dir.y) <= 1e-5f ? FLT_MAX : (nxtB.y - mi.y) / IDRAY(threadIdx.x).dir.y,
                      fabsf(IDRAY(threadIdx.x).dir.z) <= 1e-5f ? FLT_MAX : (nxtB.z - mi.z) / IDRAY(threadIdx.x).dir.z,
                      (aabb.tmax - aabb.tmin)};

        // tMax = math::castTo<float4>(traversal::intBoundRay(idRay, aabb.tmin));
        // tMax.w = (aabb.tmax - aabb.tmin) / minElem(fluidMemory.cell_size) + 1.f;
        // tMax = math::castTo<float4>(traversal::intBoundRay(IDRAY(threadIdx.x), aabb.tmin));
        // tMax.w = ((aabb.tmax - aabb.tmin) / minElem(fluidMemory.cell_size) + 1.f);
        // rs.rayBounced = 1;
      } else {
        rs.rayHitFluidAABB = 0;
        rs.rayDone = 1;
        rs.index = -1;
      }
    }
    if (rs.index >= 0 && !rs.rayDone) {
      int32_t ctr = 0;
      do {
        if (!rs.rayHitFluidSurface && !rs.rayBounced) {
          auto cell_idx = traversal::lookup_cell(VOXEL(threadIdx.x));
          // if (cell_idx != INT_MAX) {
          //  float3 min = fluidMemory.min_coord + math::castTo<float3>(VOXEL(threadIdx.x)) * fluidMemory.cell_size;
          //  float3 max = min + fluidMemory.cell_size;
          //  auto rH = aabb::rayIntersectAABB(IDRAY(threadIdx.x), min, max);
          //  auto hitPosition = IDRAY(threadIdx.x).orig + rH.tmin * IDRAY(threadIdx.x).dir;
          //  if (hitPosition.x < fluidMemory.vrtxDomainMin.x + fluidMemory.vrtxDomainEpsilon ||
          //      hitPosition.y < fluidMemory.vrtxDomainMin.y + fluidMemory.vrtxDomainEpsilon ||
          //      hitPosition.z < fluidMemory.vrtxDomainMin.z + fluidMemory.vrtxDomainEpsilon ||
          //      hitPosition.x > fluidMemory.vrtxDomainMax.x - fluidMemory.vrtxDomainEpsilon ||
          //      hitPosition.y > fluidMemory.vrtxDomainMax.y - fluidMemory.vrtxDomainEpsilon ||
          //      hitPosition.z > fluidMemory.vrtxDomainMax.z - fluidMemory.vrtxDomainEpsilon)
          //    cell_idx = INT_MAX;
          //}
          if (cell_idx != INT_MAX) {
            rs.rayDone = 1;
            rs.rayHitFluidSurface = 1;
          }
        }
        if (!rs.rayDone) {
          rs.rayBounced = 0;
          //if(rs.rayStart == 1) {
          //    rs.rayStart = 0;
          //    rs.internal = 0;
          //}
          if (tMax.x < tMax.y) {
            if (tMax.x < tMax.z) {
              if (tMax.x > tMax.w) {
                rs.rayDone = 1;
                rs.rayHitFluidSurface = 0;
                continue;
              }
              smVoxel_x[threadIdx.x] += sgn(smRayDir_x[threadIdx.x]);
              tMax.x += sgn(smRayDir_x[threadIdx.x]) / smRayDir_x[threadIdx.x] * arrays.cell_size.x;
            } else {
              if (tMax.z > tMax.w) {
                rs.rayDone = 1;
                rs.rayHitFluidSurface = 0;
                continue;
              }
              smVoxel_z[threadIdx.x] += sgn(smRayDir_z[threadIdx.x]);
              tMax.z += sgn(smRayDir_z[threadIdx.x]) / smRayDir_z[threadIdx.x] * arrays.cell_size.z;
            }
          } else {
            if (tMax.y < tMax.z) {
              if (tMax.y > tMax.w) {
                rs.rayDone = 1;
                rs.rayHitFluidSurface = 0;
                continue;
              }
              smVoxel_y[threadIdx.x] += sgn(smRayDir_y[threadIdx.x]);
              tMax.y += sgn(smRayDir_y[threadIdx.x]) / smRayDir_y[threadIdx.x] * arrays.cell_size.y;
            } else {
              if (tMax.z > tMax.w) {
                rs.rayDone = 1;
                rs.rayHitFluidSurface = 0;
                continue;
              }
              smVoxel_z[threadIdx.x] += sgn(smRayDir_z[threadIdx.x]);
              tMax.z += sgn(smRayDir_z[threadIdx.x]) / smRayDir_z[threadIdx.x] * arrays.cell_size.z;
            }
          }
        }
        if (++ctr > 1024) {
          rs.rayHitFluidAABB = 0;
          rs.rayHitFluidSurface = 0;
          rs.rayDone = true;
        }
      } while (__all_sync(__activemask(), !rs.rayDone));
    }
    __syncwarp();
    float depth = 1e21f;
    uint32_t mask = __brev(__ballot_sync(__activemask(), rs.rayHitFluidSurface));
    __syncwarp();
    while ((mask & __activemask()) != 0) {
      const int32_t offset = __clz(mask);
      mask = mask ^ (1 << (31 - offset));
      __syncwarp();
      float4 position;
      float t;
      {
        float3 min = fluidMemory.min_coord + math::castTo<float3>(VOXEL(wOffset + offset)) * fluidMemory.cell_size;
        auto rH = aabb::rayIntersectAABB(IDRAY(wOffset + offset), min, min + fluidMemory.cell_size);
        if (rH.hit == false)
          continue;
        rH.tmin = math::max(0.f, rH.tmin);
        // if (rH.tmin > rH.tmax) {
        //  rH.hit = false;
        //  continue;
        //}
        t = rH.tmin + (rH.tmax - rH.tmin) * (((float)(wIdx)) / 31.f);
        position = math::castTo<float4>(IDRAY(wOffset + offset)(t));
        position.w = support_from_volume(4.f / 3.f * CUDART_PI_F * powf(fluidMemory.renderRadius, 3.f));
      }
      __syncwarp();
      if (wIdx < 27) {
        smBegin[wOffset + wIdx] = 0;
        smLength[wOffset + wIdx] = 0;
        auto s = arrays.compactHashMap[hCoord(VOXEL(wOffset + offset) + fullLoopLUT[wIdx])];
        if (s.compacted == 0 && s.beginning != INVALID_BEG) {
          int32_t morton = zCoord(VOXEL(wOffset + offset) + fullLoopLUT[wIdx]);
          for (int32_t i = s.beginning; i < s.beginning + s.length; ++i) {
            auto cs = arrays.compactCellSpan[i];
            if (zCoord(arrays.position[cs.beginning]) == morton) {
              smBegin[wOffset + wIdx] = cs.beginning;
              smLength[wOffset + wIdx] = cs.length == INVALID_LEN ? arrays.auxLength[i] : cs.length;
              break;
            }
          }

        } else if (s.beginning != INVALID_BEG) {
          smBegin[wOffset + wIdx] = s.beginning;
          smLength[wOffset + wIdx] = s.length;
        }
      }
      __syncwarp();
      int32_t internal_flag = 0;
#ifdef ANISOTROPIC_SURFACE
      float levelSet = 0.f;
      int32_t flag = __shfl_sync(__activemask(), rs.index, offset);
      flag = cRaysDepth[0].internal;
      __syncwarp();
      for (uint32_t id = 0; id < 27; ++id) {
          __syncwarp();
        for (uint32_t j = smBegin[wOffset + id]; j < smBegin[wOffset + id] + smLength[wOffset + id]; ++j) {
            __syncwarp();
          bool pred = levelSet >= fluidMemory.vrtxR * 1.01f;
          uint32_t ballot = __ballot_sync(__activemask(), pred);
          if ((~ballot) == 0) break;
          uint32_t mask = __activemask();
#define GETBIT( val,n) ((val >> n) & 1u)
          if (pred && (wIdx == 0)) { 
              if (GETBIT(ballot, 1u) == 1)
                  continue; 
          }
          else if (pred && (wIdx == 31)) { 
              if (GETBIT(ballot, 30u) == 1)
                  continue;
          }
          else if (pred) {
              if (GETBIT(ballot, wIdx - 1u) && GETBIT(ballot, wIdx + 1u))
                  continue;
          }
          levelSet += util::turkAnisotropic(position, j);
        }
      }
      __syncwarp();
      internal_flag = __shfl_sync(__activemask(), rs.index, offset);
      // if (cRaysDepth[internal_flag].internal == 0 ? levelSet > fluidMemory.vrtxR : levelSet <= fluidMemory.vrtxR) {
      //  if (position.x < fluidMemory.vrtxDomainMin.x + fluidMemory.vrtxDomainEpsilon ||
      //      position.y < fluidMemory.vrtxDomainMin.y + fluidMemory.vrtxDomainEpsilon ||
      //      position.z < fluidMemory.vrtxDomainMin.z + fluidMemory.vrtxDomainEpsilon ||
      //      position.x > fluidMemory.vrtxDomainMax.x - fluidMemory.vrtxDomainEpsilon ||
      //      position.y > fluidMemory.vrtxDomainMax.y - fluidMemory.vrtxDomainEpsilon ||
      //      position.z > fluidMemory.vrtxDomainMax.z - fluidMemory.vrtxDomainEpsilon) {
      //    levelSet = cRaysDepth[internal_flag].internal ? -10.f : 10.f;
      //  }
      //}
      // uint32_t mask =
      //     __ballot_sync(__activemask(), cRaysDepth[internal_flag].internal == 0 ? levelSet > fluidMemory.vrtxR
//                                                                                : levelSet <= fluidMemory.vrtxR);
#endif
#ifdef ISO_DENSITY_SURFACE
      float levelSet = 0.f;
      for (int32_t id = 0; id < 27; ++id) {
        for (int32_t i = smBegin[wOffset + id]; i < smBegin[wOffset + id] + smLength[wOffset + id]; ++i) {
          float4 p = arrays.position[i];
          float v = arrays.volume[i];
          levelSet += v * spline4_kernel(p, position);
           if (levelSet > 0.2f) break;
        }
         if (levelSet > 0.2f) break;
      }
      int32_t internal_flag = 0;
      internal_flag = __shfl_sync(__activemask(), internal, offset);
      uint32_t mask = __ballot_sync(__activemask(),
                                    internal_flag == 0 ? levelSet > fluidMemory.vrtxR : levelSet < fluidMemory.vrtxR);
#endif
#if defined(ZHU_BRIDSON_SURFACE)
      float kernelSum = 0.f;
      float rho = 0.f;
      float4 xBar{0.f, 0.f, 0.f, 0.f};
      int32_t ctr = 0;
      float hSum = 0.f;
      for (int32_t id = 0; id < 27; ++id) {
        for (int32_t i = smBegin[wOffset + id]; i < smBegin[wOffset + id] + smLength[wOffset + id]; ++i) {
          float4 p = arrays.position[i];          
          auto H = p.w * kernelSize();// *fluidMemory.wmin;
          H = (support_from_volume(4.f / 3.f * CUDART_PI_F * powf(fluidMemory.renderRadius, 3.f)) * kernelSize() + H) * 0.5f;

          auto scale = 16.f / (CUDART_PI_F * H * H * H);
          auto q = math::distance3(p, position) / H;
          auto q1 = 1.f - q;
          auto q2 = 0.5f - q;
          auto k = (math::max(0.f, q1 * q1 * q1) - 4.f * math::max(0.f, q2 * q2 * q2)) * scale;
          rho += arrays.volume[i] * k;          
          kernelSum += k;
          xBar += p * k;
          hSum += ( k > 1e-3f ? H : 0.f);
          ctr = ctr + (k > 1e-3f ? 1 : 0);
        }
      }
      float levelSet = fluidMemory.vrtxR * 1.05f;
      if (ctr > 0 && rho > fluidMemory.wmax) {
          hSum /= ctr;
          levelSet = math::distance3(position, xBar / kernelSum) / hSum;
      }
      //float levelSet = ctr > 0  && rho > fluidMemory.wmax ? math::distance3(position, xBar / kernelSum) : fluidMemory.vrtxR * 100.5f;
#endif
#if defined(SMALL_SCALE_DETAIL)
      float levelSet = 0.f;
      float4 C1{0.f, 0.f, 0.f, 0.f};
      float C2 = 0.f;
      for (int32_t id = 0; id < 27; ++id) {
        for (int32_t i = smBegin[wOffset + id]; i < smBegin[wOffset + id] + smLength[wOffset + id]; ++i) {
          float4 p = arrays.position[i];
          float w = arrays.auxIsoDensity[i];
          C1 += kernel(p, position) / w * p;
          C2 += kernel(p, position) / w;
        }
      }
      auto Cp = C1 / C2;
      Cp.w = position.w;
      float w_c = 0.f;
      levelSet = math::distance3(position, Cp);
      for (int32_t id = 0; id < 27; ++id) {
        for (int32_t j = smBegin[wOffset + id]; j < smBegin[wOffset + id] + smLength[wOffset + id]; ++j) {
          float4 p = arrays.position[j];
          float w = arrays.auxIsoDensity[j];
          w_c += kernel(p, Cp) / w;
        }
      }
      float decay = square(1.f - square(w_c - fluidMemory.wmax) / square(fluidMemory.wmax - fluidMemory.wmin));
      levelSet -= fluidMemory.vrtxR * decay;
      // float levelSet = math::distance3(position, xBar / kernelSum) - fluidMemory.radius;
      // uint32_t mask = __ballot_sync(__activemask(), cInternalFlag[state.rs.index] != 0 ? levelSet > 0.8f *
      // fluidMemory.radius : levelSet < 0.8f * fluidMemory.radius); uint32_t mask = __ballot_sync(__activemask(),
      // levelSet < 0.f);
#endif
      internal_flag = __shfl_sync(__activemask(), rs.index, offset);

      uint32_t startFlag = rs.rayStart;
      startFlag = __shfl_sync(__activemask(), startFlag, offset);

      uint32_t internal = rs.internal;
      internal = __shfl_sync(__activemask(), internal, offset);
      if ((t <= 1e-5f && bounced)) {
#ifdef ZHU_BRIDSON_SURFACE
          if (internal) levelSet = fluidMemory.vrtxR * 0.95f;
          else levelSet = fluidMemory.vrtxR * 1.05f;
#else
          if (internal) levelSet = fluidMemory.vrtxR * 1.05f;
          else levelSet = fluidMemory.vrtxR * 0.95f;
#endif
      //    levelSet = (internal != 0 && )

      //    //    rs.internal = maskL & 0x1 ? 1 : 0;
      //    //    rs.internal = maskL & 0x1 ? 1 : 0;
      //    //    rs.rayStart = 0;
      }
#ifdef ZHU_BRIDSON_SURFACE
      uint32_t maskL = __ballot_sync(__activemask(), levelSet < fluidMemory.vrtxR);
#else
      uint32_t maskL = __ballot_sync(__activemask(), levelSet >= fluidMemory.vrtxR);
#endif
      uint32_t maskH = __ballot_sync(__activemask(), levelSet < fluidMemory.vrtxR);
      __syncwarp();
      uint32_t maskF = (maskL << 1) & maskH;
      uint32_t maskR = (~maskL << 1) & ~maskH;
      uint32_t idx = internal == 1 ? __ffs(~maskL) - 1 : __ffs(maskL) - 1;
      // uint32_t idx = __ffs(maskL) - 1;
       //uint32_t idxF = __ffs(maskF);
       //uint32_t idxR = __ffs(maskR);
       //uint32_t idx = idxF > idxR || idxR == 0 ? idxF - 1 : idxR - 1;
      if ((((maskL&0xFFFFFFFEul) != 0 && !internal) || ((maskL | 0x1) != __activemask() && internal)) && idx != 0)
      // if ((maskF != 0 || maskR != 0) && idx < 31)
      {
            float x0 = (float)(idx - 1);
            float y0 = __shfl_sync(__activemask(), levelSet, idx - 1);
            float x1 = (float)idx;
            float y1 = __shfl_sync(__activemask(), levelSet, idx);
            float dy = y1 - y0;
            float alpha = (fluidMemory.vrtxR - y0) / dy;
            alpha = math::clamp(alpha, 0.f, 1.f);
            //alpha = 0.f;
            float t0 = __shfl_sync(__activemask(), t, idx - 1);
            float t1 = __shfl_sync(__activemask(), t, idx);
            float t = t0 * (1.f - alpha) + t1 * alpha;
            if (wIdx == offset) {
                //auto position = IDRAY(wIdx)(t);
                //if (position.x < fluidMemory.vrtxDomainMin.x + fluidMemory.vrtxDomainEpsilon ||
                //    position.y < fluidMemory.vrtxDomainMin.y + fluidMemory.vrtxDomainEpsilon ||
                //    position.z < fluidMemory.vrtxDomainMin.z + fluidMemory.vrtxDomainEpsilon ||
                //    position.x > fluidMemory.vrtxDomainMax.x - fluidMemory.vrtxDomainEpsilon ||
                //    position.y > fluidMemory.vrtxDomainMax.y - fluidMemory.vrtxDomainEpsilon ||
                //    position.z > fluidMemory.vrtxDomainMax.z - fluidMemory.vrtxDomainEpsilon)
                //{
                //    rs.internal = !rs.internal;
                //}
                //else {
                    depth = t;
                    rs.rayHitFluidSurface = 0;
                    rs.rayBounced = 1;
                //}
            }
      }
      // if (maskL != 0/* && maskL != __activemask()*/) {
      //  if (wIdx == offset) {
      //    auto position = IDRAY(wOffset + offset)(depth);
      //    if (position.x < fluidMemory.vrtxDomainMin.x + fluidMemory.vrtxDomainEpsilon ||
      //        position.y < fluidMemory.vrtxDomainMin.y + fluidMemory.vrtxDomainEpsilon ||
      //        position.z < fluidMemory.vrtxDomainMin.z + fluidMemory.vrtxDomainEpsilon ||
      //        position.x > fluidMemory.vrtxDomainMax.x - fluidMemory.vrtxDomainEpsilon ||
      //        position.y > fluidMemory.vrtxDomainMax.y - fluidMemory.vrtxDomainEpsilon ||
      //        position.z > fluidMemory.vrtxDomainMax.z - fluidMemory.vrtxDomainEpsilon) {
      //      depth = 1e21f;
      //      rs.rayHitFluidSurface = 1;
      //    }
      //  }
      //}
      __syncwarp();
    }
    if ((rs.index != -1)) {
      if (rs.rayHitFluidSurface == 1 && (depth > 1e19f)) {
        rs.rayDone = 0;
        rs.rayBounced = 1;
        rs.rayHitFluidSurface = 0;
        rs.rayStart = 0;
      }
      if (rs.rayDone || rs.index == -2) {
        if (rs.rayDone)
          cfluidDepth[rs.index] = depth;
        bool init = true;
        // __syncwarp();
        do {
          // if (rs.index == -2 || init)
          rs = compactRayState{0, 0, 0, 0, util::atomicAggInc(cRayCounter),1,0};
          if (rs.index >= numRays) {
            rs.index = -1;
          } else {
            auto idRay = cRaysDepth[rs.index];
            smRayOrig_x[threadIdx.x] = idRay.orig.x;
            smRayOrig_y[threadIdx.x] = idRay.orig.y;
            smRayOrig_z[threadIdx.x] = idRay.orig.z;
            idRay.dir = math::normalize3(idRay.dir);
            smRayDir_x[threadIdx.x] = idRay.dir.x;
            smRayDir_y[threadIdx.x] = idRay.dir.y;
            smRayDir_z[threadIdx.x] = idRay.dir.z;
            internal = idRay.internal;
            if (idRay.bounces >= fluidMemory.bounces)
              rs.index = -1;
            bounced = idRay.bounces >= 1;
            rs.internal = idRay.internal;
          }
          init = false;
        } while (rs.index == -2);
        //__syncwarp(__activemask());
      }
    }
    __syncwarp();
  } // while (__any_sync(__activemask(), rs.index != -1));
}
__device__ float intersectParticle(const Ray &r, const float3 &pos, float rad) {
  // float radius2 = rad * rad;
  // float t0, t1; // solutions for t if the ray intersects
  // float3 L = pos - r.orig;
  // float tca = math::dot3(L,r.dir);
  //// if (tca < 0) return false;
  // float d2 = math::dot3(L, L) - tca * tca;
  // if (d2 > radius2) return FLT_MAX;
  // float thc = sqrt(radius2 - d2);
  // t0 = tca - thc;
  // t1 = tca + thc;
  // if (t0 > t1) {
  //	float tt = t0;
  //	t0 = t1;
  //	t1 = tt;
  //}

  float3 oc = r.orig - pos;
  float a = math::dot3(r.dir, r.dir);
  float b = 2.f * math::dot3(oc, r.dir);
  float c = math::dot3(oc, oc) - rad * rad;
  float discriminant = b * b - 4.f * a * c;
  if (discriminant < 0) {
    return FLT_MAX;
  } else {
    return (-b - sqrtf(discriminant)) / (2.f * a);
  }
  // float t = t0;
  // return t;

  // float3 op = pos - r.orig; //
  // float t, epsilon = 0.01f;
  // float b = math::dot(op, r.dir);
  // float disc = b * b - math::dot(op, op) + rad * rad; // discriminant
  // if (disc < 0)
  //	return 0;
  // else
  //	disc = sqrtf(disc);
  // return (t = b - disc) > epsilon ? t : ((t = b + disc) > epsilon ? t : 0);
}
__global__ void raySchedulerIntersectFluidStaticParticles(bool particles) {
  // const int32_t wIdx = threadIdx.x % 32;
  // const int32_t wOffset = (threadIdx.x / 32) * 32;
  compactRayState rs = compactRayState{0u, 1u, 0u, 0u, (int32_t)(threadIdx.x + blockIdx.x * blockDim.x)};
  Ray idRay;
  int3 voxelPosition;
  float4 tMax;
  if (rs.index >= cNumRays) {
    rs.index = -1;
  } else {
    auto ir = cRaysDepth[rs.index];
    if (ir.bounces >= fluidMemory.bounces)
      rs.index = -1;
    idRay = ir;
  }
  int32_t counter = 0;
  while (rs.index != -1) {
    if (rs.index != -1 && !rs.rayHitFluidAABB) {
      auto aabb = aabb::rayIntersectAABB(idRay, fluidMemory.min_coord, fluidMemory.max_coord);
      aabb.tmin = math::max(0.f, aabb.tmin);
      // aabb.tmin = 0.f;
      auto mi = idRay(aabb.tmin);
      voxelPosition = getVoxel(mi, fluidMemory.min_coord, fluidMemory.cell_size);
      if (aabb.hit) {
        rs.rayHitFluidAABB = 1;
        auto nxt = voxelPosition + int3{idRay.dir.x > 0 ? 1 : 0, idRay.dir.y > 0 ? 1 : 0, idRay.dir.z > 0 ? 1 : 0};
        auto nxtB = fluidMemory.min_coord + math::castTo<float3>(nxt) * fluidMemory.cell_size;
        tMax = float4{fabsf(idRay.dir.x) <= 1e-5f ? FLT_MAX : (nxtB.x - mi.x) / idRay.dir.x,
                      fabsf(idRay.dir.y) <= 1e-5f ? FLT_MAX : (nxtB.y - mi.y) / idRay.dir.y,
                      fabsf(idRay.dir.z) <= 1e-5f ? FLT_MAX : (nxtB.z - mi.z) / idRay.dir.z, (aabb.tmax - aabb.tmin)};

        // tMax = math::castTo<float4>(traversal::intBoundRay(idRay, aabb.tmin));
        // tMax.w = (aabb.tmax - aabb.tmin) / minElem(fluidMemory.cell_size) + 1.f;
        // rs.rayBounced = 1;
      } else {
        rs.rayHitFluidAABB = 0;
        rs.rayDone = 1;
        rs.index = -1;
      }
    }
    while (rs.index != -1 && !rs.rayDone) {
      if (counter++ > (1 << 16)) {
        rs.rayDone = 1;
        rs.rayHitFluidSurface = 0;
      }
      if (!rs.rayBounced) {
        auto cell_idx = traversal::lookup_cell(voxelPosition);
        if (cell_idx != INT_MAX) {
          rs.rayDone = 1;
          rs.rayHitFluidSurface = 1;
        }
      }
      if (!rs.rayDone) {
        rs.rayBounced = 0;
        if (tMax.x < tMax.y) {
          if (tMax.x < tMax.z) {
            if (tMax.x > tMax.w) {
              rs.rayDone = 1;
              rs.rayHitFluidSurface = 0;
              continue;
            }
            voxelPosition.x += sgn(idRay.dir.x);
            tMax.x += sgn(idRay.dir.x) / idRay.dir.x * arrays.cell_size.x;
          } else {
            if (tMax.z > tMax.w) {
              rs.rayDone = 1;
              rs.rayHitFluidSurface = 0;
              continue;
            }
            voxelPosition.z += sgn(idRay.dir.z);
            tMax.z += sgn(idRay.dir.z) / idRay.dir.z * arrays.cell_size.z;
          }
        } else {
          if (tMax.y < tMax.z) {
            if (tMax.y > tMax.w) {
              rs.rayDone = 1;
              rs.rayHitFluidSurface = 0;
              continue;
            }
            voxelPosition.y += sgn(idRay.dir.y);
            tMax.y += sgn(idRay.dir.y) / idRay.dir.y * arrays.cell_size.y;
          } else {
            if (tMax.z > tMax.w) {
              rs.rayDone = 1;
              rs.rayHitFluidSurface = 0;
              continue;
            }
            voxelPosition.z += sgn(idRay.dir.z);
            tMax.z += sgn(idRay.dir.z) / idRay.dir.z * arrays.cell_size.z;
          }
        }
      }
    }
    //__syncwarp();
    float depth = 1e20f;
    // float3 n;
    if (rs.rayHitFluidSurface) {
      if (!particles) {
        constexpr auto epsilon = 1e-1f;
        float3 min = fluidMemory.min_coord + math::castTo<float3>(voxelPosition) * fluidMemory.cell_size;
        float3 max = min + fluidMemory.cell_size;
        auto rH = aabb::rayIntersectAABB(idRay, min, max);
        auto hitPosition = idRay.orig + rH.tmin * idRay.dir;
        auto c = (min + max) * 0.5f;
        auto prel = hitPosition - c;
        auto d = math::abs((min - max) * 0.5f);
        auto n = math::castTo<int3>(prel / d * (1.f + epsilon));
        char3 nc = char3{static_cast<char>(n.x), static_cast<char>(n.y), static_cast<char>(n.z)};
        int32_t morton = zCoord(voxelPosition);

        // if (rH.hit && rH.tmin > 0.f && rH.tmin < depth) {
        //  auto hitPosition = idRay.orig + rH.tmin * idRay.dir;
        //  auto c = (min + max) * 0.5f;
        //  auto prel = hitPosition - c;
        //  auto d = math::abs((min - max) * 0.5f);
        //  auto n = math::castTo<int3>(prel / d * (1.f + epsilon));
        //  char3 nc = char3{static_cast<char>(n.x), static_cast<char>(n.y), static_cast<char>(n.z)};

        //  float3 nf = math::castTo<float3>(nc);
        //  cfluidIntersection[rs.index] = float4{(float)n.x, (float)n.y, (float)n.z, rH.tmin};
        //  float4 col{0.8f, 0.8f, 0.8f, 1.f};
        //  depth = rH.tmin;
        //  cFluidColor[rs.index] = col;
        //} else {
        //  rs.rayDone = 0;
        //  rs.rayBounced = 1;
        //  rs.rayHitFluidSurface = 0;
        //}

        auto beginning = 0u;
        auto length = 0u;
        auto s = arrays.compactHashMap[hCoord(voxelPosition)];
        if (s.compacted == 0u && s.beginning != INVALID_BEG) {
          for (auto i = s.beginning; i < s.beginning + s.length; ++i) {
            auto cs = arrays.compactCellSpan[i];
            if (zCoord(arrays.position[cs.beginning]) == morton) {
              beginning = cs.beginning;
              length = length == INVALID_LEN ? arrays.auxLength[i] : cs.length;
              break;
            }
          }

        } else if (s.beginning != INVALID_BEG) {
          if (zCoord(arrays.position[(int32_t)s.beginning]) == morton) {
            beginning = s.beginning;
            length = s.length;
          }
        }
        if (length == 0) {
          rs.rayDone = 0;
          rs.rayBounced = 1;
          rs.rayHitFluidSurface = 0;
        } else {
          /*if (rH.hit && rH.tmin > 0.f && rH.tmin < depth) {
            auto hitPosition = idRay.orig + rH.tmin * idRay.dir;
            auto c = (min + max) * 0.5f;
            auto prel = hitPosition - c;
            auto d = math::abs((min - max) * 0.5f);
            auto n = math::castTo<int3>(prel / d * (1.f + epsilon));
            char3 nc = char3{static_cast<char>(n.x), static_cast<char>(n.y), static_cast<char>(n.z)};

            float3 nf = math::castTo<float3>(nc);
            cfluidIntersection[rs.index] = float4{(float)n.x, (float)n.y, (float)n.z, rH.tmin};
            float4 col{0.8f, 0.8f, 0.8f, 1.f};
            depth = rH.tmin;
            cFluidColor[rs.index] = col;
          } else {
            rs.rayDone = 0;
            rs.rayBounced = 1;
            rs.rayHitFluidSurface = 0;
          }*/

          float intensity = length > 0 ? arrays.renderArray[beginning].w : 0.f;

          int32_t min_level = INT32_MAX;
          for (int32_t i = beginning; i < beginning + length; ++i) {
            min_level = math::min(arrays.MLMResolution[i], min_level);
            intensity = math::min(intensity, arrays.renderArray[i].w);
          }
          {
            // float factor = powf(1.f / 2.f, (float)min_level);
            // auto pos = hitPosition;
            // int3 voxel = zCoord(pos, fluidMemory.min_coord, fluidMemory.cell_size.x * factor);
            // float3 min = fluidMemory.min_coord + math::castTo<float3>(voxel) * fluidMemory.cell_size * factor;
            // float3 max = min + fluidMemory.cell_size * factor;
            // auto rH = aabb::rayIntersectAABB(idRay, min, max);
            if (rH.hit && rH.tmin > 0.f && rH.tmin < depth) {
              auto hitPosition = idRay.orig + rH.tmin * idRay.dir;
              // if (pos.x < fluidMemory.vrtxDomainMin.x ||
              //	pos.y < fluidMemory.vrtxDomainMin.y ||
              //	pos.z < fluidMemory.vrtxDomainMin.z ||
              //	pos.x > fluidMemory.vrtxDomainMax.x ||
              //	pos.y > fluidMemory.vrtxDomainMax.y ||
              //	pos.z > fluidMemory.vrtxDomainMax.z)
              //	continue;
              auto c = (min + max) * 0.5f;
              auto prel = hitPosition - c;
              auto d = math::abs((min - max) * 0.5f);
              auto n = math::castTo<int3>(prel / d * (1.f + epsilon));
              char3 nc = char3{static_cast<char>(n.x), static_cast<char>(n.y), static_cast<char>(n.z)};

              float3 nf = math::castTo<float3>(nc);
              cfluidIntersection[rs.index] = float4{(float)n.x, (float)n.y, (float)n.z, rH.tmin};
              float4 col{0.8f, 0.8f, 0.8f, 1.f};
              depth = rH.tmin;
            }
          }
          if (depth > 1e19f) {
            rs.rayDone = 0;
            rs.rayBounced = 1;
            rs.rayHitFluidSurface = 0;
          } else {
            // auto intensity = (float)min_level;
            auto mapValue = [&](float value, float min, float max, int32_t mode) {
              // if (mode == 0)
              //	return (value - min) / (max - min);
              // if (mode == 1)
              //	return (sqrt(value) - sqrt(min)) / (sqrt(max) - sqrt(min));
              // if (mode == 2)
              //	return (value * value - min * min) / (max * max - min * min);
              // if (mode == 3)
              //	return (pow(value, 1.f / 3.f) - pow(min, 1.f / 3.f)) / (pow(max, 1.f / 3.f) - pow(min, 1.f
              /// 3.f)); if (mode == 4) 	return (value * value * value - min * min * min) / (max * max * max -
              /// min
              /// * min
              //* min); if (mode == 5) 	return (log(value) - log(min)) / (log(max) - log(min));
              return (value - min) / (max - min);
            };
            if (arrays.minMap < arrays.maxMap)
              intensity = mapValue(intensity, arrays.minMap, arrays.maxMap, arrays.transferFn);
            else
              intensity = mapValue(intensity, arrays.maxMap, arrays.minMap, arrays.transferFn);
            intensity = math::clamp(intensity, 0.f, 1.f);

            if (fluidMemory.colorMapFlipped)
              intensity = 1.f - intensity;
            intensity *= ((float)fluidMemory.colorMapLength);
            int32_t lower = floorf(intensity);
            int32_t upper = ceilf(intensity);
            float mod = intensity - (float)lower;
            float4 col = fluidMemory.colorMap[lower] * mod + fluidMemory.colorMap[upper] * (1.f - mod);
            cFluidColor[rs.index] = float4{col.x, col.y, col.z, 1.f};
          }
          // float3 nf = math::castTo<float3>(nc);
          // cfluidIntersection[rs.index] = float4{ (float)n.x,  (float)n.y,  (float)n.z, rH.tmin };
          // float4 col{ 0.8f, 0.8f, 0.8f, 1.f };
          // cFluidColor[rs.index] = float4{ col.x, col.y, col.z, 1.f };
        }
      } else {
        //
        int32_t idx = -1;
        float3 min = fluidMemory.min_coord + math::castTo<float3>(voxelPosition) * fluidMemory.cell_size;
        float3 max = min + fluidMemory.cell_size;
        auto rH = aabb::rayIntersectAABB(idRay, min, max);
        for (auto ii : util::compactIterator(voxelPosition)) {
          float3 pos = math::castTo<float3>(arrays.position[ii]);
          auto t = intersectParticle(idRay, pos, radiusFromVolume(ii));
          if (t < depth && t > 1e-2f && t < rH.tmax) {
            // auto hitPosition = idRay.orig + t * idRay.dir;
            // if (pos.x < fluidMemory.vrtxDomainMin.x ||
            //	pos.y < fluidMemory.vrtxDomainMin.y ||
            //	pos.z < fluidMemory.vrtxDomainMin.z ||
            //	pos.x > fluidMemory.vrtxDomainMax.x ||
            //	pos.y > fluidMemory.vrtxDomainMax.y ||
            //	pos.z > fluidMemory.vrtxDomainMax.z)
            //	continue;
            idx = ii;
            depth = t;
          }
        }
        if (idx != -1) {
          float3 pos = math::castTo<float3>(arrays.position[idx]);
          auto x = idRay.orig + depth * idRay.dir;
          // n = math::normalize(float3{x.x - pos.x, x.y - pos.y, x.z - pos.z});
          cfluidIntersection[rs.index] = float4{x.x - pos.x, x.y - pos.y, x.z - pos.z, depth};
          auto mapValue = [&](float value, float min, float max, int32_t mode) {
            if (arrays.transferFn == 0)
              return (value - min) / (max - min);
            if (arrays.transferFn == 1)
              return (sqrtf(value) - sqrtf(min)) / (sqrtf(max) - sqrtf(min));
            if (arrays.transferFn == 2)
              return (value * value - min * min) / (max * max - min * min);
            if (arrays.transferFn == 3)
              return (pow(value, 1.f / 3.f) - pow(min, 1.f / 3.f)) / (pow(max, 1.f / 3.f) - pow(min, 1.f / 3.f));
            if (arrays.transferFn == 4)
              return (value * value * value - min * min * min) / (max * max * max - min * min * min);
            if (arrays.transferFn == 5)
              return (log(value) - log(min)) / (log(max) - log(min));
            return (value - min) / (max - min);
          };
          float intensity = arrays.renderArray[idx].w;
          if (arrays.minMap < arrays.maxMap)
            intensity = mapValue(intensity, arrays.minMap, arrays.maxMap, arrays.transferFn);
          else
            intensity = mapValue(intensity, arrays.maxMap, arrays.minMap, arrays.transferFn);
          intensity = math::clamp(intensity, 0.f, 1.f);

          if (fluidMemory.colorMapFlipped)
            intensity = 1.f - intensity;
          intensity = mapValue(intensity, 0.f, 1.f, arrays.mappingFn);
          float scaled = intensity * ((float)fluidMemory.colorMapLength);
          int32_t lower = floorf(scaled);
          int32_t upper = ceilf(scaled);
          float mod = scaled - (float)lower;
          float4 col = fluidMemory.colorMap[lower] * mod + fluidMemory.colorMap[upper] * (1.f - mod);
          cFluidColor[rs.index] = float4{col.x, col.y, col.z, 1.f};

          rs.rayDone = 1;
        } else {
          rs.rayDone = 0;
          rs.rayBounced = 1;
          rs.rayHitFluidSurface = 0;
        }
      }
    }
    if (rs.index != -1) {
      if (rs.rayDone) {
        // cfluidIntersection[rs.index] = float4{ tMax.x / tMax.w, tMax.y / tMax.w, tMax.z / tMax.w, FLT_MAX };
        rs = compactRayState{0, 1, 0, 0, util::atomicAggInc(cRayCounter)};
        counter = 0;
        if (rs.index >= cNumRays) {
          rs.index = -1;
        } else {
          auto ir = cRaysDepth[rs.index];
          if (ir.bounces >= fluidMemory.bounces)
            rs.index = -1;
          idRay = ir;
        }
      }
    }
  }
}
__global__ void raySchedulerCalculateNormalsSolo(int32_t numRays) {
  int32_t rIdx = threadIdx.x + blockIdx.x * blockDim.x;
  float depth;
  float4 position;
  if (rIdx < numRays) {
    Ray idRay = cRaysDepth[rIdx];
    depth = cfluidDepth[rIdx];
    position = math::castTo<float4>(idRay(depth));
    position.w = support_from_volume(4.f / 3.f * CUDART_PI_F * powf(fluidMemory.renderRadius, 3.f));
  } else {
    rIdx = -1;
  }
  while (rIdx != -1) {
    if (depth < 1e19f) {
      float4 normal{0.f, 0.f, 0.f, 0.f};
#ifndef ANISOTROPIC_SURFACE
      auto vIdx = getVoxel(float3{position.x, position.y, position.z}, arrays.min_coord, arrays.cell_size);
      for (const auto &j : util::compactIterator(vIdx)) {
          position.w = arrays.position[j].w;
        normal += arrays.auxIsoDensity[j] / arrays.density[j] * gradient(position, arrays.position[j]);
      }
      normal = -math::normalize3(normal);
      cfluidIntersection[rIdx] = float4{normal.x, normal.y, normal.z, depth};
#endif
#ifdef ANISOTROPIC_SURFACE
      float levelSetx = 0.f, levelSety = 0.f, levelSetz = 0.f, levelSet = 0.f;
      float dr = 0.001f;
      // for (const auto& j : compactIterator(position))
      // iterateCells()
      normal = float4{0.f, 0.f, 0.f, 0.f};
      auto vIdx = getVoxel(float3{position.x, position.y, position.z}, arrays.min_coord, arrays.cell_size);
      for (const auto &j : util::compactIterator(vIdx)) {
        normal += math::castTo<float4>(util::turkAnisotropicGradient(position, j));
        // levelSet += util::turkAnisotropic(position, j);
        // levelSetx += util::turkAnisotropic(position + float4{dr, 0.f, 0.f, 0.f}, j);
        // levelSety += util::turkAnisotropic(position + float4{0.f, dr, 0.f, 0.f}, j);
        // levelSetz += util::turkAnisotropic(position + float4{0.f, 0.f, dr, 0.f}, j);
      }
      // normal = float4{levelSetx, levelSety, levelSetz, 0.f};
      // normal = (normal - levelSet) / dr;
      normal = -math::normalize3(normal);
      // normal = float4{1.f, 0.f, 0.f, 0.f};
      cfluidIntersection[rIdx] = float4{normal.x, normal.y, normal.z, depth};
#endif
    }
    if (rIdx != -1) {
      rIdx = util::atomicAggInc(cRayCounter);
      if (rIdx < numRays) {
        Ray idRay = cRaysDepth[rIdx];
        depth = cfluidDepth[rIdx];
        position = math::castTo<float4>(idRay(depth));
        position.w = support_from_volume(4.f / 3.f * CUDART_PI_F * powf(fluidMemory.renderRadius, 3.f));
      } else {
        rIdx = -1;
      }
    }
  }
}
} // namespace sphIntersection
} // namespace render
} // namespace vrtx
#define TOKENPASTE(x, y) x##y
#define TOKENPASTE2(x, y) TOKENPASTE(x, y)

struct valid_fn {
  deviceInline bool operator()(const int32_t x) { return x < 2 * vrtx::cNumRays; }
};

std::map<std::string, std::vector<float>>
cuVRTXRender(SceneInformation scene, cudaGraphicsResource_t resource, vrtx::objectLoader &sceneMeshes,
             vrtxFluidMemory fmem, vrtxFluidArrays farrays, float3 *acc, unsigned framenumber, unsigned hashedframes,
             int32_t renderMode, int32_t bounces, bool fluidRender, int32_t renderGrid, int32_t surfacingTechnique,
             vrtx::Refl_t fluidMaterial, bool dirty, std::vector<vrtx::Sphere> spheres, std::vector<vrtx::Box> boxes) {
  // using namespace vrtx;
  static std::random_device rd;
  static std::uniform_int_distribution<uint32_t> dist(0, UINT_MAX);
  static bool once = true;
  static cudaStream_t stream;
  constexpr auto msaa = 1;
  constexpr int32_t blocks_1080 = 16 * 68;
  constexpr int32_t blockSize_1080 = 64;
  static vrtx::gpuBVH *bvhs = nullptr;
  int32_t width = static_cast<int32_t>(scene.width);
  int32_t height = static_cast<int32_t>(scene.height);
  int32_t numRays = width * height * msaa;
  int32_t num_blocks = blocks_1080 * blockSize_1080;
  std::map<std::string, std::vector<float>> timers;
  // std::cout << "vRTX renderer built at " << __TIMESTAMP__ << std::endl;
  if (once) {
    std::cout << "vRTX renderer built at " << __TIMESTAMP__ << std::endl;
    cudaStreamCreate(&stream);
    cudaMalloc(&vrtx::cuImage, sizeof(vrtx::Pixel) * width * height);
    cudaMalloc(&vrtx::cuCurrentRays, sizeof(vrtx::worldRay) * width * height * 2);
    cudaMalloc(&vrtx::cuCompactedRays, sizeof(vrtx::worldRay) * width * height * 2);
    cudaMalloc(&vrtx::rayCounter, sizeof(int32_t));
    cudaMalloc(&vrtx::cRNGSeeds, sizeof(uint32_t) * numRays);
    cudaMalloc(&vrtx::cuResortIndex, sizeof(int32_t) * numRays);
    cudaMalloc(&vrtx::cuResortKey, sizeof(int32_t) * numRays);
    cudaMalloc(&vrtx::cufluidDepth, sizeof(float) * width * height);
    // cudaMalloc(&vrtx::cuInternalFlag, sizeof(int32_t) * width * height);
    cudaMalloc(&vrtx::cuFluidColor, sizeof(float4) * width * height);
    cudaMalloc(&vrtx::cufluidIntersection, sizeof(float4) * width * height);
    cudaMalloc(&bvhs, sizeof(vrtx::gpuBVH));
    std::vector<int32_t> seeds;
    for (int32_t i = 0; i < numRays; ++i)
      seeds.push_back(dist(rd));
    cudaMemcpy(vrtx::cRNGSeeds, seeds.data(), sizeof(int32_t) * numRays, cudaMemcpyHostToDevice);
    cudaArray_t color_arr;
    cudaGraphicsMapResources(1, &resource, 0);
    cudaGraphicsSubResourceGetMappedArray(&color_arr, resource, 0, 0);
    cudaBindSurfaceToArray(vrtx::surfaceWriteOut, color_arr);
    once = false;
    CPSYMBOL(vrtx::cNumRays, numRays);
    CPSYMBOL(vrtx::cResortIndex, vrtx::cuResortIndex);
    CPSYMBOL(vrtx::cResortKey, vrtx::cuResortKey);
    CPSYMBOL(vrtx::cRaysDepth, vrtx::cuCurrentRays);
    CPSYMBOL(vrtx::cCompactRays, vrtx::cuCompactedRays);
    CPSYMBOL(vrtx::cImage, vrtx::cuImage);
    CPSYMBOL(vrtx::cRayCounter, vrtx::rayCounter);
    // CPSYMBOL(vrtx::cInternalFlag, vrtx::cuInternalFlag);
    CPSYMBOL(vrtx::cFluidColor, vrtx::cuFluidColor);
    CPSYMBOL(vrtx::cuSeeds, vrtx::cRNGSeeds);
    CPSYMBOL(vrtx::cfluidDepth, vrtx::cufluidDepth);
    CPSYMBOL(vrtx::cfluidIntersection, vrtx::cufluidIntersection);
  }
  if (dirty) {
    if (vrtx::cuBoxes != nullptr)
      cudaFree(vrtx::cuBoxes);
    if (vrtx::cuSpheres != nullptr)
      cudaFree(vrtx::cuSpheres);
    int32_t numBoxes = (int32_t)boxes.size();
    int32_t numSpheres = (int32_t)spheres.size();
    cudaMalloc(&vrtx::cuBoxes, sizeof(vrtx::Box) * numBoxes);
    cudaMalloc(&vrtx::cuSpheres, sizeof(vrtx::Sphere) * numSpheres);
    cudaMemcpy(vrtx::cuBoxes, boxes.data(), sizeof(vrtx::Box) * numBoxes, cudaMemcpyHostToDevice);
    cudaMemcpy(vrtx::cuSpheres, spheres.data(), sizeof(vrtx::Sphere) * numSpheres, cudaMemcpyHostToDevice);
    CPSYMBOL(vrtx::cBoxes, vrtx::cuBoxes);
    CPSYMBOL(vrtx::cNumBoxes, numBoxes);
    CPSYMBOL(vrtx::cSpheres, vrtx::cuSpheres);
    CPSYMBOL(vrtx::cNumSpheres, numSpheres);
  }
  // cuda::sync(std::to_string(__LINE__));
  // scene.m_camera.apertureRadius = 0.f;
  CPSYMBOL(vrtx::cScene, scene);
  CPSYMBOL(vrtx::fluidMemory, fmem);
  CPSYMBOL(vrtx::arrays, farrays);
  if (get<parameters::boundary_volumes::volume>().size() > 0) {
    vrtx::gpuBVH bvhs_host[] = {sceneMeshes.getGPUArrays()};
    cudaDeviceSynchronize();
    cudaMemcpy(bvhs, bvhs_host, sizeof(vrtx::gpuBVH), cudaMemcpyHostToDevice);
  }
  dim3 texturedim((uint32_t)scene.width, (uint32_t)scene.height, 1);
  dim3 blockdim(8, 8, 1);
  dim3 griddim(texturedim.x / blockdim.x, texturedim.y / blockdim.y, 1);
  if (texturedim.x % blockdim.x != 0)
    griddim.x += 1;
  if (texturedim.y % blockdim.y != 0)
    griddim.y += 1;
  auto launchDim = [](int32_t threadCount, int32_t blockSize) {
    return threadCount / blockSize + (threadCount % blockSize == 0 ? 0 : 1);
  };

  // cuda::sync(std::to_string(__LINE__));
#define MEASURE_CUDA(name, str, x)                                                                                     \
  static cudaEvent_t TOKENPASTE2(start_ev, __LINE__), TOKENPASTE2(end_ev, __LINE__);                                   \
  static bool TOKENPASTE2(onces, __LINE__) = true;                                                                     \
  if (TOKENPASTE2(onces, __LINE__)) {                                                                                  \
    cudaEventCreate(&TOKENPASTE2(start_ev, __LINE__));                                                                 \
    cudaEventCreate(&TOKENPASTE2(end_ev, __LINE__));                                                                   \
    TOKENPASTE2(onces, __LINE__) = false;                                                                              \
  }                                                                                                                    \
  cudaEventRecord(TOKENPASTE2(start_ev, __LINE__));                                                                    \
  x;                                                                                                                   \
  cudaEventRecord(TOKENPASTE2(end_ev, __LINE__));                                                                      \
  name = [&]() mutable {                                                                                               \
    cudaEventSynchronize(TOKENPASTE2(end_ev, __LINE__));                                                               \
    float TOKENPASTE2(milliseconds, __LINE__) = 0;                                                                     \
    cudaEventElapsedTime(&TOKENPASTE2(milliseconds, __LINE__), TOKENPASTE2(start_ev, __LINE__),                        \
                         TOKENPASTE2(end_ev, __LINE__));                                                               \
    return TOKENPASTE2(milliseconds, __LINE__);                                                                        \
  };

  std::function<float()> generateRays, intersectFluid, calculateNormals, bounceRays, toneMap, sortRays;
  MEASURE_CUDA(generateRays, stream,
               LAUNCH(vrtx::common::generateBlockedRays, griddim, dim3(msaa, blockdim.x, blockdim.y), 0,
                      stream)(hashedframes, vrtx::cuImage, vrtx::cuCurrentRays, vrtx::cuCurrentRays, msaa););
  int32_t rayCount = numRays;
  auto resortRays = [&]() mutable {
    // cuda::sync(std::to_string(__LINE__));
    thrust::cuda::par.on(stream);
    // cuda::sync(std::to_string(__LINE__));
    int32_t blockSize = 256;
    int32_t blocks = rayCount / blockSize + (rayCount % blockSize == 0 ? 0 : 1);
    vrtx::render::intersectAABB<<<blocks, blockSize, 0, stream>>>(rayCount);
    // cuda::sync(std::to_string(__LINE__));
    // cudaDeviceSynchronize();
    algorithm::sort_by_key(rayCount, vrtx::cuResortKey, vrtx::cuResortIndex);
    // cuda::sync(std::to_string(__LINE__));
    cudaDeviceSynchronize();
    cudaMemcpy(vrtx::cuCompactedRays, vrtx::cuCurrentRays, sizeof(vrtx::worldRay) * rayCount, cudaMemcpyDeviceToDevice);
    vrtx::render::sort<<<blocks, blockSize, 0, stream>>>(rayCount);
    // cuda::sync(std::to_string(__LINE__));
    std::swap(vrtx::cuCurrentRays, vrtx::cuCompactedRays);
    // cuda::sync(std::to_string(__LINE__));
    cudaMemcpyToSymbolAsync(vrtx::cRaysDepth, &vrtx::cuCurrentRays, sizeof(vrtx::cRaysDepth), 0, cudaMemcpyHostToDevice,
                            stream);
    // cuda::sync(std::to_string(__LINE__));
    cudaMemcpyToSymbolAsync(vrtx::cCompactRays, &vrtx::cuCompactedRays, sizeof(vrtx::cCompactRays), 0,
                            cudaMemcpyHostToDevice, stream);
    // cuda::sync(std::to_string(__LINE__));
    thrust::cuda::par.on(0);
    // cuda::sync(std::to_string(__LINE__));
    return (float)algorithm::count_if(vrtx::cuResortKey, rayCount, valid_fn{});
    // std::cout << __LINE__ << std::endl;
    // cudaDeviceSynchronize();
    // getchar();
  };

  // MEASURE_CUDA(sortRays, stream, auto ctr = resortRays());
  // launch<sort>(diff, mem, diff, mem.auxCellSpan, (compactListEntry*)mem.compactCellSpanSwap);

  // cuda::sync(std::to_string(__LINE__));
  timers[std::string("00Ray generation")].push_back(generateRays());
  // timers[std::string("sort rays    ")].push_back(sortRays());
  // timers[std::string("collision ctr")].push_back(-ctr);
  for (int32_t i = 0; i < (/*get<parameters::render_settings::vrtxRenderNormals>() ? 1 :*/ get<parameters::render_settings::vrtxBounces>()); ++i) {
    // std::cout << "Iteration " << i << " " << rayCount << "\n";
    MEASURE_CUDA(sortRays, stream, rayCount = resortRays());
    // std::cout << "Valid rays: " << rayCount << std::endl;
    timers[std::string("02Ray sorting   ")].push_back((float)sortRays());
    timers[std::string("01Valid rays    ")].push_back(-rayCount);
    if (rayCount == 0)
      break;
    cuda::sync(std::to_string(__LINE__));
    if (get<parameters::render_settings::vrtxRenderFluid>()) {
      if (get<parameters::render_settings::vrtxRenderSurface>()) {
        MEASURE_CUDA(intersectFluid, stream,
                     cudaMemcpyAsync(vrtx::rayCounter, &num_blocks, sizeof(int32_t), cudaMemcpyHostToDevice, stream);
                     cudaStreamSynchronize(stream);
                     LAUNCH(vrtx::render::sphIntersection::raySchedulerIntersectFluidStatic6, blocks_1080,
                            blockSize_1080, 0, stream)(rayCount);
                     cudaStreamSynchronize(stream););
        MEASURE_CUDA(calculateNormals, stream,
                     cudaMemcpyAsync(vrtx::rayCounter, &num_blocks, sizeof(int32_t), cudaMemcpyHostToDevice, stream);
                     LAUNCH(vrtx::render::sphIntersection::raySchedulerCalculateNormalsSolo, blocks_1080,
                            blockSize_1080, 0, stream)(rayCount););
      } else {
        MEASURE_CUDA(intersectFluid, stream,
                     cudaMemcpyAsync(vrtx::rayCounter, &num_blocks, sizeof(int32_t), cudaMemcpyHostToDevice, stream);
                     LAUNCH(vrtx::render::sphIntersection::raySchedulerIntersectFluidStaticParticles, blocks_1080,
                            blockSize_1080, 0, stream)(!get<parameters::render_settings::vrtxRenderGrid>()););
      }
    }
    MEASURE_CUDA(bounceRays, stream,
                 LAUNCH(vrtx::render::shading::shadeAndBounceRays, launchDim(rayCount, 512), 512, 0, stream)(
                     dist(rd),
                     get<parameters::render_settings::vrtxRenderBVH>() && get<parameters::boundary_volumes::volume>().size() > 0 ? 1 : 0, bvhs););

    if (get<parameters::render_settings::vrtxRenderFluid>()) {
      timers[std::string("03Intersection  ")].push_back((float)intersectFluid());
      if (get<parameters::render_settings::vrtxRenderSurface>())
        timers[std::string("04Fluid Normal  ")].push_back((float)calculateNormals());
    }
    timers[std::string("05Shade & Bounce")].push_back((float)bounceRays());
    // timers[std::string("collision ctr")].push_back(0);
    // if (get<parameters::render_settings::vrtxRenderNormals>() == 1)
    //   break;
  }
  if (get<parameters::render_settings::vrtxRenderNormals>()) {
    MEASURE_CUDA(toneMap, stream,
                 LAUNCH(vrtx::common::toneMapNormals, griddim, blockdim, 0, stream)(framenumber, (float3 *)acc,
                                                                                    vrtx::cuImage, 1.f););
  } else {
    MEASURE_CUDA(
        toneMap, stream,
        LAUNCH(vrtx::common::toneMap, griddim, blockdim, 0, stream)(framenumber, (float3 *)acc, vrtx::cuImage, 1.f););
  }
  timers[std::string("06Tone mapping  ")].push_back(toneMap());

  // cuda::sync(std::to_string(__LINE__));
  cudaStreamSynchronize(stream);
  cuda::sync("end of vrtx step");
  return timers;
}