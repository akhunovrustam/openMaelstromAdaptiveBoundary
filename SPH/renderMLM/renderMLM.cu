#include <SPH/renderMLM/renderMLM.cuh>
#include <utility/include_all.h>
namespace SPH {
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
hostDeviceInline auto iCoord(float4 p, const SPH::renderMLM::Memory &arrays) {
  return getVoxel(p, arrays.min_coord, arrays.cell_size);
}
hostDeviceInline auto iCoord(float3 p, const SPH::renderMLM::Memory &arrays) {
  return getVoxel(p, arrays.min_coord, arrays.cell_size);
}
hostDeviceInline auto zCoord(int3 idx) { return i3toZ(idx); }
hostDeviceInline auto zCoord(uint3 idx) { return ui3toZ(idx); }
hostDeviceInline auto zCoord(float4 p, const SPH::renderMLM::Memory &memory) { return zCoord(iCoord(p, memory)); }
hostDeviceInline auto zCoord(float3 p, const SPH::renderMLM::Memory &memory) { return zCoord(iCoord(p, memory)); }
hostDeviceInline auto hCoord(uint3 idx, const SPH::renderMLM::Memory &arrays) {
  return zCoord(idx) % arrays.hash_entries;
}
hostDeviceInline auto hCoord(int3 idx, const SPH::renderMLM::Memory &arrays) {
  return zCoord(idx) % arrays.hash_entries;
}
hostDeviceInline auto hCoord(float4 p, const SPH::renderMLM::Memory &arrays) {
  return zCoord(iCoord(p, arrays)) % arrays.hash_entries;
}
hostDeviceInline auto hCoord(float3 p, const SPH::renderMLM::Memory &arrays) {
  return zCoord(iCoord(p, arrays)) % arrays.hash_entries;
}

namespace renderMLM {
basicFunctionType cudetectFluidSurface(SPH::renderMLM::Memory arrays) {
  checkedParticleIdx(i);
  auto x_i = arrays.position[i];
  auto xB_i = arrays.centerPosition[i];
  // auto neighs = arrays.neighborListLength[i];
  auto d = math::distance3(xB_i, x_i) - arrays.radius;
  //auto surface = d > -0.f * arrays.radius ? true : false;
  //if (neighs < 15)
  //  surface = true;
  //if (neighs > arrays.internalLimit)
  //  surface = false;
  //if (neighs > kernelNeighbors() * 1.2f)
  //  surface = false;
  arrays.auxDistance[i] = arrays.auxTest[i] < 0.f ? -1.f : 1.f;
}

basicFunctionType cudaHashTablea(SPH::renderMLM::Memory arrays, int32_t threads) {
  checkedThreadIdx(i);
  auto h = arrays.resortIndex[i];
  if (i == 0 || h != arrays.resortIndex[i - 1]) {
    arrays.compactHashMap[h].beginning = i;
  }
}
basicFunctionType cudaHashTableb(SPH::renderMLM::Memory arrays, int32_t threads) {
  checkedThreadIdx(i);
  auto h = arrays.resortIndex[i];
  if (i == threads - 1 || arrays.resortIndex[i + 1] != arrays.resortIndex[i]) {
    arrays.compactHashMap[h].length = i - arrays.compactHashMap[h].beginning + 1;
  }
}
basicFunctionType cudaCellTablea(SPH::renderMLM::Memory arrays, int32_t threads, int32_t *compact) {
  checkedThreadIdx(i);
  arrays.compactCellSpan[i] =
      compact_cellSpan{0, (uint32_t)compact[i], math::min(INVALID_LEN, (uint32_t)(compact[i + 1] - compact[i]))};
  arrays.auxLength[i] = (uint32_t)(compact[i + 1] - compact[i]);
}
basicFunctionType cudaCellTableb(SPH::renderMLM::Memory arrays, int32_t threads, int32_t *compact, float4 anisoH) {
  checkedThreadIdx(i);
  auto x_i = arrays.centerPosition[compact[i]];
  arrays.resortIndex[i] = hCoord(x_i, arrays);
  arrays.particleparticleIndex[i] = i;
}
cellFunctionType cudaHashParticles(SPH::renderMLM::Memory arrays, float4 anisoH) {
  checkedParticleIdx(i);
  auto x_i = arrays.centerPosition[i];
  arrays.ZOrder_32[i] = zCoord(x_i, arrays);
  arrays.resortIndex[i] = hCoord(x_i, arrays);
  arrays.particleparticleIndex[i] = i;
}
cellFunctionType cudaIndexCells(SPH::renderMLM::Memory arrays, int32_t threads, int32_t *cell_indices) {
  checkedThreadIdx(i);
  if (i == 0)
    cell_indices[0] = 0;
  i++;
  cell_indices[i] = i == arrays.num_ptcls || arrays.ZOrder_32[i - 1] != arrays.ZOrder_32[i] ? i : -1;
}
basicFunctionType compactHashMap(SPH::renderMLM::Memory arrays, int32_t threads) {
  checkedThreadIdx(i);
  auto h = arrays.resortIndex[i];
  auto hashEntry = arrays.compactHashMap[h];
  auto cell = arrays.compactCellSpan[i];
  if (hashEntry.length == 1 && cell.length != INVALID_LEN) {
    arrays.compactHashMap[h] = compactHashSpan{1u, cell.beginning, cell.length};
    arrays.compactCellSpan[i] = compactHashSpan{1u, cell.beginning, cell.length};
  } else
    arrays.compactHashMap[h] = compactHashSpan{0u, hashEntry.beginning, hashEntry.length};
}
template <typename T>
hostDeviceInline void cudaSortCompactmlm(SPH::renderMLM::Memory arrays, int32_t threads, T *input, T *output, T *copy) {
  checkedThreadIdx(i);
  auto in = input[arrays.particleparticleIndex[i]];
  if (copy != nullptr)
    copy[i] = output[i];
  output[i] = in;
}

basicFunction(detectSurface, cudetectFluidSurface, "detecting surface");
cellFunction(hashParticles, cudaHashParticles, "hashing particles", caches<float4>{});
basicFunction(buildCellTable1, cudaCellTablea, "creating cell table I");
basicFunction(buildCellTable2, cudaCellTableb, "creating cell table II");
basicFunction(buildHashTable1, cudaHashTablea, "hashing cell table I");
basicFunction(buildHashTable2, cudaHashTableb, "hashing cell table II");
cellFunction(indexCells, cudaIndexCells, "indexing cells");
basicFunction(sort, cudaSortCompactmlm, "compact resorting cells");
basicFunction(compact, compactHashMap, "compact hashmap");

struct is_valid {
  hostDeviceInline bool operator()(const int x) { return x != -1; }
};
struct count_if {
  hostDeviceInline bool operator()(const compactHashSpan x) { return x.beginning != INVALID_BEG && x.length > 1; }
};
struct invalid_position {
  hostDeviceInline bool operator()(float4 x) {
    return x.w == FLT_MAX || x.x != x.x || x.y != x.y || x.z != x.z || x.w != x.w;
  }
};
struct invalid_volume {
  hostDeviceInline bool operator()(float x) { return x == FLT_MAX || x == 0.f || x != x; }
};
struct hash_spans {
  hostDeviceInline compactHashSpan operator()() { return compactHashSpan{0, INVALID_BEG, INVALID_LEN}; }
};
} // namespace renderMLM

namespace auxMLM {
basicFunctionType cuAuxHashTableA(SPH::renderMLM::Memory arrays, int32_t threads) {
  checkedThreadIdx(i);
  auto h = (arrays.auxCellSpan[i].cell.hash % arrays.hash_entries);
  if (i == 0 || h != (arrays.auxCellSpan[i - 1].cell.hash % arrays.hash_entries)) {
    arrays.auxHashMap[h].hash.beginning = i;
  }
}
basicFunctionType cuAuxHashTableB(SPH::renderMLM::Memory arrays, int32_t threads) {
  checkedThreadIdx(i);
  auto h = (arrays.auxCellSpan[i].cell.hash % arrays.hash_entries);
  if (i == threads - 1 || (arrays.auxCellSpan[i + 1].cell.hash % arrays.hash_entries) != h)
    arrays.auxHashMap[h].hash.length = i - arrays.auxHashMap[h].hash.beginning + 1;
}
cellFunctionType cuCompactCellTable(SPH::renderMLM::Memory arrays, int32_t threads, int32_t *compact, float4 anisoH) {
  checkedThreadIdx(i);
  auto xi = arrays.centerPosition[compact[i]];
  auto z = zCoord(xi, arrays);
  uint32_t h = hCoord(xi, arrays);
  // auto s = arrays.compactHashMap[h];
  // auto cs = compact_cellSpan{0,0, 0};
  auto surfaceCell = arrays.pruneVoxel == 1 ? false : true;

  // if (s.compacted == 1)
  //  cs = compact_cellSpan{0,s.beginning, s.length};
  // else
  //  for (auto ci = s.beginning; ci < s.beginning + s.length; ++ci)
  //    if (zCoord(arrays.centerPosition[arrays.compactCellSpan[ci].beginning], arrays) == z)
  //      cs = arrays.compactCellSpan[ci];

  //// if (xi.y < 82.f)
  ////  surfaceCell = false;
  //// else
  // for (auto j = cs.beginning; j < cs.beginning + cs.length; ++j) {
  //  // printf("%d[%d] : %d -> %f => %d\n", i, compact[i], j, arrays.auxDistance[j], surfaceCell ? 1 : 0);
  //  // surfaceCell = true;
  //  // if (compact[i] == j)
  //  // surfaceCell = true;
  //  if (arrays.auxDistance[j] > 0.f)
  //    surfaceCell = true;
  //}

  uint32_t pred = surfaceCell ? LIST_COMPACT : LIST_ALWAYS_FALSE;
  arrays.auxCellSpan[i].cell = compactListEntry::cellEntry{pred, (uint32_t)z};
  arrays.resortIndex[i] = h;
  arrays.particleparticleIndex[i] = i;
}
cellFunctionType cuAuxIndexCells(SPH::renderMLM::Memory arrays, int32_t threads, int32_t *cell_indices) {
  // checkedThreadIdx(i);
  checkedThreadIdx(i);
  if (i == 0)
    cell_indices[0] = 0;
  i++;
  cell_indices[i] = i == arrays.num_ptcls || arrays.ZOrder_32[i - 1] != arrays.ZOrder_32[i] ? i : -1;

  // if (i == 0)
  //  cell_indices[0] = -1;
  // cell_indices[i + 1] = -1;

  // if (i == 0)
  //  cell_indices[0] = 0;
  // i++;
  // cell_indices[i] = -1;
  // if (i == arrays.num_ptcls)
  //  cell_indices[i] = i;
  // else {
  //  if (arrays.ZOrder_32[i - 1] == arrays.ZOrder_32[i])
  //    return;
  //  // if (surface)
  //  cell_indices[i] = i;
  //}
  // cell_indices[i] = i == arrays.num_ptcls || arrays.ZOrder_32[i - 1] != arrays.ZOrder_32[i] ? i : -1;
}
cellFunctionType cuAuxSpreadCells(SPH::renderMLM::Memory arrays, int32_t threads, int32_t *cell_indices) {
  checkedThreadIdx(i);

  if (i == 0)
    cell_indices[0] = 0;
  i++;
  cell_indices[i] = i == arrays.num_ptcls || arrays.ZOrder_32[i - 1] != arrays.ZOrder_32[i] ? i : -1;
}

basicFunctionType cuAuxCompactHashTable(SPH::renderMLM::Memory arrays, int32_t threads) {
  checkedThreadIdx(i);
  auto h = arrays.auxCellSpan[i].cell.hash % arrays.hash_entries;
  auto hashEntry = arrays.auxHashMap[h];
  hashEntry.hash.kind = LIST_ITERATE;
  auto cell = arrays.auxCellSpan[i];
  if (hashEntry.hash.length == 1)
    arrays.auxHashMap[h] = cell;
  else
    arrays.auxHashMap[h] = hashEntry;
}
__device__ __host__ uint32_t cLEasuint(compactListEntry cle) { return *reinterpret_cast<uint32_t *>(&cle); }
basicFunctionDeviceType spreadCells(SPH::renderMLM::Memory arrays, int32_t threads, float4 anisoH) {
  checkedThreadIdx(i);
  auto cell = arrays.compactCellSpan[i];
  auto xi = arrays.centerPosition[cell.beginning];
  auto z = zCoord(xi, arrays);
  uint32_t h = hCoord(xi, arrays);
  auto s = arrays.compactHashMap[h];
  auto cs_beg = 0u;
  auto cs_len = 0u;
  if (s.compacted == 1) {
    cs_beg = s.beginning;
    cs_len = s.length == INVALID_LEN ? arrays.auxLength[i] : s.length;
  } else
    for (auto ci = s.beginning; ci < s.beginning + s.length; ++ci)
      if (zCoord(arrays.centerPosition[arrays.compactCellSpan[ci].beginning], arrays) == z) {
        auto cs = arrays.compactCellSpan[ci];
        cs_beg = cs.beginning;
        cs_len = cs.length == INVALID_LEN ? arrays.auxLength[ci] : cs.length;
      }
  auto surfaceCell = arrays.pruneVoxel == 1 ? false : true;
  for (uint32_t j = cs_beg; j < cs_beg + cs_len; ++j)
    if (arrays.auxDistance[j] > 0.f)
      surfaceCell = true;
  // return;
  if (!surfaceCell)
    return;

  float4 minAABB{FLT_MAX, FLT_MAX, FLT_MAX, 0.f};
  float4 maxAABB{-FLT_MAX, -FLT_MAX, -FLT_MAX, 0.f};
  auto len = cs_len;

  for (uint32_t j = cs_beg; j < cs_beg + cs_len; ++j) {
#ifdef ANISOTROPIC_SURFACE
    float4 support = arrays.anisotropicSupport[j];
#else
    float4 support = anisoH;
#endif
    float4 pos = arrays.centerPosition[j];
    if (len < 15)
      support = float4{1.f, 1.f, 1.f, 1.f} * arrays.radius * 2.f;
    minAABB = math::min(pos - support, minAABB);
    maxAABB = math::max(pos + support, maxAABB);
  }
  float4 center = (maxAABB + minAABB) * 0.5f;
  float4 halfSize = (maxAABB - minAABB) * 0.5f;

  auto voxel = iCoord(xi, arrays);
  auto cellCenter = arrays.min_coord +
                    float3{(float)voxel.x * anisoH.x, (float)voxel.y * anisoH.y, (float)voxel.z * anisoH.z} +
                    float3{anisoH.x, anisoH.y, anisoH.z} * 0.5f;
  for (int32_t i = -1; i <= 1; ++i) {
    for (int32_t j = -1; j <= 1; ++j) {
      for (int32_t k = -1; k <= 1; ++k) {
        int3 localVoxel = voxel + int3{i, j, k};
        auto zN = zCoord(localVoxel);
        auto hN = hCoord(localVoxel, arrays);
        auto cell = cellCenter + float3{(float)i * anisoH.x, (float)j * anisoH.y, (float)k * anisoH.z};

        bool x = fabsf(center.x - cell.x) <= (halfSize.x + anisoH.x * 0.5f) * 0.9f;
        bool y = fabsf(center.y - cell.y) <= (halfSize.y + anisoH.y * 0.5f) * 0.9f;
        bool z = fabsf(center.z - cell.z) <= (halfSize.z + anisoH.z * 0.5f) * 0.9f;
        bool intersect = x && y && z;
        if (!intersect)
          continue;

        auto auxHash = arrays.auxHashMap[hN];
        if (auxHash.cell.kind == LIST_COMPACT && auxHash.cell.hash == zN)
          continue;
        bool found = false;
        if (auxHash.cell.kind == LIST_ITERATE)
          for (auto cs = auxHash.hash.beginning; cs < auxHash.hash.beginning + auxHash.hash.length; ++cs)
            if (arrays.auxCellSpan[cs].cell.hash == zN)
              found = true;
        if (found)
          continue;
        //if (auxHash.cell.kind == LIST_COMPACT && auxHash.cell.hash == zN)
        //  continue;

        // auto s = arrays.compactHashMap[hN];
        // auto csn = compact_cellSpan{0, 0};
        // if (s.compacted == 1 && zCoord(arrays.position[s.beginning],arrays))
        //  csn = compact_cellSpan{(int32_t)s.beginning, s.length};
        // else
        //  for (int32_t ci = (int32_t)s.beginning; ci < ((int32_t)s.beginning) + s.length; ++ci)
        //    if (zCoord(arrays.centerPosition[arrays.compactCellSpan[ci].beginning], arrays) == z)
        //      cs = arrays.compactCellSpan[ci];
        // if (cs.length > arrays.internalLimit)
        //  continue;

        uint32_t *address_as_ul = (uint32_t *)(arrays.auxHashMap + hN);
        uint32_t old = cLEasuint(arrays.auxHashMap[hN]), assumed;
        do {
          assumed = old;
          compactListEntry new_value = arrays.auxHashMap[hN];
          if (new_value.cell.kind != LIST_ALWAYS_FALSE)
            new_value.cell = compactListEntry::cellEntry{LIST_ALWAYS_TRUE, (uint32_t) zN};
          else
            new_value.cell = compactListEntry::cellEntry{LIST_COMPACT, (uint32_t)zN};
          old = atomicCAS(address_as_ul, assumed, cLEasuint(new_value));
        } while (old != assumed);
      }
    }
  }
}
struct count_if {
  hostDeviceInline bool operator()(const compactListEntry x) {
    return x.cell.kind == LIST_ITERATE && x.hash.beginning != INVALID_BEG;
  }
};
struct is_valid {
  hostDeviceInline bool operator()(const int x) { return x != -1; }
};
struct is_valid_cell {
  hostDeviceInline bool operator()(const compactListEntry x) { return x.hash.kind != LIST_ALWAYS_FALSE; }
};
struct invalid_position {
  hostDeviceInline bool operator()(float4 x) { return x.w == FLT_MAX; }
};
struct hash_spans {
  hostDeviceInline compactListEntry operator()() { return compactListEntry{LIST_ALWAYS_FALSE, INVALID_BEG, 0}; }
};

basicFunctionDevice(spread, spreadCells, "spread cells");

cellFunction(buildCellTable, cuCompactCellTable, "creating cell table I");
basicFunction(buildHashTable1, cuAuxHashTableA, "hashing cell table I");
basicFunction(buildHashTable2, cuAuxHashTableB, "hashing cell table II");
cellFunction(indexCells, cuAuxIndexCells, "indexing cells");
basicFunction(compact, cuAuxCompactHashTable, "compact hashmap");

hostDeviceInline auto square(float x) { return x * x; }
hostDeviceInline auto cube(float x) { return x * x * x; }
hostDeviceInline float k(float4 x_i, float4 x_j) {
  auto h = (x_i.w + x_j.w) * 0.5f * kernelSize();
  auto d = math::distance3(x_i, x_j);
  auto s = d / h;
  return math::max(0.f, cube(1.f - square(s)));
}

neighFunctionType estimateSurface(SPH::renderMLM::Memory arrays, float *isoDensity) {
  checkedParticleIdx(i);
  cache_arrays((pos, position), (vol, volume));
  float4 normal{0.f, 0.f, 0.f, 0.f};
  auto numNeighs = 0;
  iterateNeighbors(j) {
    if (i == j)
      continue;
    auto distance = pos[i] - pos[j];
    normal += math::normalize3(distance);
    auto w_ij = W_ij;
    if (w_ij > 0.f)
      numNeighs += 1;
  }
  normal = math::normalize3(normal);
  bool state = false;
  iterateNeighbors(j) {
    if (i == j)
      continue;
    auto distance = arrays.position[j] - arrays.position[i];
    auto angle = acosf(math::dot3(normal, math::normalize3(distance)));
    state = state || angle <= CUDART_PI_F / 6.f;
  }
  auto phi = state ? -20.f : 0.f;
  arrays.auxDistance[i] = phi;
#ifndef ANISOTROPIC_SURFACE
  arrays.centerPosition[i] = arrays.position[i];
  isoDensity[i] = arrays.volume[i];
#endif
  // arrays.auxIsoDensity[i] = isoDensity;
}
neighFunctionType predicateParticles(SPH::renderMLM::Memory arrays) {
  checkedParticleIdx(i);
  int32_t max_neighs = INT_MIN;
  iterateNeighbors(j) { max_neighs = math::max(max_neighs, arrays.neighborListLength[j]); }
  if (/*max_neighs >= arrays.vrtxNeighborLimit && */ arrays.auxDistance[i] > -5.f)
    arrays.auxTest[i] = 1.f;
  else if (max_neighs < arrays.vrtxNeighborLimit)
    arrays.auxTest[i] = 0.f;
  else
    arrays.auxTest[i] = -1.f;
}

neighFunctionDeviceType spreadSurfaceInformation(SPH::renderMLM::Memory arrays, uint32_t *atomicCtr) {
  checkedParticleIdx(i);
  float auxTi = arrays.auxTest[i];
  if (auxTi > 1.f || auxTi < FLT_MIN)
    return;
  auto x_i = arrays.position[i];
  iterateNeighbors(j) {
    float auxTj = arrays.auxTest[j];
    if (auxTj == 0.f) {
      auto x_j = arrays.position[j];
      if (math::distance3(x_i, x_j) < support(x_i, x_j)) {
        arrays.auxTest[j] = 0.5f;
        atomicInc(atomicCtr, UINT32_MAX);
      }
    }
  }
  arrays.auxTest[i] = 2.f;
}
neighFunction(predication, predicateParticles, "estimating surface");
neighFunctionDevice(spreadSurface, spreadSurfaceInformation, "spreading surface");
neighFunction(surfaceEstimate, estimateSurface, "estimating surface", caches<float4, float>{});

} // namespace auxMLM

} // namespace SPH

void SPH::renderMLM::generateAuxilliaryGrid(Memory mem) {
  auto n = mem.num_ptcls;
  launch<auxMLM::surfaceEstimate>(mem.num_ptcls, mem, arrays::auxIsoDensity::ptr);
  cuda::sync();
  launch<auxMLM::predication>(mem.num_ptcls, mem);
  cuda::sync();
  static uint32_t *atomicCtr = nullptr;
  static uint32_t hostCtr;
  static bool once = true;
  if (once) {
    cudaMalloc(&atomicCtr, sizeof(uint32_t));
    once = false;
  }
  do {
    cudaMemset(atomicCtr, 0x00, sizeof(uint32_t));
    cuda::sync();
    launchDevice<auxMLM::spreadSurface>(mem.num_ptcls, mem, atomicCtr);
    cuda::sync();
    cudaMemcpy(&hostCtr, atomicCtr, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cuda::sync();
  } while (hostCtr > 0);
  cuda::sync();
  launch<detectSurface>(mem.num_ptcls, mem);
  cuda::sync();
#ifdef ANISOTROPIC_SURFACE
  float4 maxAnisotropicSupport =
      thrust::reduce(thrust::device, mem.anisotropicSupport, mem.anisotropicSupport + n,
                     float4{-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX},
                     [] __host__ __device__(const float4 &lhs, const float4 &rhs) { return math::max(lhs, rhs); });
#else
  float4 maxAnisotropicSupport =
      get<parameters::render_settings::anisotropicKs>() * get<parameters::internal::ptcl_support>() * kernelSize() * float4{1.f, 1.f, 1.f, 1.f};
#endif
  auto mm = math::max(maxAnisotropicSupport.x, math::max(maxAnisotropicSupport.y, maxAnisotropicSupport.z));
  mem.cell_size = float3{maxAnisotropicSupport.x, maxAnisotropicSupport.y, maxAnisotropicSupport.z};
  get<parameters::render_settings::maxAnisotropicSupport>() = maxAnisotropicSupport;
  {
    //*(int *)0 = 0;
    cuda::sync();
    launch<hashParticles>(mem.num_ptcls, mem, maxAnisotropicSupport);
    cuda::sync();

    algorithm::sort_by_key(mem.num_ptcls, mem.ZOrder_32, mem.particleparticleIndex);
    cuda::sync();
    float4 *r4Array = arrays::resortArray4::ptr;
    float *rArray = arrays::resortArray::ptr;
    float *iArray = arrays::anisotropicMatrices::ptr;
#ifdef ANISOTROPIC_SURFACE
    launch<sort>(n, mem, n, mem.anisotropicSupport, r4Array, (float4 *)nullptr);
    launch<sort>(n, mem, n, arrays::renderArray::ptr, r4Array, mem.anisotropicSupport);
    launch<sort>(n, mem, n, mem.centerPosition, r4Array, arrays::renderArray::ptr);
    cuda::memcpy(mem.centerPosition, r4Array, sizeof(float4) * n, cudaMemcpyDeviceToDevice);
    cuda::sync();
    launch<sort>(n, mem, n, arrays::volume::ptr, mem.auxIsoDensity, (float *)nullptr);
    launch<sort>(n, mem, n, mem.auxDistance, rArray, arrays::density::ptr);
    launch<sort>(n, mem, n, iArray, rArray, mem.auxDistance);
    launch<sort>(n, mem, n, iArray + 1 * n, rArray, iArray + 0 * n);
    launch<sort>(n, mem, n, iArray + 2 * n, rArray, iArray + 1 * n);
    launch<sort>(n, mem, n, iArray + 3 * n, rArray, iArray + 2 * n);
    launch<sort>(n, mem, n, iArray + 4 * n, rArray, iArray + 3 * n);
    launch<sort>(n, mem, n, iArray + 5 * n, rArray, iArray + 4 * n);
    launch<sort>(n, mem, n, iArray + 6 * n, rArray, iArray + 5 * n);
    launch<sort>(n, mem, n, iArray + 7 * n, rArray, iArray + 6 * n);
    launch<sort>(n, mem, n, iArray + 8 * n, rArray, iArray + 7 * n);
    cuda::memcpy(iArray + 8 * n, rArray, sizeof(float) * n, cudaMemcpyDeviceToDevice);
    cuda::sync();
#else
    launch<sort>(n, mem, n, arrays::renderArray::ptr, r4Array, (float4 *)nullptr);
    cuda::sync();
    launch<sort>(n, mem, n, mem.centerPosition, r4Array, arrays::renderArray::ptr);
    cuda::sync();
    cuda::memcpy(mem.centerPosition, r4Array, sizeof(float4) * n, cudaMemcpyDeviceToDevice);
    cuda::sync();

    launch<sort>(n, mem, n, arrays::auxIsoDensity::ptr, rArray, (float *)nullptr);
    cuda::sync();
    launch<sort>(n, mem, n, arrays::density::ptr, rArray, arrays::auxIsoDensity::ptr);
    cuda::sync();
    launch<sort>(n, mem, n, mem.auxDistance, rArray, arrays::density::ptr);
    cuda::sync();
    cuda::memcpy(mem.auxDistance, rArray, sizeof(float) * n, cudaMemcpyDeviceToDevice);
    cuda::sync();
#endif
  }
  get<parameters::resort::auxCells>() = 0;
  get<parameters::resort::auxCollisions>() = 0;
  compactHashSpan *hashMap = arrays::compactHashMap::ptr;
  compact_cellSpan *cellSpan = arrays::compactCellSpan::ptr;
  mem.compactCellSpan = cellSpan;
  mem.compactHashMap = hashMap;
  cuda::arrayMemset<arrays::cellparticleIndex>(0xFFFFFFFF);
  cuda::arrayMemset<arrays::ZOrder_32>(0xFFFFFFFF);
  cuda::sync();
  launch<hashParticles>(mem.num_ptcls, mem, maxAnisotropicSupport);
  cuda::sync();
  algorithm::generate(hashMap, mem.hash_entries, hash_spans());
  cuda::sync();
  launch<indexCells>(mem.num_ptcls, mem, mem.num_ptcls, arrays::cellparticleIndex::ptr);
  cuda::sync();
  int32_t diff = static_cast<int32_t>(algorithm::copy_if(
      arrays::cellparticleIndex::ptr, arrays::compactparticleIndex::ptr, mem.num_ptcls + 1, is_valid()));
  cuda::sync();
  launch<buildCellTable1>(diff, mem, diff, arrays::compactparticleIndex::ptr);
  cuda::sync();
  launch<buildCellTable2>(diff, mem, diff, arrays::compactparticleIndex::ptr, maxAnisotropicSupport);
  cuda::sync();
  diff--;
  get<parameters::resort::auxCells>() = diff;
  algorithm::sort_by_key(diff, mem.resortIndex, mem.particleparticleIndex);
  cuda::sync();
  launch<sort>(diff, mem, diff, cellSpan, mem.compactCellSpanSwap, (compact_cellSpan *)nullptr);
  launch<sort>(diff, mem, diff, mem.auxLength, (uint32_t *)arrays::resortArray::ptr, (uint32_t *)nullptr);
  cudaMemcpy(mem.auxLength, arrays::resortArray::ptr, sizeof(uint32_t) * diff, cudaMemcpyDeviceToDevice);
  cuda::sync();
  cuda::memcpy(cellSpan, mem.compactCellSpanSwap, sizeof(compact_cellSpan) * diff, cudaMemcpyDeviceToDevice);
  cuda::sync();
  launch<buildHashTable1>(diff, mem, diff);
  cuda::sync();
  launch<buildHashTable2>(diff, mem, diff);
  cuda::sync();
  launch<compact>(diff, mem, diff);
  get<parameters::render_settings::auxCellCount>() = diff;
  get<parameters::resort::auxCollisions>() = (int32_t) algorithm::count_if(mem.compactHashMap, diff, count_if());

  //{
  //  compactHashSpan *cellSpanHost = new compactHashSpan[diff + 1];
  //  // compactHashSpan *hashMapHost = new compactHashSpan[diff + 1];
  //  cudaMemcpy(cellSpanHost, cellSpan, sizeof(compactHashSpan) * (diff + 1), cudaMemcpyDeviceToHost);
  //  std::cout << "########################################################################################"
  //            << std::endl;
  //  for (int32_t i = 0; i < diff + 1; ++i) {
  //    std::cout << cellSpanHost[i].compacted << " : " << cellSpanHost[i].beginning << " -> " << cellSpanHost[i].length
  //              << std::endl;
  //  }

  //  delete[] cellSpanHost;
  //  // delete[] hashMapHost;
  //}

  cuda::arrayMemset<arrays::cellparticleIndex>(0xFFFFFFFF);
  cuda::arrayMemset<arrays::ZOrder_32>(0xFFFFFFFF);

  launch<hashParticles>(mem.num_ptcls, mem, maxAnisotropicSupport);
  cuda::sync();
  algorithm::generate(mem.auxHashMap, mem.hash_entries, auxMLM::hash_spans());
  cuda::sync();
  launch<SPH::auxMLM::indexCells>(mem.num_ptcls, mem, mem.num_ptcls, arrays::cellparticleIndex::ptr);
  cuda::sync();
  diff = static_cast<int32_t>(algorithm::copy_if(arrays::cellparticleIndex::ptr, arrays::compactparticleIndex::ptr,
                                                 mem.num_ptcls + 1, is_valid()));
  cuda::sync();
  launch<SPH::auxMLM::buildCellTable>(diff, mem, diff, arrays::compactparticleIndex::ptr, maxAnisotropicSupport);
  diff--;
  algorithm::sort_by_key(diff, mem.resortIndex, mem.particleparticleIndex);
  cuda::sync();
  launch<sort>(diff, mem, diff, mem.auxCellSpan, (compactListEntry *)mem.compactCellSpanSwap,
               (compactListEntry *)nullptr);
  diff = static_cast<int32_t>(algorithm::copy_if((compactListEntry *)mem.compactCellSpanSwap, mem.auxCellSpan, diff,
                                                 SPH::auxMLM::is_valid_cell()));
  cuda::sync();
  launch<auxMLM::buildHashTable1>(diff, mem, diff);
  cuda::sync();
  launch<auxMLM::buildHashTable2>(diff, mem, diff);
  cuda::sync();
  launch<auxMLM::compact>(diff, mem, diff);
  launchDevice<auxMLM::spread>(get<parameters::resort::auxCells>(), mem, get<parameters::resort::auxCells>(), maxAnisotropicSupport);

  launch<hashParticles>(mem.num_ptcls, mem, maxAnisotropicSupport);
}