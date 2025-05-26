#include "math.h" // CUDA math library
#include <SPH/kernelSupport/anisotropy.cuh>
#include <algorithm>
#include <cuda.h>
#include <sstream>
#include <math/SVD.h>
#include <utility/include_all.h>
//#define HSCALE 1.9f
//#define INTERPOLATOR_SCALE ::ceilf(HSCALE)
namespace SPH {
namespace anisotropy {
struct interpolateMLM {
  int3 central_idx;
  int32_t scale;
  bool full_loop;
  Memory &arrays;

  hostDevice interpolateMLM(float4 position, Memory &memory, float hScale)
      : arrays(memory), scale((int32_t)::floorf(hScale) + 1) {
    central_idx = position_to_idx3D_i(position, memory.min_coord, memory.cell_size.x);
  }

  struct cell_iterator {
    int3 idx;
    int32_t scale = 1;
    int32_t i = -scale, j = -scale, k = -scale;
    uint32_t ii = 0;
    int32_t jj = 0;

    int32_t neighbor_idx;

    Memory &arrays;
    compactHashSpan s{0, INVALID_BEG, 0};
    compact_cellSpan cs{0, 0};

    hostDevice int32_t cs_loop() {
      if (cs.beginning != INVALID_BEG && jj < (int32_t)(cs.beginning + cs.length)) {
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
        auto morton = idx3D_to_morton(cell);
        if (s.compacted && ii < s.beginning + s.length) {
          cs = compact_cellSpan{0, s.beginning, s.length};
          jj = cs.beginning;
          ii = s.beginning + s.length;
          if (position_to_morton(arrays.position[cs.beginning], arrays, 1.f) == morton) {
            if (cs_loop() != -1) {
              return neighbor_idx;
            }
          }
        }
        for (; ii < s.beginning + s.length;) {
          cs = arrays.compactCellSpan[ii];
          ++ii;
          jj = cs.beginning;
          if (position_to_morton(arrays.position[cs.beginning], arrays, 1.f) == morton) {
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

      for (; i <= scale; ++i) {
        for (; j <= scale; ++j) {
          for (; k <= scale;) {
            uint3 cell = uint3{static_cast<uint32_t>(idx.x + i), static_cast<uint32_t>(idx.y + j),
                               static_cast<uint32_t>(idx.z + k)};
            if (cell.x == UINT32_MAX || cell.y == UINT32_MAX || cell.z == UINT32_MAX) {
              ++k;
              continue;
            }
            auto morton = idx3D_to_morton(cell);

            s = arrays.compactHashMap[idx3D_to_hash(cell, arrays.hash_entries)];
            ii = s.beginning;
            if (s.beginning == INVALID_BEG) {
              ++k;
              continue;
            }
            if (s_loop() != -1)
              return;
          }
          k = -scale;
        }
        j = -scale;
      }
    }

    hostDevice cell_iterator(int3 c_idx, Memory &memory, int32_t hscale, int32_t _i = -1, int32_t _j = -1,
                             int32_t _k = -1)
        : idx(c_idx), scale(hscale), i(_i), j(_j), k(_k), arrays(memory) {
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

  hostDeviceInline cell_iterator begin() const {
    return cell_iterator(central_idx, arrays, scale, -scale, -scale, -scale);
  }
  hostDeviceInline cell_iterator end() const {
    return cell_iterator(central_idx, arrays, scale, scale + 1, scale + 1, scale + 1);
  }
  hostDeviceInline cell_iterator cbegin() const {
    return cell_iterator(central_idx, arrays, scale, -scale, -scale, -scale);
  }
  hostDeviceInline cell_iterator cend() const {
    return cell_iterator(central_idx, arrays, scale, scale + 1, scale + 1, scale + 1);
  }
};
template <typename T> __device__ __host__ __inline__ auto square(T &&x) { return x * x; }
template <typename T> __device__ __host__ __inline__ auto cube(T &&x) { return x * x * x; }
__device__ __host__ __inline__ float k(float4 x_i, float4 x_j, float scale = 1.f) {
  auto h = (x_i.w + x_j.w) * 0.5f * kernelSize() * scale;
  auto d = math::distance3(x_i, x_j);
  auto s = d / h;
  return math::max(0.f, 1.f - cube(s));
}

neighFunctionType calculateCenterPosition(Memory arrays, float hScale) {
  checkedParticleIdx(i);
  cache_arrays((pos, position), (vol, volume));
  float4 positionSum{0.f, 0.f, 0.f, 0.f};
  float kernelSum = 0.f;
  for (int32_t j : interpolateMLM(pos[i], arrays, hScale)) {
    // iterateNeighbors(j){
    float4 x_j = pos[j];
    float w_ij = k(pos[i], x_j, hScale);
    positionSum += pos[j] * w_ij;
    kernelSum += w_ij;
  }
  positionSum /= kernelSum;
  positionSum.w = kernelSum;
  arrays.centerPosition[i] = positionSum;
}

#define G(i, x, y) (arrays.anisotropicMatrices[i + (x * 3 + y) * arrays.num_ptcls])
neighFunctionType calculateCovarianceMatrix(Memory arrays, float hScale) {
  checkedParticleIdx(i);
  cache_arrays((pos, position), (vol, volume));
  float4 x_iW = arrays.centerPosition[i];
  x_iW.w = pos[i].w;
  float c_00 = 0.f, c_01 = 0.f, c_02 = 0.f;
  float c_10 = 0.f, c_11 = 0.f, c_12 = 0.f;
  float c_20 = 0.f, c_21 = 0.f, c_22 = 0.f;
  float ctr = 0.f;
  float kernelSum = 0.f;
  for (int32_t j : interpolateMLM(x_iW, arrays, hScale)) {
    // iterateNeighbors(j) {
    float4 x_j = pos[j];
    float w_ij = k(x_iW, x_j, hScale);
    kernelSum += w_ij;
    if (w_ij > 0.f) {
      c_00 += w_ij * (x_iW.x - x_j.x) * (x_iW.x - x_j.x);
      c_01 += w_ij * (x_iW.x - x_j.x) * (x_iW.y - x_j.y);
      c_02 += w_ij * (x_iW.x - x_j.x) * (x_iW.z - x_j.z);

      c_10 += w_ij * (x_iW.y - x_j.y) * (x_iW.x - x_j.x);
      c_11 += w_ij * (x_iW.y - x_j.y) * (x_iW.y - x_j.y);
      c_12 += w_ij * (x_iW.y - x_j.y) * (x_iW.z - x_j.z);

      c_20 += w_ij * (x_iW.z - x_j.z) * (x_iW.x - x_j.x);
      c_21 += w_ij * (x_iW.z - x_j.z) * (x_iW.y - x_j.y);
      c_22 += w_ij * (x_iW.z - x_j.z) * (x_iW.z - x_j.z);
      ctr += 1.f;
    }
  }
  auto ks = 1.f / kernelSum;
  if (ks == ks && ks < HUGE_VALF && ks > -HUGE_VALF) {
    G(i, 0, 0) = ks * c_00;
    G(i, 0, 1) = ks * c_01;
    G(i, 0, 2) = ks * c_02;

    G(i, 1, 0) = ks * c_10;
    G(i, 1, 1) = ks * c_11;
    G(i, 1, 2) = ks * c_12;

    G(i, 2, 0) = ks * c_20;
    G(i, 2, 1) = ks * c_21;
    G(i, 2, 2) = ks * c_22;
  } else {
    float e = sqrt(1.f / 3.f);
    G(i, 0, 0) = e;
    G(i, 0, 1) = 0.f;
    G(i, 0, 2) = 0.f;

    G(i, 1, 0) = 0.f;
    G(i, 1, 1) = e;
    G(i, 1, 2) = 0.f;

    G(i, 2, 0) = 0.f;
    G(i, 2, 1) = 0.f;
    G(i, 2, 2) = e;
  }
  arrays.auxDistance[i] = ctr;
}

basicFunctionType calculateAnisotropicMatrix(Memory arrays, float hScale) {
  checkedParticleIdx(i);
  auto A = SVD::Mat3x3::fromPtr(arrays.anisotropicMatrices, i, arrays.num_ptcls);
  auto usv = SVD::svd(A);
  auto &U = usv.U;
  auto &S = usv.S;
  auto &V = usv.V;
  auto r = powf(arrays.volume[i] * PI4O3_1, 1.f / 3.f);
  // auto ratio = (PI4O3 * arrays.radius * arrays.radius * arrays.radius) / arrays.volume[i];
  auto ratio = (arrays.radius) / r;
  //auto ks = arrays.anisotropicKs * ratio * ratio;
  int32_t numNeighs = (int32_t)arrays.auxDistance[i];
  //constexpr float sigma = 16.f / CUDART_PI_F;
  auto H = arrays.position[i].w * kernelSize();
  // float hScale = INTERPOLATOR_SCALE;
  if (numNeighs > arrays.anisotropicNepsilon) {
    S.m_01 = S.m_10 = S.m_02 = S.m_20 = S.m_21 = S.m_12 = 0.f;
    S.m_00 = fabsf(S.m_00);
    S.m_11 = fabsf(S.m_11);
    S.m_22 = fabsf(S.m_22);
    // auto maxSingular = math::max(math::max(S.m_00, S.m_11), S.m_22);
    float s1 = fabsf(S.m_00);
    float s2 = fabsf(S.m_11);
    float s3 = fabsf(S.m_22);
    const float maxSingularVal = math::max(math::max(s1, s2), s3);

    auto scale = H * H; // (H * H);
    // const float maxSingularVal = std::max(std::max(s1, s2), s3);
    float c0 = 0.0116955f;
    float c1 = -0.00484276f;
    float c2 = 0.150583f;
    auto ev = c0 + c1 * hScale + c2 * hScale * hScale;

    s1 = scale * 1.f / (math::max(s1, maxSingularVal / 4.f) / ev);
    s2 = scale * 1.f / (math::max(s2, maxSingularVal / 4.f) / ev);
    s3 = scale * 1.f / (math::max(s3, maxSingularVal / 4.f) / ev);

    S.m_00 = s1;
    S.m_11 = s2;
    S.m_22 = s3;

    // S.m_00 = arrays.anisotropicKs * math::max(S.m_00, maxSingular / arrays.anisotropicKr);
    // S.m_11 = arrays.anisotropicKs * math::max(S.m_11, maxSingular / arrays.anisotropicKr);
    // S.m_22 = arrays.anisotropicKs * math::max(S.m_22, maxSingular / arrays.anisotropicKr);

    // S.m_01 = S.m_10 = S.m_02 = S.m_20 = S.m_21 = S.m_12 = 0.f;

    // S.m_00 = 1.f / (S.m_00);
    // S.m_11 = 1.f / (S.m_11);
    // S.m_22 = 1.f / (S.m_22);

    auto G = V * S * U.transpose();
    S.m_00 = 1.f / s1;
    S.m_11 = 1.f / s2;
    S.m_22 = 1.f / s3;
    float h_i = arrays.position[i].w * kernelSize() * hScale;
    auto M = U * S * V.transpose() * h_i;
    auto sx = sqrt(M.m_00 * M.m_00 + M.m_01 * M.m_01 + M.m_02 * M.m_02);
    auto sy = sqrt(M.m_10 * M.m_10 + M.m_11 * M.m_11 + M.m_12 * M.m_12);
    auto sz = sqrt(M.m_20 * M.m_20 + M.m_21 * M.m_21 + M.m_22 * M.m_22);
    arrays.anisotropicSupport[i] = float4{sx, sy, sz, H * hScale / std::min(std::min(s1, s2), s3)};
    // G = A;
    // auto scale = powf(S.m_00* S.m_11* S.m_22, 1.f / 3.f);

    G *= 1.f / (h_i);
    G.toPtr(arrays.anisotropicMatrices, i, arrays.num_ptcls);
    G(i, 1, 0) = arrays.volume[i] / arrays.density[i] * G.det();
    arrays.centerPosition[i].w = arrays.volume[i] / arrays.density[i] * G.det();
  } else {
    auto G = SVD::Mat3x3();
    G.m_00 = 1.f / (arrays.position[i].w * kernelSize() * arrays.anisotropicKn);
    G.m_11 = 1.f / (arrays.position[i].w * kernelSize() * arrays.anisotropicKn);
    G.m_22 = 1.f / (arrays.position[i].w * kernelSize() * arrays.anisotropicKn);
    G.toPtr(arrays.anisotropicMatrices, i, arrays.num_ptcls);
    G(i, 1, 0) = arrays.volume[i] / arrays.density[i] * G.det();
    arrays.centerPosition[i].w = arrays.volume[i] / arrays.density[i] * G.det();
    arrays.anisotropicSupport[i] = float4{H * hScale, H * hScale, H * hScale, H * hScale};
  }
  arrays.auxIsoDensity[i] = arrays.volume[i];
  // if(arrays.auxTest[i] == 0.f)
  //	arrays.centerPosition[i].w = 0.f;
  // arrays.centerPosition[i].w = arrays.auxTest[i];
}
basicFunctionType calcualteSmoothPositions(Memory arrays) {
  checkedParticleIdx(i);
  float4 position = arrays.position[i];
  float4 centerPosition = arrays.centerPosition[i];
  float l = arrays.anisotropicLambda;
  float4 smoothedPosition = (1.f - l) * position + l * centerPosition;
  smoothedPosition.w = centerPosition.w;
  smoothedPosition.w = arrays.anisotropicSupport[i].w;
  arrays.centerPosition[i] = smoothedPosition;
}
neighFunction(centerPositions, calculateCenterPosition, "calculating center positions", caches<float4, float>{});
neighFunction(covarianceMatrices, calculateCovarianceMatrix, "calculating covariance matrices",
              caches<float4, float>{});
basicFunction(anisotropicMatrices, calculateAnisotropicMatrix, "calculating anisotropy matrices");
basicFunction(smoothPositions, calcualteSmoothPositions, "calculating smooth center positions");

void generateAnisotropicMatrices(Memory mem) {
  cudaDeviceSynchronize();
  launch<centerPositions>(mem.num_ptcls, mem, get<parameters::render_settings::anisotropicKs>());
  launch<covarianceMatrices>(mem.num_ptcls, mem, get<parameters::render_settings::anisotropicKs>());
  launch<anisotropicMatrices>(mem.num_ptcls, mem, get<parameters::render_settings::anisotropicKs>());
  launch<smoothPositions>(mem.num_ptcls, mem);
}
} // namespace anisotropy
} // namespace SPH