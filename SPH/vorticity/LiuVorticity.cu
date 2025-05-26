#include <SPH/vorticity/LiuVorticity.cuh>
#include <utility/include_all.h>

neighFunctionType computeVorticity(SPH::LiuVorticity::Memory arrays) {
  checkedParticleIdx(i);
  cache_arrays((pos, position), (vol, volume), (vel, velocity), (den, density));
  auto v_i = arrays.velocity[i] + arrays.timestep * arrays.acceleration[i];

  auto rho_i = arrays.density[i];

  auto vorticity = float4{0.f, 0.f, 0.f, 0.f};
  iterateNeighbors(j) {
    if (i != j)
      vorticity += -1.f / rho_i * arrays.volume[j] * math::cross(vel[i] - vel[j], GW_ij);
  }
  arrays.vorticity.first[i] = vorticity;
}
#include <math/SVD.h>
neighFunctionType estimateVorticity(SPH::LiuVorticity::Memory arrays) {
  checkedParticleIdx(i);
  cache_arrays((pos, position), (vol, volume), (vel, velocity), (den, density));
  auto v_i = arrays.velocity[i] + arrays.timestep * arrays.acceleration[i];

  auto rho_i = arrays.density[i];

  auto vorticity = arrays.vorticity.first[i];
  auto vorticityTilde = float4{0.f, 0.f, 0.f, 0.f};
  auto lapVorticitiy = float4{0.f, 0.f, 0.f, 0.f};
  SVD::Mat3x3 vgrad{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  auto outerProd = [](float4 a, float4 b) {
    return SVD::Mat3x3{a.x * b.x, a.x * b.y, a.x * b.z, 
        a.y * b.x, a.y * b.y, a.y * b.z, a.z * b.x, a.z * b.y, a.z * b.z};
  };

  iterateNeighbors(j) {
    if (i == j)
      continue;
    auto xij = pos[i] - pos[j];
    auto tij = vorticity - arrays.vorticity.first[j];
    vgrad = vgrad + outerProd(vel[i] - vel[j], GW_ij) * vol[j] * (-1.f / rho_i);
    lapVorticitiy += 8.f * arrays.volume[j] / rho_i * math::dot3(tij, xij) /
                     (math::sqlength3(xij) + 0.01f * support(pos[i], pos[j] * kernelSize())) * GW_ij;

    auto vij = v_i - vel[j] - arrays.timestep * arrays.acceleration[j];
    vorticityTilde += -1.f / rho_i * arrays.volume[j] * math::cross(vij, GW_ij);
  }
  auto dvdt = float4{
      vorticity.x * vgrad.m_00 + vorticity.y * vgrad.m_01+ vorticity.z * vgrad.m_02,
      vorticity.x * vgrad.m_10 + vorticity.y * vgrad.m_11+ vorticity.z * vgrad.m_12,
      vorticity.x * vgrad.m_20 + vorticity.y * vgrad.m_21+ vorticity.z * vgrad.m_22,0.f} +
              0.05f * lapVorticitiy;
  auto delVorticity = vorticity + arrays.timestep * dvdt - vorticityTilde;

  arrays.vorticity.second[i] = delVorticity;
}

neighFunctionType computeStreamFunction(SPH::LiuVorticity::Memory arrays) {
  checkedParticleIdx(i);
  cache_arrays((pos, position), (vol, volume), (vor, vorticity.second), (den, density));

  float4 stream{0.f, 0.f, 0.f, 0.f};
  iterateNeighbors(j) {
    auto xij = math::length3(pos[i] - pos[j]);
    if (i != j)
      stream += 1.f / (4.f * CUDART_PI_F) * vor[j] * vol[j] / xij;
  }
  arrays.vorticity.first[i] = stream;
}
neighFunctionType correctVorticity(SPH::LiuVorticity::Memory arrays, float vorticityStrength) {
  checkedParticleIdx(i);
  cache_arrays((pos, position), (vol, volume), (streamFn, vorticity.first), (den, density));

  auto rho_i = arrays.density[i];
  auto dv = float4{0.f, 0.f, 0.f, 0.f};
  iterateNeighbors(j) {
    if (i != j)
      dv += -1.f / rho_i * arrays.volume[j] * math::cross(streamFn[i] - streamFn[j], GW_ij);
  }
  auto alpha = vorticityStrength;
  arrays.vorticity.second[i] = alpha * dv;
}
neighFunctionType updateAcceleration(SPH::LiuVorticity::Memory arrays) {
  checkedParticleIdx(i);
  arrays.vorticity.first[i] = arrays.vorticity.second[i];
  arrays.acceleration[i] += arrays.vorticity.first[i] / arrays.timestep;
}

neighFunction(vorticity1, computeVorticity, "Integrate: Velocity", caches<float4, float4, float, float>{});
neighFunction(vorticity2, estimateVorticity, "Integrate: Velocity", caches<float4, float4, float, float>{});
neighFunction(vorticity3, computeStreamFunction, "Integrate: Velocity", caches<float4, float4, float, float>{});
neighFunction(vorticity4, correctVorticity, "Integrate: Velocity", caches<float4, float4, float, float>{});
neighFunction(vorticity5, updateAcceleration, "Integrate: Velocity", caches<float4, float4, float, float>{});

void SPH::LiuVorticity::vorticity(Memory mem) {
  if (mem.num_ptcls == 0)
    return;
  launch<vorticity1>(mem.num_ptcls, mem);
  launch<vorticity2>(mem.num_ptcls, mem);
  launch<vorticity3>(mem.num_ptcls, mem);
  launch<vorticity4>(mem.num_ptcls, mem, get<parameters::vorticitySettings::vorticityCoeff>());
  launch<vorticity5>(mem.num_ptcls, mem);
}