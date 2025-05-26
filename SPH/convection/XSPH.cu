#include <SPH/convection/XSPH.cuh>
#include <utility/include_all.h>

// This function implements the artificial viscosity from XSPH, and additionally adds a stronger
// viscosity if a particles lifetime is negative. This is only the case for particles created with
// splitting which results in an additional layer of blending on top of the normal one. Contrary to
// the normal velocity blending this method basically blends the velocity of split particles with
// all surrounding particles which helps dampen the effects caused by density errors on splitting.
neighFunctionType xsph_viscosity(SPH::XSPH::Memory arrays) {
  checkedParticleIdx(i);
  cache_arrays((pos, position), (vol, volume), (vel, velocity.first), (den, density_f));

  if (arrays.particle_type[i] != 0) return;
  auto beta_i = getBlendWeight(arrays, i);

  auto sum = vel[i];// - arrays.xsph_viscosity * 0.25f * (beta_i)* vel[i] * pV_b * pW_ib * 0.f;


  auto boundaryDistance = pDistance.val;
  if (boundaryDistance.w < 1e20f) {
      auto pb = arrays.position[i] - boundaryDistance.w * boundaryDistance;
      auto n = math::normalize3(boundaryDistance);
      n.w = 0.f;
      auto boundaryKernel = boundary::spline(arrays.position[i], arrays.volume[i], 1.f, arrays,
          boundary::kind::plane, -1);
      auto viscosity = 0.5f / (1.f + den[i]) * 2.f * (-vel[i]) * boundaryKernel;
      auto vortho = math::dot3(viscosity, n) * n;
      auto vpar = viscosity - vortho;
      sum += vortho;
  }

  iterateNeighbors(j) {
      if (arrays.particle_type[j] != 0) continue;
    auto beta_j = getBlendWeight(arrays, j); 
    auto beta_ij = 0.5f * (beta_i + beta_j);
    //beta_ij = 1.f;
    auto viscosity_constant = arrays.xsph_viscosity * (1.f + beta_ij);

    sum += viscosity_constant * vol[j] / (den[j] + den[i]) * 2.f * (vel[j] - vel[i]) * W_ij; 
  }
  arrays.velocity.second[i] = sum;
#ifdef DEBUG_INVALID_PARITLCES
  auto v_old = vel[i].val;
  auto v_add = sum.val;
  if (v_add.x != v_add.x)
	  printf("%s: Invalid particle %d: " _VECSTR " + " _VECSTR "\n",
		  __FUNCTION__, i,
		  _VEC(v_old), _VEC(v_add));
#endif
}

neighFunction(xsphViscosity, xsph_viscosity, "XSPH Viscosity", caches<float4, float4, float, float>{});

void SPH::XSPH::viscosity(Memory mem) {
  launch<xsphViscosity>(mem.num_ptcls, mem);
  info<arrays::velocity>().swap();
}
