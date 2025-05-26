#include <SPH/density/density.cuh>
#include <utility/include_all.h>

/** This function calculates a density estimate using SPH for all particles and additionally uses a
 * lookup table to improve the boundary density contribution of flat boundary planes (no
 * source/novel). Only calculates unit density not actual density  **/
neighFunctionType estimate_density(SPH::Density::Memory arrays) { 
  checkedParticleIdx(i);
  cache_arrays((pos, position), (vol, volume));
  float_u<> unit_density = 0.f;
  float_u<> unit_density_b = 0.f;
  float_u<> unit_density_f = 0.f;
  // SWH::spline(pos[i], arrays);
  // unit_density = math::clamp(unit_density, 0.f, 1.f - vol[i] * kernel(pos[i], pos[i]));
  int32_t ctr = 0;
  // arrays.lifetime[i] = unit_density.val;
  int ncnt = 0;
  if (arrays.particle_type[i] == 0)
      iterateNeighbors(j) {
        /// VERY STRANGE CONDITIONS, CHECK IT
    #ifdef DEBUG_INVALID_PARITLCES
      if (arrays.particle_type[i] == FLUID_PARTICLE ||
          (arrays.particle_type[i] != FLUID_PARTICLE &&
              (arrays.particle_type[j] == FLUID_PARTICLE || arrays.particle_type[j] == arrays.particle_type[i])))
#endif
          if (W_ij > 0.f)
              ncnt++;
      float_u<> ttmp = vol[j] * W_ij;
          unit_density += ttmp;
         //printf("dens_part %.10f %.10f %.10f %.10f\n", vol[j].val, W_ij.val, ttmp.val, pos[j].val.w);
         //if (i == j) printf("i == j\n");
        //printf("partdens %.30f dist %f\n", (W_ij).val, math::length3(pos[i] - pos[j]).val);
            if (arrays.particle_type[j] != 0 || i == j)
            {
                unit_density_b += vol[j] * W_ij;
            }
            if (arrays.particle_type[j] == 0)
                unit_density_f += vol[j] * W_ij;

        ctr++;
      }
  else
  {
      iterateNeighbors(j) {
          if (arrays.particle_type[j] == 0)
              unit_density += vol[j] * W_ij;
              unit_density_b += vol[j] * W_ij;
              unit_density_f += vol[j] * W_ij;
      }
      float cf = 0.6;
      unit_density += cf;
      unit_density_b += cf;
      unit_density_f += cf;
  }

 
  arrays.density[i] = unit_density;
  arrays.density_b[i] = unit_density_b;
  arrays.density_f[i] = unit_density_f;
  //arrays.density_b[i] = unit_density;
  //arrays.density_f[i] = unit_density;
#ifdef DEBUG_INVALID_PARITLCES
  if (unit_density != unit_density || unit_density == 0.f) {
    printf("%s: Invalid particle %d: %f %f %f " _VECSTR " %d\n", __FUNCTION__, i, unit_density.val, vol[i].val,
           SWH::spline(pos[i], arrays).val, _VEC(pos[i].val), ctr);
    iterateNeighbors(j) {
      printf("i: %d - " _VECSTR " -> %f : %f\n", j, _VEC(pos[j].val),
             math::distance3(pos[i], pos[j]).val / kernelSize(), (vol[j] * W_ij).val);
    }

  }
#endif

}


neighFunctionDevice(estimateDensity, estimate_density, "Estimate Density", caches<float4, float>{});
#include <fstream>
void SPH::Density::estimate_density(Memory mem) {
  if (mem.num_ptcls == 0)
    return;
  {
    launchDevice<estimateDensity>(mem.num_ptcls, mem);
  }
  cuda::sync();
  if (get<parameters::internal::frame>() == 0)
		for (int jj = 0; jj < mem.num_ptcls; jj++)
            if (mem.particle_type[jj] != 0 && mem.density[jj].val > get<parameters::particle_settings::max_density>())
				get<parameters::particle_settings::max_density>() = mem.density[jj].val;

  //{
  //    testGVDB <<<1, 32 >>> (mem);
  //}
}