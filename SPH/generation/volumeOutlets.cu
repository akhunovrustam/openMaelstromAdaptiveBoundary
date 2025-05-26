#include <SPH/generation/volumeOutlets.cuh>
#include <utility/include_all.h>

basicFunctionDeviceType delete_particles(SPH::Outlet::Memory arrays, float t, float o_t) {
  checkedParticleIdx(i);
  auto p = arrays.position[i];
  auto H = p.w * Kernel<kernel_kind::spline4>::kernel_size();
  for (int32_t b = 0; b < arrays.volumeOutletCounter; ++b) {
    cuda_atomic<float> res_atomic(arrays.volumeOutletRateAccumulator + b);

    float4 d_min = arrays.volumeOutletMin[b];
    float4 d_max = arrays.volumeOutletMax[b];

    if ((d_min.x < p.x) && (d_min.y < p.y) && (d_min.z < p.z) && (d_max.x > p.x) && (d_max.y > p.y) &&
        (d_max.z > p.z)) {
      float4 d_p = (p - d_min) / (d_max - d_min);
      float4 ap = p;// -d_min.val;
      //printf("GVDB -> App space: [%f %f %f]\n", ap.x, ap.y, ap.z);
      auto vp = ap * arrays.volumeOutletVoxelSizes[b] + arrays.volumeOutletOffsets[b] + 0.5f;
      //printf("GVDB -> Voxel space: [%f %f %f]\n", vp.x, vp.y, vp.z);
      //printf("Compval: [%f %f %f]\n", d_p.val.x * arrays.gvdbVoxelSizes[i].x, d_p.val.y * arrays.gvdbVoxelSizes[i].y, d_p.val.z * arrays.gvdbVoxelSizes[i].z);

      float3 offs, vmin, vdel; uint64 nid;
      float3 wpos{ vp.x, vp.y, vp.z };

      //printf("%d -> %p[%d] -> [%f %f %f] - [%f %f %f]\n", i, arrays.outletGVDBVolumes[i], b
      //);

      gvdb::VDBNode* node = gvdb::getNodeAtPoint(arrays.outletGVDBVolumes[b], wpos, &offs, &vmin, &vdel, &nid);				// find vdb node at point
      if (node == 0x0) {
          continue;
      }
      //else {
      //    bool removed = true;
      //    if (removed)
      //        arrays.position[i].w = FLT_MAX;
      //}
      //continue;
      //float d = 1.f;
      float3 p = offs + (wpos - vmin) / vdel;
      ////printf("GVDB -> [%f %f %f] + [%f %f %f] - [%f %f %f] / [%f %f %f] -> [%f %f %f]\n",
      //    //offs.x, offs.y, offs.z, wpos.x, wpos.y, wpos.z, vmin.x, vmin.y, vmin.z,
      //    //vdel.x, vdel.y, vdel.z, p.x, p.y, p.z);
      auto lookup = [&](auto p) {return tex3D<float>(arrays.outletGVDBVolumes[b]->volIn[0], p.x, p.y, p.z); };
      float d = lookup(p);
      //auto dx = (lookup(p + float3{ 0.5f,0.f,0.f }) - lookup(p - float3{ 0.5f,0.f,0.f })) * 1.f;
      //auto dy = (lookup(p + float3{ 0.f,0.5f,0.f }) - lookup(p - float3{ 0.f,0.5f,0.f })) * 1.f;
      //auto dz = (lookup(p + float3{ 0.f,0.f,0.5f }) - lookup(p - float3{ 0.f,0.f,0.5f })) * 1.f;
      //auto n = float4{ dx, dy, dz,0.f };
      //if (math::length3(n) < 1e-5f) {
      //    auto dx = (lookup(p + float3{ 1.f,0.f,0.f }) - d) * 0.5f;
      //    auto dy = (lookup(p + float3{ 0.f,1.f,0.f }) - d) * 0.5f;
      //    auto dz = (lookup(p + float3{ 0.f,0.f,1.f }) - d) * 0.5f;
      //    n = float4{ dx, dy, dz,0.f };
      //}
      //n = math::normalize3(n);
      //n.w = 0.f;
      //n.w = d;

      if (d < 0.f){
        float d_t = t - o_t;
        float v = arrays.volume[i];
        float rate = arrays.volumeOutletRate[b];
				float snapped_mlm, old = res_atomic.val();
        bool removed = true;
        if(rate > 0.f)
				do {
					snapped_mlm = old;
          float new_rate = snapped_mlm + 1.f;
					if (new_rate  >= d_t * rate){
            removed = false;
						break;
          }
					old = res_atomic.CAS(snapped_mlm, new_rate);
				} while (old != snapped_mlm);
        if(removed)
          arrays.position[i].w = FLT_MAX;
      }
    }
  }
}

basicFunctionDevice(deleteParticles, delete_particles, "Delete Particles");

void SPH::Outlet::remove(Memory mem) {
  static std::vector<float> old_rates;
  std::vector<float> rates;
  for(auto vol : get<parameters::outlet_volumes::volume>()){
    rates.push_back(vol.flowRate);
  }
  if(rates.size() != old_rates.size() || rates != old_rates){
    old_rates = rates;
    cudaMemcpy(arrays::volumeOutletRate::ptr, old_rates.data(), sizeof(float) * old_rates.size(), cudaMemcpyHostToDevice);
  }
  if (get<parameters::internal::simulationTime>() > get<parameters::outlet_volumes::volumeOutletTime>() + 1.f) {
    cudaMemset(arrays::volumeOutletRateAccumulator::ptr, 0x00, sizeof(float) * mem.volumeOutletCounter);
    get<parameters::outlet_volumes::volumeOutletTime>() = get<parameters::internal::simulationTime>();
  }
  launchDevice<deleteParticles>(mem.num_ptcls, mem, get<parameters::internal::simulationTime>(), get<parameters::outlet_volumes::volumeOutletTime>());
}
