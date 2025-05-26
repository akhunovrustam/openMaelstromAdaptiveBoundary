#define NEW_STYLE
#include <SPH/boundary/moving_planes.cuh>
#include <utility/include_all.h>

// This function adjusts the velocity of a particle against a moving boundary by simply removing the
// velocity of the particle in direction of the boundary. This function is called once per
// boundary described by an implicit plane equation E(n.x, n.y, n.z, d) and the velocity of the
// boundary.
basicFunctionType correct_velocity_moving(SPH::moving_planes::Memory arrays, float4_u<void_unit_ty> E, float4_u<SI::velocity>) {
  checkedParticleIdx(i);

  auto r = arrays.radius;
  auto p = arrays.position.first[i];
  auto v = arrays.velocity.first[i];
  auto dt = arrays.timestep;

  float4_u<> n{ E.val.x, E.val.y, E.val.z, 0.f };
  n = math::normalize(n);

  auto pe = p + v * dt;
  auto dist = math::planeDistance(E, pe);

  if (dist < r) {
    auto vTn = math::dot3(n, v) * n;
    auto v2 = v - vTn;
    v = v2;
  }
  arrays.velocity.first[i] = v;
}

// This function adjusts the position of a particle against a moving boundary by simply reprojecting
// the particle away from the wall. This process usually causes some errors, usually visible as
// pressure fluctuations on the boundary, but due to these walls possibly being infinite and planar
// representing them with particles could become very expensive. Additionally using implicit planes
// to describe the boundaries makes this process very quick. This function is called once per
// boundary described by an implicit plane equation E(n.x, n.y, n.z, d) and the velocity of the
// boundary.
basicFunctionType correct_position_moving(SPH::moving_planes::Memory arrays, float4_u<void_unit_ty> E, float4_u<SI::velocity> v_diff) {
  checkedParticleIdx(i);
  auto r = arrays.radius;
  auto p = arrays.position.first[i];
  auto v = arrays.velocity.first[i];
  auto dt = arrays.timestep;
  float4_u<> n{E.val.x, E.val.y, E.val.z, 0.f};
  n = math::normalize(n);

  auto d = math::planeDistance(E, p);
  if (d < r)
    p += (r - d) * n;

  auto pe = p + v * dt;
  d = math::planeDistance(E, pe);

  bool hit = false;
  if (d < r) {
    v = v - math::dot3(n, v) * n;
    hit = true;
  }
  if (hit)
    v += v_diff;
  v.val.w = 0.f;

  p -= arrays.velocity.first[i] * dt;
  arrays.velocity.first[i] = v;
  pe = p + dt * v;
  d = math::planeDistance(E, pe);
  if (d < r)
    p += (r - d) * n;
  arrays.position.first[i] = p;
}

// Launcher to correct the velocity of particles against a single boundary
basicFunction(correctVelocity, correct_velocity_moving, "Moving Planes: correct velocity");
// Launcher to correct the position of particles against a single boundary
basicFunction(correctPosition,correct_position_moving, "Moving Planes: correct position");
// Launcher to correct the position of particles against a single boundary
basicFunctionType update_plane_moving(int32_t threads, SPH::moving_planes::Memory arrays, float4_u<void_unit_ty> E, float4_u<SI::velocity> v, int32_t idx) {
  checkedThreadIdx(i);
  arrays.boundaryPlanes[idx] = E;
  arrays.boundaryPlaneVelocity[idx] = v;
}
basicFunction(updatePlane,update_plane_moving, "Moving Planes: update planes");


// Main entry function of the module for correcting positions. Iterates over all boundaries and
// calls the appropriate correction functions once for each boundary. This method can be called
// safely if no boundaries exist.
void SPH::moving_planes::correct_position(Memory mem) {
  for (auto plane : get<parameters::moving_plane::plane>()) {
    float t     = plane.duration;
    float f     = plane.frequency;
    float m     = plane.magnitude;
    float3 p    = plane.plane_position;
    float3 n    = plane.plane_normal;
    float3 dir  = plane.plane_direction;

    p          += dir * m * sinf(2.f * CUDART_PI_F * f * (get<parameters::internal::simulationTime>()));

    auto p_prev = dir * m * sinf(2.f * CUDART_PI_F * f * (get<parameters::internal::simulationTime>() - get<parameters::internal::timestep>()));
    auto p_diff = p - p_prev;
    auto v_diff = p_diff / get<parameters::internal::timestep>();

    auto nn     = math::normalize(n);
    auto d      = math::dot3(p, nn);
    float4_u<> E{nn.x, nn.y, nn.z, d};

    if (t < get<parameters::internal::simulationTime>() && t > 0.f)
      continue;
    launch<correctVelocity>(mem.num_ptcls, mem, E, float4_u<SI::velocity>{v_diff.x, v_diff.y, v_diff.z, 0.f});
  }
}

// Main entry function of the module for correcting velocities. Iterates over all boundaries and
// calls the appropriate correction functions once for each boundary. This method can be called
// safely if no boundaries exist.
void SPH::moving_planes::correct_velocity(Memory mem) {
  for (auto plane : get<parameters::moving_plane::plane>()) {
    auto t = plane.duration;
    auto f = plane.frequency;
    auto m = plane.magnitude;
    auto p = plane.plane_position;
    auto n = plane.plane_normal;
    auto dir = plane.plane_direction;
    p += dir * m * sinf(2.f * CUDART_PI_F * f * get<parameters::internal::simulationTime>());

    auto p_prev = plane.plane_position + dir * m * sinf(2.f * CUDART_PI_F * f * (get<parameters::internal::simulationTime>() - get<parameters::internal::timestep>()));
    auto p_diff = p - p_prev;
    auto v_diff = -p_diff / get<parameters::internal::timestep>();

    auto nn     = math::normalize(n);
    auto d      = math::dot3(p, nn);
    float4_u<> E{nn.x, nn.y, nn.z, d};

    if (t < get<parameters::internal::simulationTime>() && t > 0.f)
      continue;
    launch<correctPosition>(mem.num_ptcls, mem, E, float4_u<SI::velocity>{v_diff.x, v_diff.y, v_diff.z, 0.f});
  }
}

void SPH::moving_planes::update_boundaries(Memory mem){
  for (auto plane : get<parameters::moving_plane::plane>()) {
    auto t = plane.duration;
    auto f = plane.frequency;
    auto m = plane.magnitude;
    auto p = plane.plane_position;
    auto n = plane.plane_normal;
    auto dir = plane.plane_direction;
    auto idx = plane.index;
    //if (get<parameters::internal::simulationTime>() > 30.f) return;
    p += dir * m * sinf(2.f * CUDART_PI_F * f * get<parameters::internal::simulationTime>());
     
    if (math::dot3(n, float3{ 1,0,0 }) > 0.5f)
        get<parameters::render_settings::vrtxRenderDomainMin>().x = p.x;
    if (math::dot3(n, float3{ 1,0,0 }) < -0.5f)
        get<parameters::render_settings::vrtxRenderDomainMax>().x = -p.x;
    if (math::dot3(n, float3{ 0,1,0 }) > 0.5f)
        get<parameters::render_settings::vrtxRenderDomainMin>().y = p.y;
    if (math::dot3(n, float3{ 0,1,0 }) < -0.5f)
        get<parameters::render_settings::vrtxRenderDomainMax>().y = -p.y;
    if (math::dot3(n, float3{ 0,0,1 }) > 0.5f)
        get<parameters::render_settings::vrtxRenderDomainMin>().z = p.z;
    if (math::dot3(n, float3{ 0,0,1 }) < -0.5f)
        get<parameters::render_settings::vrtxRenderDomainMax>().z = -p.z;

    auto p_prev = plane.plane_position + dir * m * sinf(2.f * CUDART_PI_F * f * (get<parameters::internal::simulationTime>() - get<parameters::internal::timestep>()));
    auto p_diff = p - p_prev;
    auto v_diff = -p_diff / get<parameters::internal::timestep>();

    auto nn     = math::normalize(n);
    auto d      = math::dot3(p, nn);
    float4_u<> E{nn.x, nn.y, nn.z, d};

    if (t < get<parameters::internal::simulationTime>() && t > 0.f)
      continue;
	uFloat4<SI::velocity> v{ math::castTo<float4>(v_diff) };
    launch<updatePlane>(1, 1, mem, E, v, idx);
  }

}