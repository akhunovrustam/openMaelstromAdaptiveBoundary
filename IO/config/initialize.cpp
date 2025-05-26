#if defined(__MSC_VER__) && defined(__clang__)
#include <host_defines.h>
#undef __builtin_align__
#define __builtin_align__(x)
#endif
#include <IO/config/config.h>
#include <IO/config/initialize.h>
#include <IO/config/snapshot.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <utility/cuda.h>
#include <tools/log.h>
#include <tools/pathfinder.h>
#include <math.h>
#include <math/template/metafunctions.h>
#include <math/template/tuple_for_each.h>

bool IsPrime(int32_t number) {
  if (number == 2 || number == 3)
    return true;
  if (number % 2 == 0 || number % 3 == 0)
    return false;
  int divisor = 6;
  while (divisor * divisor - 2 * divisor + 1 <= number) {
    if (number % (divisor - 1) == 0)
      return false;
    if (number % (divisor + 1) == 0)
      return false;
    divisor += 6;
  }
  return true;
}

int32_t NextPrime(int32_t a) {
  while (!IsPrime(++a)) {
  }
  return a;
}

void IO::config::initParameters() {
  if (get<parameters::modules::resorting>() == "mlm")
    get<parameters::modules::resorting>() = "MLM";
  if (get<parameters::modules::launch_cfg>() == "cpu") {
    get<parameters::internal::target>() = launch_config::host;
  } else if (get<parameters::modules::launch_cfg>() == "debug") {
    get<parameters::internal::target>() = launch_config::debug;
  } else if (get<parameters::modules::launch_cfg>() == "pure") {
    get<parameters::internal::target>() = launch_config::pure_host;
  } else
    get<parameters::internal::target>() = launch_config::device;

  if (get<parameters::modules::resorting>() == "linear_cell") {
    get<parameters::resort::resort_algorithm>() = 0;
    get<parameters::internal::cell_ordering>() = cell_ordering::linear_order;
    get<parameters::internal::cell_structure>() = cell_structuring::complete;
    get<parameters::internal::hash_size>() = hash_length::bit_32;
  } else if (get<parameters::modules::resorting>() == "hashed_cell" && get<parameters::modules::hash_width>() == "64bit") {
    get<parameters::resort::resort_algorithm>() = 1;
    get<parameters::internal::cell_ordering>() = cell_ordering::z_order;
    get<parameters::internal::cell_structure>() = cell_structuring::hashed;
    get<parameters::internal::hash_size>() = hash_length::bit_64;
  } else if (get<parameters::modules::resorting>() == "hashed_cell" && get<parameters::modules::hash_width>() == "32bit") {
    get<parameters::resort::resort_algorithm>() = 2;
    get<parameters::internal::cell_ordering>() = cell_ordering::z_order;
    get<parameters::internal::cell_structure>() = cell_structuring::hashed;
    get<parameters::internal::hash_size>() = hash_length::bit_32;
  } else if (get<parameters::modules::resorting>() == "MLM" && get<parameters::modules::hash_width>() == "64bit") {
    get<parameters::resort::resort_algorithm>() = 3;
    get<parameters::internal::cell_ordering>() = cell_ordering::z_order;
    get<parameters::internal::cell_structure>() = cell_structuring::MLM;
    get<parameters::internal::hash_size>() = hash_length::bit_64;
  } else if (get<parameters::modules::resorting>() == "MLM" && get<parameters::modules::hash_width>() == "32bit") {
    get<parameters::resort::resort_algorithm>() = 3;
    get<parameters::internal::cell_ordering>() = cell_ordering::z_order;
    get<parameters::internal::cell_structure>() = cell_structuring::MLM;
    get<parameters::internal::hash_size>() = hash_length::bit_32;
  } else if (get<parameters::modules::resorting>() == "compactMLM") {
    get<parameters::resort::resort_algorithm>() = 3;
    get<parameters::internal::cell_ordering>() = cell_ordering::z_order;
    get<parameters::internal::cell_structure>() = cell_structuring::compactMLM;
    get<parameters::internal::hash_size>() = hash_length::bit_64;
  }

  if (get<parameters::modules::resorting>() == "MLM" || get<parameters::modules::resorting>() == "hashed_cell" ||
      get<parameters::modules::resorting>() == "compactMLM") {
    if (get<parameters::simulation_settings::hash_entries>() == UINT_MAX)
      get<parameters::simulation_settings::hash_entries>() = NextPrime(get<parameters::simulation_settings::maxNumptcls>());
    if (get<parameters::simulation_settings::mlm_schemes>() == UINT_MAX) {
      if (get<parameters::modules::adaptive>() == true &&
          (get<parameters::modules::resorting>() == "MLM" || get<parameters::modules::resorting>() == "compactMLM"))
        get<parameters::simulation_settings::mlm_schemes>() =
            static_cast<uint32_t>(floorf(log2f(pow(get<parameters::adaptive::resolution>(), 1.f / 3.f)))) + 1u;
      else
        get<parameters::simulation_settings::mlm_schemes>() = 1;
    }
  }

  if (get<parameters::modules::neighborhood>() == "constrained") {
    get<parameters::internal::neighborhood_kind>() = neighbor_list::constrained;
  } else if (get<parameters::modules::neighborhood>() == "cell_based") {
    get<parameters::internal::neighborhood_kind>() = neighbor_list::cell_based;
  } else if (get<parameters::modules::neighborhood>() == "basic") {
    get<parameters::internal::neighborhood_kind>() = neighbor_list::basic;
  } else if (get<parameters::modules::neighborhood>() == "compactCell") {
    get<parameters::internal::neighborhood_kind>() = neighbor_list::compactCell;
  } else if (get<parameters::modules::neighborhood>() == "compactMLM") {
    get<parameters::internal::neighborhood_kind>() = neighbor_list::compactMLM;
  } else if (get<parameters::modules::neighborhood>() == "masked") {
    get<parameters::internal::neighborhood_kind>() = neighbor_list::masked;
  }

  // timestep
  auto &dt = get<parameters::internal::timestep>();
  if (dt == 0.f)
    dt = (get<parameters::simulation_settings::timestep_max>() - get<parameters::simulation_settings::timestep_min>()) / 2.f;
  // Init parameters for constrained neighbor list
  uint32_t preferred_count = Kernel<kernel_kind::spline4>::neighbor_number;
  uint32_t count = get<parameters::simulation_settings::neighborlimit>() - 1;
  uint32_t leeway = count - preferred_count;
  get<parameters::support::support_leeway>() = leeway;
  get<parameters::support::target_neighbors>() = preferred_count;
  get<parameters::support::overhead_size>() = get<parameters::simulation_settings::maxNumptcls>() / 10;
  if (get<parameters::internal::neighborhood_kind>() == neighbor_list::constrained ||
      get<parameters::internal::neighborhood_kind>() == neighbor_list::basic) {
    info<arrays::neighborListSwap>().allocate(sizeof(uint32_t));
  }

  std::string host = get<parameters::simulation_settings::hostRegex>();
  std::string device = get<parameters::simulation_settings::deviceRegex>();
  std::string debug = get<parameters::simulation_settings::debugRegex>();
  if (host != "" || device != "" || debug != "")
    get<parameters::modules::regex_cfg>() = true;
}

void IO::config::initBoundary() {
  if (get<parameters::moving_plane::plane>().size() > 0)
    get<parameters::modules::movingBoundaries>() = true;
  if (get<parameters::inlet_volumes::volume>().size() > 0)
    get<parameters::modules::volumeInlets>() = true;
  // implicit boundary parameters
  std::vector<float4> planes;
  float3 max = get<parameters::internal::max_domain>();
  float3 min = get<parameters::internal::min_domain>();
  get<parameters::render_settings::vrtxDomainMin>() = float3{-FLT_MAX, -FLT_MAX, -FLT_MAX};
  get<parameters::render_settings::vrtxDomainMax>() = float3{FLT_MAX, FLT_MAX, FLT_MAX};
  float r = -get<parameters::particle_settings::radius>();
  {
    r = 0.f;
    char prev = 'x';
    for (auto c : get<parameters::simulation_settings::domainWalls>()) {
      if (c == 'x' || c == 'y' || c == 'z')
        prev = c;
      float3 dim;
      float sign = -1.f;
      if (c == '+')
        dim = max - r;
      if (c == '-') {
        dim = -(min + r);
        sign = 1.f;
      }
      if (c == '+' || c == '-') {
        if (prev == 'x') {
          planes.push_back(float4{sign, 0.f, 0.f, dim.x});
          if (c == '-')
            get<parameters::render_settings::vrtxDomainMin>().x = -dim.x - 0.5f;
          else
            get<parameters::render_settings::vrtxDomainMax>().x = dim.x + 0.5f;
        }
        if (prev == 'y') {
          planes.push_back(float4{0.f, sign, 0.f, dim.y});
          if (c == '-')
            get<parameters::render_settings::vrtxDomainMin>().y = -dim.y - 0.5f;
          else
            get<parameters::render_settings::vrtxDomainMax>().y = dim.y + 0.5f;
        }
        if (prev == 'z') {
          planes.push_back(float4{0.f, 0.f, sign, dim.z});
          if (c == '-')
            get<parameters::render_settings::vrtxDomainMin>().z = -dim.z - 0.5f;
          else
            get<parameters::render_settings::vrtxDomainMax>().z = dim.z + 0.5f;
        }
      }
    }
  }

  // {
  //   std::vector<float4> planes;
  //   float r = -get<parameters::particle_settings::radius>();
  //   r = 0.f;
  //   char prev = 'x';
  //   for (auto c : std::string("x+-y+-z+-")) {
  //     if (c == 'x' || c == 'y' || c == 'z')
  //       prev = c;
  //     float3 dim;
  //     float sign = -1.f;
  //     if (c == '+')
  //       dim = max - r;
  //     if (c == '-') {
  //       dim = -(min + r);
  //       sign = 1.f;
  //     }
  //     if (c == '+' || c == '-') {
  //       if (prev == 'x') {
  //         planes.push_back(float4{sign, 0.f, 0.f, dim.x});
  //       }
  //       if (prev == 'y') {
  //         planes.push_back(float4{0.f, sign, 0.f, dim.y});
  //       }
  //       if (prev == 'z') {
  //         planes.push_back(float4{0.f, 0.f, sign, dim.z});
  //       }
  //     }
  //   }
  // std::vector<float4> particles/* = some function to generate them*/;
  // std::vector<float4> outParticles;
  // for (const auto& ptcl : particles) {
  //     bool state = true;
  //     for (const auto &p : planes) {
  //       auto dot = p.x * ptcl.x + p.y * ptcl.y + p.z * ptcl.z + p.w;
  //       if(dot < get<parameters::particle_settings::radius>()) state = false;
  //     }
  //     if (state)
  //       outParticles.push_back(ptcl);
  //}
  //// outParticles now contains the final set of particles.
  // }

  for (auto &plane : get<parameters::moving_plane::plane>()) {
    // auto t = plane.duration.value;
    auto f = plane.frequency;
    auto m = plane.magnitude;
    auto p = plane.plane_position;
    auto n = plane.plane_normal;
    auto dir = plane.plane_direction;
    plane.index = (int32_t)planes.size();
    p += dir * m * sinf(2.f * CUDART_PI_F * f * get<parameters::internal::simulationTime>());
    auto nn = math::normalize(n);
    auto d = math::dot3(p, nn);
    planes.push_back(float4{nn.x, nn.y, nn.z, d});
  }
  get<parameters::internal::boundaryCounter>() = (int32_t)planes.size();
  // std::cout << "Number of boundary planes: " << planes.size() << std::endl;
  arrays::boundaryPlanes::ptr = (float4 *)cuda::malloc(sizeof(float4) * planes.size());
  arrays::boundaryPlaneVelocity::ptr = (float4 *)cuda::malloc(sizeof(float4) * planes.size());
  cuda::Memset(arrays::boundaryPlaneVelocity::ptr, 0x00, sizeof(float4) * planes.size());
  // info<arrays::boundaryPlanes>().allocate(sizeof(float4) * planes.size());
  cuda::memcpy(arrays::boundaryPlanes::ptr, planes.data(), planes.size() * sizeof(float4), cudaMemcpyHostToDevice);
  // for (uint32_t i = 0; i < planes.size(); ++i)
  //	arrays::boundaryPlanes::ptr[i] = planes[i];

  auto loadLUT = [&](auto name, auto &ptr) {
    std::string file = std::string(R"(cfg/)") + std::string(name) + std::string(R"(.lut)");
    std::ifstream iFile(resolveFile(file).string());
    std::vector<float> LUT;
    float f;
    while (iFile >> f)
      LUT.push_back(f);
    /*if (std::string(name).find("spline") != std::string::npos) {
            for (auto e : LUT)
                    std::cout << e << " ";
            std::cout << std::endl;
    }*/
    // std::reverse(LUT.begin(), LUT.end());
    // std::cout << name << " : " << LUT.size() << std::endl;
    ptr = (float *)cuda::malloc(sizeof(float) * LUT.size());
    cuda::memcpy(ptr, LUT.data(), LUT.size() * sizeof(float), cudaMemcpyHostToDevice);
    get<parameters::internal::boundaryLUTSize>() = (int32_t)LUT.size();
  };
  loadLUT("adhesion", arrays::adhesionLUT::ptr);
  loadLUT("offsetLUT", arrays::offsetLUT::ptr);
  loadLUT("spline", arrays::splineLUT::ptr);
  loadLUT("spline2", arrays::spline2LUT::ptr);
  loadLUT("splineGradient", arrays::splineGradientLUT::ptr);
  loadLUT("spiky", arrays::spikyLUT::ptr);
  loadLUT("spikyGradient", arrays::spikyGradientLUT::ptr);
  loadLUT("cohesion", arrays::cohesionLUT::ptr);
  loadLUT("volume", arrays::volumeLUT::ptr);
}

void IO::config::initKernel() {
  float_u<SI::volume> volume(PI4O3 * math::power<3>(uGet<parameters::particle_settings::radius>()));
  auto h = support_from_volume(volume);
  auto H = h * ::kernelSize();

  auto gen_position = [&](float_u<SI::m> r, int32_t i, int32_t j, int32_t k) {
    float4_u<> initial{2.f * i + ((j + k) % 2), sqrtf(3.f) * (j + 1.f / 3.f * (k % 2)), 2.f * sqrtf(6.f) / 3.f * k,
                       h / r};
    return initial * r;
  };

  auto radius = uGet<parameters::particle_settings::radius>();

  auto r = math::brentsMethod(
      [=](float_u<SI::m> r) {
        float_u<> density = -1.0f;
        int32_t requiredSlices_x = (int32_t)math::ceilf(h / r).val;
        int32_t requiredSlices_y = (int32_t)math::ceilf(h / (sqrtf(3.f) * r)).val;
        int32_t requiredSlices_z = (int32_t)math::ceilf(h / r * 3.f / (sqrtf(6.f) * 2.f)).val;
        density = -1.f;
        float4_u<SI::m> center_position{0.f, 0.f, 0.f, h.val};
        for (int32_t x_it = -requiredSlices_x; x_it <= requiredSlices_x; x_it++)
          for (int32_t y_it = -requiredSlices_y; y_it <= requiredSlices_y; y_it++)
            for (int32_t z_it = -requiredSlices_z; z_it <= requiredSlices_z; z_it++)
              density += volume * spline4_kernel(center_position, gen_position(r, x_it, y_it, z_it));
        return density;
      },
      radius * 0.1f, radius * 4.f, 1e-6_m, 10000);
  uGet<parameters::internal::ptcl_support>() = h;
  uGet<parameters::internal::cell_size>() = float3_u<SI::m>{H.val, H.val, H.val};
  uGet<parameters::internal::ptcl_spacing>() = r;
}

void IO::config::initDomain() {
  if (get<parameters::simulation_settings::boundaryObject>() != "") {
    std::string boundaryFile = get<parameters::internal::config_folder>() + get<parameters::simulation_settings::boundaryObject>();

    auto &min = get<parameters::internal::min_domain>();
    auto &max = get<parameters::internal::max_domain>();
    min = vector_t<float, 3>::max();
    max = -vector_t<float, 3>::max();

    std::string line;
    std::ifstream file(boundaryFile);
    while (std::getline(file, line)) {
      std::istringstream iss(line);
      char id;
      iss >> id;
      if (id == 'v') {
        float3 vtx;
        iss >> vtx.x >> vtx.y >> vtx.z;
        min = math::min(min, vtx);
        max = math::max(max, vtx);
      }
    }
    auto distFlowCoord = max - min;
    auto grid_size = math::ceilf(distFlowCoord / get<parameters::internal::cell_size>());

    get<parameters::internal::gridSize>() = int3{(int32_t)grid_size.y, (int32_t)grid_size.y, (int32_t)grid_size.y};
  } else {
    throw std::runtime_error("No boundary object provided.");
  }
  get<parameters::internal::min_coord>() = get<parameters::internal::min_domain>();
  get<parameters::internal::max_coord>() = get<parameters::internal::max_domain>();
}
#include <utility/generation.h>
//#include <math/SVD.h>

void IO::config::initVolumeBoundary() {
  auto vols = get<parameters::boundary_volumes::volume>();

  std::vector<VDBInfo *> gvdbVolumes;
  std::vector<float4> gvdbOffsets;
  std::vector<float4> gvdbVoxelSizes;
  std::vector<cudaTextureObject_t> textures;
  std::vector<float4> minAABB;
  std::vector<float4> maxAABB;
  std::vector<int4> dims;
  std::vector<float4> centersOfMasses;
  std::vector<float> volumes;
  std::vector<float> densities;
  std::vector<float4> positions;
  std::vector<float4> angles;
  std::vector<float4> velocities;
  std::vector<float4> angularVelocities;
  std::vector<int32_t> kinds;
  std::vector<Matrix4x4> inertias;
  std::vector<Matrix4x4> inertiasInv;

  namespace vdb = openvdb;
  namespace vdbt = openvdb::tools;
  namespace vdbm = openvdb::math;

  for (auto boundaryVolume : vols) {
    auto [texture, min, max, dimension, centerOfMass, inertia] = generation::cudaVolume(boundaryVolume.fileName);

     //std::cout << boundaryVolume.fileName << std::endl;
     //std::cout << min << " " << max << " -> " << dimension << std::endl;
     //std::cout << centerOfMass << " : " << boundaryVolume.position << std::endl;
     //std::cout << boundaryVolume.angle << " : " << boundaryVolume.velocity << std::endl;

    auto fileName = resolveFile(boundaryVolume.fileName, {get<parameters::internal::config_folder>()});
    std::string vdbFile = fileName.string();
    vdb::initialize();
    vdb::io::File file(vdbFile);
    file.open();
    auto grids = file.getGrids();
    vdb::GridBase::Ptr baseGrid;
    for (vdb::io::File::NameIterator nameIter = file.beginName(); nameIter != file.endName(); ++nameIter)
      if (nameIter.gridName() == "surface")
        baseGrid = file.readGrid(nameIter.gridName());
    file.close();
    vdb::FloatGrid::Ptr grid = vdb::gridPtrCast<vdb::FloatGrid>(baseGrid);

    vdb::CoordBBox box = grid->evalActiveVoxelBoundingBox();
    auto minVDB = grid->indexToWorld(box.getStart()); //    -openvdb::Vec3d(0.5, 0.5, 0.5);
    auto maxVDB = grid->indexToWorld(box.getEnd());   // + openvdb::Vec3d(0.5, 0.5, 0.5);
    auto voxelSize = grid->voxelSize();

    //gvdbVolumeManagers.emplace_back(VolumeGVDB());
    //auto &gvdb = gvdbVolumeManagers[gvdbVolumeManagers.size() - 1];
    //gvdb.SetVerbose(true); // enable/disable console output from gvdb
    //gvdb.SetCudaDevice(GVDB_DEV_FIRST);
    //gvdb.Initialize();
    //gvdb.AddPath(parameters::internal::working_directory::ptr->c_str());
    //gvdb.AddPath(parameters::internal::binary_directory::ptr->c_str());
    //gvdb.AddPath(parameters::internal::source_directory::ptr->c_str());
    //gvdb.AddPath(parameters::internal::build_directory::ptr->c_str());
    //gvdb.AddPath(parameters::internal::config_folder::ptr->c_str());

    // Load VDB
    //char scnpath[1024];
    //if (!gvdbInstance.getScene()->FindFile(fileName.string().c_str(), scnpath)) {
    //  gprintf("Cannot find vdb file.\n");
    //  gerror();
    //}
    //printf("Loading VDB. %s\n", scnpath);
    //gvdbInstance.SetChannelDefault(16, 16, 1);
    //gvdbInstance.getScene()->mVClipMin = nvdb::Vector3DF(-1e10f, -1e10f, -1e10f);
    //gvdbInstance.getScene()->mVClipMax = nvdb::Vector3DF(1e10f, 1e10f, 1e10f);
    //if (!gvdbInstance.LoadVDB(scnpath)) { // Load OpenVDB format
    //  gerror();
    //}
    //gvdbInstance.PrepareVDB();
    //gvdbInstance.Measure(true);
    Matrix4x4 itow, wtoi;
    auto [gptr, offset, size] = gvdbHelper::loadIntoGVDB(boundaryVolume.fileName);
    gvdbVolumes.push_back(gptr);
    //std::cout << offset << " -> " << size << std::endl;
    //std::cout << "center of mass: " << centerOfMass << std::endl;
    gvdbOffsets.push_back(offset + centerOfMass * size);
    gvdbVoxelSizes.push_back(size);

    //auto ming = gvdbInstance.getWorldMin();
    //auto maxg = gvdbInstance.getWorldMax();
    //std::cout << "ming: " << ming.x << " " << ming.y << " " << ming.z << std::endl;
    //std::cout << "maxg: " << maxg.x << " " << maxg.y << " " << maxg.z << std::endl;

    //std::cout << "minv: " << minVDB.x() << " " << minVDB.y() << " " << minVDB.z() << std::endl;
    //std::cout << "maxv: " << maxVDB.x() << " " << maxVDB.y() << " " << maxVDB.z() << std::endl;

    //std::cout << "minb: " << box.getStart().x() << " " << box.getStart().y() << " " << box.getStart().z() << std::endl;
    //std::cout << "maxb: " << box.getEnd().x() << " " << box.getEnd().y() << " " << box.getEnd().z() << std::endl;

    textures.push_back(texture);
    minAABB.push_back(min);
    maxAABB.push_back(max);
    dims.push_back(dimension);
    centersOfMasses.push_back(float4{0.f, 0.f, 0.f, 0.f});
    volumes.push_back(centerOfMass.w);
    densities.push_back(boundaryVolume.density);
    centerOfMass.w = 0.f;
    positions.push_back(math::castTo<float4>(boundaryVolume.position) + centerOfMass);
    velocities.push_back(math::castTo<float4>(boundaryVolume.velocity));
    angularVelocities.push_back(float4{0.f, 0.f, 0.f, 0.f});
    auto angle = math::castTo<float4>(boundaryVolume.angle);
    angle = angle / 180.f * CUDART_PI_F;
    angles.push_back(angle);
    kinds.push_back(boundaryVolume.kind);

    inertia = inertia * boundaryVolume.density;
    inertia(4, 4) = 1.f;
    btMatrix3x3 M{inertia(0, 0), inertia(0, 1), inertia(0, 2), inertia(1, 0), inertia(1, 1),
                  inertia(1, 2), inertia(2, 0), inertia(2, 1), inertia(2, 2)};

    auto Mp = M.inverse();
    Matrix4x4 inertiaInverse = Matrix4x4(Mp[0][0], Mp[0][1], Mp[0][2], 0.f, Mp[1][0], Mp[1][1], Mp[1][2], 0.f, Mp[2][0],
                                         Mp[2][1], Mp[2][2], 0.f, 0.f, 0.f, 0.f, 1.f);

    // SVD::Mat3x3 M{
    //	inertia(0,0), inertia(0,1), inertia(0,2),
    //	inertia(1,0), inertia(1,1), inertia(1,2),
    //	inertia(2,0), inertia(2,1), inertia(2,2) };
    // auto svd = SVD::svd(M);
    // auto U = svd.U;
    // auto S = svd.S;
    // auto V = svd.V;
    // S.m_00 = (S.m_00 > 1e-6f ? 1.f / S.m_00 : 0.f);
    // S.m_11 = (S.m_11 > 1e-6f ? 1.f / S.m_11 : 0.f);
    // S.m_22 = (S.m_22 > 1e-6f ? 1.f / S.m_22 : 0.f);
    // S.m_01 = S.m_02 = S.m_10 = S.m_12 = S.m_20 = S.m_21 = 0.f;
    // auto Mp = V * S * U.transpose();
    // Matrix4x4 inertiaInverse = Matrix4x4(
    //	Mp.m_00, Mp.m_01, Mp.m_02, 0.f,
    //	Mp.m_10, Mp.m_11, Mp.m_12, 0.f,
    //	Mp.m_20, Mp.m_21, Mp.m_22, 0.f,
    //	0.f, 0.f,0.f,1.f
    //);

    inertias.push_back(inertia);
    inertiasInv.push_back(inertiaInverse);

    // std::cout << boundaryVolume.fileName.value << std::endl;
    // std::cout << "Inertia Matrix: " << std::endl;
    // inertia.print();
    // std::cout << "Inverse Inertia Matrix: " << std::endl;
    // inertiaInverse.print();
  }
  if (textures.size() > 0) {
    using dimensions = decltype(info<arrays::volumeBoundaryDimensions>());
    using Bvolumes = decltype(info<arrays::volumeBoundaryVolumes>());
    using boundaryMins = decltype(info<arrays::volumeBoundaryMin>());
    using boundaryMaxs = decltype(info<arrays::volumeBoundaryMax>());
    get<parameters::boundary_volumes::volumeBoundaryCounter>() = (int32_t)textures.size();

    cuda::memcpy(dimensions::ptr, dims.data(), sizeof(decltype(dims)::value_type) * dims.size(),
                 cudaMemcpyHostToDevice);
    cuda::memcpy(Bvolumes::ptr, textures.data(), sizeof(decltype(textures)::value_type) * textures.size(),
                 cudaMemcpyHostToDevice);
    cuda::memcpy(boundaryMins::ptr, minAABB.data(), sizeof(decltype(minAABB)::value_type) * minAABB.size(),
                 cudaMemcpyHostToDevice);
    cuda::memcpy(boundaryMaxs::ptr, maxAABB.data(), sizeof(decltype(maxAABB)::value_type) * maxAABB.size(),
                 cudaMemcpyHostToDevice);

    using Avolumes = decltype(info<arrays::volumeBoundaryVolume>());
    using Adensities = decltype(info<arrays::volumeBoundaryDensity>());
    cuda::memcpy(Avolumes::ptr, volumes.data(), sizeof(Avolumes::type) * volumes.size(), cudaMemcpyHostToDevice);
    cuda::memcpy(Adensities::ptr, densities.data(), sizeof(Adensities::type) * volumes.size(), cudaMemcpyHostToDevice);

    using AKinds = decltype(info<arrays::volumeBoundaryKind>());
    cuda::memcpy(AKinds::ptr, kinds.data(), sizeof(AKinds::type) * kinds.size(), cudaMemcpyHostToDevice);

    using Apositions = decltype(info<arrays::volumeBoundaryPosition>());
    cuda::memcpy(Apositions::ptr, positions.data(), sizeof(Apositions::type) * volumes.size(), cudaMemcpyHostToDevice);
    using Avelocities = decltype(info<arrays::volumeBoundaryVelocity>());
    cuda::memcpy(Avelocities::ptr, velocities.data(), sizeof(Avelocities::type) * volumes.size(),
                 cudaMemcpyHostToDevice);
    using AangularVelocities = decltype(info<arrays::volumeBoundaryAngularVelocity>());
    cuda::memcpy(AangularVelocities::ptr, angularVelocities.data(), sizeof(AangularVelocities::type) * volumes.size(),
                 cudaMemcpyHostToDevice);

    std::vector<Matrix4x4> transforms;
    std::vector<Matrix4x4> iTransforms;
    std::vector<float4> quaternions;
    std::vector<float4> forces;
    std::vector<float4> torques;
    for (int32_t i = 0; i < volumes.size(); ++i) {
      Matrix4x4 Q = Matrix4x4::fromQuaternion(eul2quat(math::castTo<float3>(angles[i])));
      Matrix4x4 T = Matrix4x4::fromTranspose(positions[i]);
      auto M = T * Q;
      auto invM = M.inverse();
      transforms.push_back(M);
      inertias[i] = inertias[i];
      inertiasInv[i] = inertiasInv[i];
      iTransforms.push_back(invM);
      quaternions.push_back(eul2quat(math::castTo<float3>(angles[i])));
      forces.push_back(float4{0.f, 0.f, 0.f, 0.f});
      torques.push_back(float4{0.f, 0.f, 0.f, 0.f});
    }
    using AForces = decltype(info<arrays::volumeBoundaryAcceleration>());
    cuda::memcpy(AForces::ptr, forces.data(), sizeof(AForces::type) * volumes.size(), cudaMemcpyHostToDevice);
    using ATorques = decltype(info<arrays::volumeBoundaryAngularAcceleration>());
    cuda::memcpy(ATorques::ptr, torques.data(), sizeof(ATorques::type) * volumes.size(), cudaMemcpyHostToDevice);
    using Aquaternions = decltype(info<arrays::volumeBoundaryQuaternion>());
    cuda::memcpy(Aquaternions::ptr, quaternions.data(), sizeof(Aquaternions::type) * volumes.size(),
                 cudaMemcpyHostToDevice);
    
    using gvdbvols = decltype(info<arrays::volumeBoundaryGVDBVolumes>());
    cuda::memcpy(gvdbvols::ptr, gvdbVolumes.data(), sizeof(decltype(gvdbVolumes)::value_type) * gvdbVolumes.size(),
                 cudaMemcpyHostToDevice);
    using Awtransforms = decltype(info<arrays::gvdbOffsets>());
    using AwinverseTransforms = decltype(info<arrays::gvdbVoxelSizes>());
    cuda::memcpy(Awtransforms::ptr, gvdbOffsets.data(), sizeof(Awtransforms::type) * volumes.size(),
                 cudaMemcpyHostToDevice);
    cuda::memcpy(AwinverseTransforms::ptr, gvdbVoxelSizes.data(), sizeof(AwinverseTransforms::type) * volumes.size(),
                 cudaMemcpyHostToDevice);

    using Atransforms = decltype(info<arrays::volumeBoundaryTransformMatrix>());
    using AinverseTransforms = decltype(info<arrays::volumeBoundaryTransformMatrixInverse>());
    cuda::memcpy(Atransforms::ptr, transforms.data(), sizeof(Atransforms::type) * volumes.size(),
                 cudaMemcpyHostToDevice);
    cuda::memcpy(AinverseTransforms::ptr, iTransforms.data(), sizeof(AinverseTransforms::type) * volumes.size(),
                 cudaMemcpyHostToDevice);

    using AInertia = decltype(info<arrays::volumeBoundaryInertiaMatrix>());
    using AinverseInertia = decltype(info<arrays::volumeBoundaryInertiaMatrixInverse>());
    cuda::memcpy(AInertia::ptr, inertias.data(), sizeof(Atransforms::type) * volumes.size(), cudaMemcpyHostToDevice);
    cuda::memcpy(AinverseInertia::ptr, inertiasInv.data(), sizeof(AinverseTransforms::type) * volumes.size(),
                 cudaMemcpyHostToDevice);
    // for (auto i : inertias)
    //	i.print();

  } else {
    using dimensions = decltype(info<arrays::volumeBoundaryDimensions>());
    using volumes = decltype(info<arrays::volumeBoundaryVolumes>());
    using boundaryMins = decltype(info<arrays::volumeBoundaryMin>());
    using boundaryMaxs = decltype(info<arrays::volumeBoundaryMax>());

    get<parameters::boundary_volumes::volumeBoundaryCounter>() = 0;
  }
}

void IO::config::defaultAllocate() {
  for_each(allocations_list, [](auto x) {
    if (!decltype(x)::valid())
      return;
    decltype(x)::leanAllocate();
    // decltype(x)::defaultAllocate();
    logger(log_level::debug) << "Array: " << decltype(x)::variableName << " valid: " << decltype(x)::valid()
                             << " size: " << IO::config::bytesToString(decltype(x)::alloc_size) << std::endl;
    cuda::error_check(std::string("Checking for ") + decltype(x)::variableName);
  });
  // arrays::fluidDensity::defaultAllocate();
  //  if (get<parameters::modules::rayTracing>() == true)
  //	  arrays::previousPosition::defaultAllocate();
}

void IO::config::defaultRigidAllocate() {
  for_each(allocations_list, [](auto x) {
    if (decltype(x)::kind != memory_kind::rigidData && decltype(x)::kind != memory_kind::spareData) {
      // std::cout << "declined name: " << decltype(x)::variableName << " and type: " << (int)decltype(x)::kind <<
      // std::endl;
      return;
    }

    decltype(x)::defaultAllocate();
    // decltype(x)::defaultAllocate();
    logger(log_level::debug) << "Array: " << decltype(x)::variableName << " valid: " << decltype(x)::valid()
                             << " size: " << IO::config::bytesToString(decltype(x)::alloc_size) << std::endl;
  });
}

void IO::config::initSnapshot() {
    snaps.push_back(new UniformSnap());
  //for_each(uniforms_list, [](auto x) { snaps.push_back(new UniformSnap<decltype(x)>()); });
  for_each(sorting_list, [](auto x) { snaps.push_back(new ArraySnap<decltype(x)>()); });
  for_each(individual_list, [](auto x) { snaps.push_back(new ArraySnap<decltype(x)>()); });
  if (get<parameters::boundary_volumes::volumeBoundaryCounter>() == 0 && get<parameters::rigid_volumes::volume>().size())
    snaps.push_back(new RigidSnap());
}
