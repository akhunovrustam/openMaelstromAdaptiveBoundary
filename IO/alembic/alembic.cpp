#define BOOST_USE_WINDOWS_H
#include <IO/alembic/alembic.h>
#include <utility/cuda.h>
#include <iostream>
#include <utility/identifier/arrays.h>
#include <utility/identifier/uniform.h>
#include <math/math.h>
#ifdef _WIN32
#pragma warning(push)
#pragma warning(disable : 4275)
#endif
#include <Alembic/AbcCoreOgawa/All.h>
#include <Alembic/AbcGeom/All.h>
#include <boost/filesystem.hpp>
#ifdef _WIN32
#pragma warning(pop)
#endif
#include <boost/algorithm/string.hpp>
#include <cuda_runtime.h>
#ifdef _WIN32
#include <filesystem>
namespace fs = std::filesystem;
#else
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;
#endif
#include <fstream>
#include <tools/log.h>
#include <utility/cuda.h>

using namespace Alembic::AbcGeom;

void parseProperty(const Abc::ICompoundProperty &compound, std::vector<Abc::IArrayProperty> &arrays,
                   std::vector<Abc::IScalarProperty> &scalars) {
  for (size_t i = 0; i < compound.getNumProperties(); i++) {
    Abc::PropertyHeader header = compound.getPropertyHeader(i);
    if (header.isScalar()) {
      scalars.push_back(Abc::IScalarProperty(compound, header.getName()));
    } else if (header.isArray()) {
      arrays.push_back(Abc::IArrayProperty(compound, header.getName()));
    } else {
      parseProperty(Abc::ICompoundProperty(compound, header.getName()), arrays, scalars);
    }
  }
}

namespace IO {

void alembic::load_particles(std::string, uint32_t) {
  uint32_t num_ptcls = 0;
  {
    std::string filename =
        R"(Z:\Configurations\DamBreakObstacle\export\alembic4\particles_140.abc)";

    IArchive archive = IArchive(Alembic::AbcCoreOgawa::ReadArchive(), filename);
    auto top = archive.getTop();
    auto particles = top.getChild("svtParticles");
    auto props = particles.getProperties();
    std::vector<Alembic::AbcGeom::Abc::IArrayProperty> arrays;
    std::vector<Abc::IScalarProperty> scalars;
    parseProperty(props, arrays, scalars);

    std::vector<float3> positions;
    std::vector<float> scales;
    std::vector<float3> velocities;
    std::vector<float> densities;

    for (const auto &a : arrays) {
      // a.getNumSamples() << std::endl;
      Alembic::AbcCoreAbstract::ArraySamplePtr samp;
      a.get(samp, 0);
      if (a.getName() == "P") {
        positions.resize(samp->size());
        memcpy(positions.data(), samp->getData(), samp->size() * samp->getDataType().getNumBytes());
      }
      if (a.getName() == "s") {
        scales.resize(samp->size());
        memcpy(scales.data(), samp->getData(), samp->size() * samp->getDataType().getNumBytes());
      }
      if (a.getName() == ".velocities") {
        velocities.resize(samp->size());
        memcpy(velocities.data(), samp->getData(),
               samp->size() * samp->getDataType().getNumBytes());
      }
      if (a.getName() == "density") {
        densities.resize(samp->size());
        memcpy(densities.data(), samp->getData(), samp->size() * samp->getDataType().getNumBytes());
      }
    }

    std::vector<float4> particle_data;
    particle_data.resize(positions.size());
    for (uint32_t it = 0; it < positions.size(); ++it) {
      particle_data[it] = float4{positions[it].x, positions[it].y, positions[it].z, scales[it]};
      // std::swap(particle_data[it].y, particle_data[it].z);
    }

    num_ptcls = (uint32_t)positions.size();

    memcpy(get<arrays::position>(), particle_data.data(), num_ptcls * sizeof(float4));
  }

  get<parameters::internal::num_ptcls>() = num_ptcls;
} // namespace alembic

Alembic::Util::uint32_t tsidx;

void add_int(Alembic::Abc::OCompoundProperty &params, std::vector<int32_t> data,
	std::string identifier, GeometryScope scope = kVertexScope) {
	using namespace Alembic::AbcGeom;
	OInt32GeomParam scalOut(params, identifier, false, scope, 1, tsidx);
	auto scalSample = OInt32GeomParam::Sample(Int32ArraySample(data), scope);
	scalOut.set(scalSample);
};
void add_float(Alembic::Abc::OCompoundProperty &params, std::vector<float> data,
               std::string identifier, GeometryScope scope = kVertexScope) {
  using namespace Alembic::AbcGeom;
  OFloatGeomParam scalOut(params, identifier, false, scope, 1, tsidx);
  auto scalSample = OFloatGeomParam::Sample(FloatArraySample(data), scope);
  scalOut.set(scalSample);
};
void add_float3(Alembic::Abc::OCompoundProperty &params, std::vector<V3f> data,
                std::string identifier, GeometryScope scope = kVertexScope) {
  using namespace Alembic::AbcGeom;
  OV3fGeomParam scalOut(params, identifier, false, scope, 1, tsidx);
  auto scalSample = OV3fGeomParam::Sample(V3fArraySample(data), scope);
  scalOut.set(scalSample);
};


template <typename info, typename C> auto load_values(C fn) {
  using T = typename info::type;
  auto original_ptr = info::ptr;
  auto last_ptr = (T *)malloc(info::alloc_size);
  auto allocSize = info::alloc_size;
  using result_t = decltype(fn(*last_ptr));
  std::vector<result_t> vals;

  cuda::memcpy(last_ptr, original_ptr, allocSize, cudaMemcpyDeviceToHost);

  for (int32_t it = 0; it < get<parameters::internal::num_ptcls>(); ++it) {
    vals.push_back(fn(last_ptr[it]));
  }
  free(last_ptr);
  return vals;
};

template <typename info, typename info2, typename C> auto load_values1(C fn) {
  using T = typename info::type;
  auto original_ptr = info::ptr;
  auto last_ptr = (T *)malloc(info::alloc_size);
  auto allocSize = info::alloc_size;
  using T2 = typename info2::type;
  auto original_ptr2 = info2::ptr;
  auto last_ptr2 = (T2 *)malloc(info2::alloc_size);
  auto allocSize2 = info2::alloc_size;
  using result_t = decltype(fn(*last_ptr));
  std::vector<result_t> vals;

  cuda::memcpy(last_ptr, original_ptr, allocSize, cudaMemcpyDeviceToHost);
  cuda::memcpy(last_ptr2, original_ptr2, allocSize2, cudaMemcpyDeviceToHost);

  for (int32_t it = 0; it < get<parameters::internal::num_ptcls>(); ++it) {
      if (last_ptr2[it] == 0)
    vals.push_back(fn(last_ptr[it]));
  }
  free(last_ptr);
  free(last_ptr2);
  return vals;
};

void alembic::save_particles() {
  static int frame = 0;
  static float last_time = 0.f - 1.f / get<parameters::alembic::fps>();
  if (get<parameters::internal::simulationTime>() <= last_time + 1.f / get<parameters::alembic::fps>())
    return;
  last_time += 1.f / get<parameters::alembic::fps>();

  
  logger(log_level::info) << "Exporting data as Alembic." << std::endl;
  fs::path path(get<parameters::internal::folderName>() + "/" + get<parameters::alembic::file_name>());
  std::string fileName = path.filename().string();
  if (fileName.find("$f") != std::string::npos) {
    boost::replace_all(fileName, "$f", std::to_string(frame));
  } else
    fileName.append(std::to_string(frame));
    #ifndef _WIN32
std::string dir = path.parent_path().string();
std::string ext = path.extension().string();
path = dir;
path.append(fileName);
path.replace_extension(ext);

#else
  path.replace_filename(fileName);
#endif
  auto positionVec =
      load_values1<arrays::position, arrays::particle_type>([](auto input) { return V3f(input.x, input.y, input.z); });
  auto velocityVec =
      load_values1<arrays::velocity, arrays::particle_type>([](auto input) { return V3f(input.x, input.y, input.z); });
  auto scaleVec = load_values1<arrays::volume, arrays::particle_type>([](auto input) {
    return (Alembic::Util::float32_t)powf(input / (4.f / 3.f * CUDART_PI_F), 1.f / 3.f);
  });
  auto hVec = load_values1<arrays::position, arrays::particle_type>([](auto input) {
	  return input.w;
  });
  auto densityVec = load_values1<arrays::density, arrays::particle_type>([](auto input) {
    return (Alembic::Util::float32_t) input;
  });
  auto neighVec = load_values1<arrays::neighborListLength, arrays::particle_type>([](auto input) {
    return (Alembic::Util::float32_t) input;
  });
  std::vector<Alembic::Util::uint64_t> idVec;
  for (int32_t it = 0; it < get<parameters::internal::num_ptcls>(); ++it)
    idVec.push_back(it);

  OArchive archive = OArchive(Alembic::AbcCoreOgawa::WriteArchive(), path.string());
  tsidx = archive.getTop().getArchive().addTimeSampling(
      TimeSampling(get<parameters::alembic::fps>(), 0.0));
  Alembic::AbcGeom::OPoints partsOut = OPoints(archive.getTop(), "svtParticles", tsidx);
  OPointsSchema &pSchema = partsOut.getSchema();
  auto sample = OPointsSchema::Sample(V3fArraySample(positionVec), UInt64ArraySample(idVec),
                                      V3fArraySample(velocityVec));
  pSchema.set(sample);
  Abc::OCompoundProperty argGeomParams = pSchema.getArbGeomParams();
  add_float(argGeomParams, scaleVec, "s");
  add_float(argGeomParams, hVec, "h");
  add_float(argGeomParams, densityVec, "density");
  add_float(argGeomParams, neighVec, "neighbors");


  add_int(argGeomParams, std::vector<int32_t>{ get<parameters::internal::num_ptcls>()}, "num_ptcls", kConstantScope);
  add_int(argGeomParams, std::vector<int32_t>{ get<parameters::simulation_settings::maxNumptcls>()}, "max_numptcls", kConstantScope);
  add_float(argGeomParams, std::vector<float>{ get<parameters::adaptive::resolution>()}, "ratio", kConstantScope);
  add_float(argGeomParams, std::vector<float>{ get<parameters::internal::timestep>()}, "timestep", kConstantScope);
  add_int(argGeomParams, std::vector<int32_t>{ get<parameters::internal::frame>()}, "frame", kConstantScope);
  add_float(argGeomParams, std::vector<float>{ get<parameters::internal::simulationTime>()}, "simulationTime", kConstantScope);
  add_float(argGeomParams, std::vector<float>{ get<parameters::particle_settings::radius>()}, "radius", kConstantScope);
  add_float(argGeomParams, std::vector<float>{ get<parameters::internal::ptcl_support>()}, "support", kConstantScope);

  logger(log_level::info) << "Exporting data as Alembic done." << std::endl;

  ++frame;
}

} // namespace IO
