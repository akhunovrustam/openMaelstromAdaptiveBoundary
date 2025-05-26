#define BOOST_USE_WINDOWS_H
#include <IO/renderData/particle.h>
#include <utility/cuda.h>
#include <iostream>
#include <utility/identifier/arrays.h>
#include <utility/identifier/uniform.h>
#include <math/math.h>
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
#include <simulation/particleSystem.h>
#include <tools/pathfinder.h>
#include <yaml-cpp/yaml.h>
#include <sstream>

namespace IO {
void renderData::saveParticles() {
	std::lock_guard<std::mutex> guard(cuda_particleSystem::instance().simulation_lock);
	static int32_t frame = 0;
	std::stringstream sstream;
	std::string renderDir = (arguments::cmd::instance().renderDirectory).string();
	renderDir = arguments::cmd::instance().outputDirectory.string();
	if (renderDir == "")
		renderDir = "testRender";
	if (!fs::exists(renderDir))
		fs::create_directory(renderDir);
	sstream << renderDir << "/frame_"<< std::setw(4) << std::setfill('0') << frame++;
	auto fileName = sstream.str();

	std::ofstream data;
	data.open(fileName + ".ptcl", std::ios::binary);
	//std::clog << "Writing particle data to " << fileName << ".ptcl" << std::endl;
	auto writeArray = [&](std::string str, auto* ptr, auto n ) {
		using P = std::decay_t<decltype(*ptr)>;
		auto size = (int32_t)str.size();
		//data.write(reinterpret_cast<char*>(&size), sizeof(size));
		//data.write(str.c_str(), size);
		std::size_t elemSize = sizeof(P);
		std::size_t allocSize = elemSize * n;
		//data.write(reinterpret_cast<char*>(&elemSize), sizeof(elemSize));
		//data.write(reinterpret_cast<char*>(&allocSize), sizeof(allocSize));

		void* tmp = malloc(allocSize);
		cudaMemcpy(tmp, ptr, allocSize, cudaMemcpyDeviceToHost);
		data.write(reinterpret_cast<char*>(tmp), allocSize);
		//std::cout << "Writing " << str << " with elemSize " << elemSize << " and allocSize " << allocSize << " for n = " << n << " elements" << std::endl;
		free(tmp);
	};
	
	writeArray("pos", arrays::position::ptr, get<parameters::internal::num_ptcls>());
	writeArray("vol", arrays::volume::ptr, get<parameters::internal::num_ptcls>());
	writeArray("vel", arrays::velocity::ptr, get<parameters::internal::num_ptcls>());
	writeArray("ren", arrays::renderArray::ptr, get<parameters::internal::num_ptcls>());
	if (get<parameters::boundary_volumes::volume>().size() > 0) {
		auto b = get<parameters::boundary_volumes::volume>().size();
		writeArray("bnd", arrays::volumeBoundaryTransformMatrix::ptr, b);
		writeArray("bnd", arrays::volumeBoundaryTransformMatrixInverse::ptr, b);
		writeArray("bnd", arrays::volumeBoundaryPosition::ptr, b);
		writeArray("bnd", arrays::volumeBoundaryVolume::ptr, b);
		writeArray("bnd", arrays::volumeBoundaryDensity::ptr, b);
		writeArray("bnd", arrays::volumeBoundaryKind::ptr, b);
	}
	data.close();

	YAML::Node root;
	auto addParameter = [&](std::string parameter) {
		auto& p = ParameterManager::instance().getParameter(parameter);
		auto ns = p.identifierNamespace;
		auto id = p.identifier;
		if(ns != "")
			root[ns][id] = ParameterManager::instance().encoders[p.type](p.param.isVec ? p.param.valVec.value() : p.param.val.value());
		else 
			root[id] = ParameterManager::instance().encoders[p.type](p.param.isVec ? p.param.valVec.value() : p.param.val.value());
	};

	addParameter("hash_entries");
	addParameter("num_ptcls");
	addParameter("color_map.min");
	addParameter("color_map.max");
	addParameter("vrtxRenderDomainMin");
	addParameter("vrtxRenderDomainMax");
	addParameter("frame");
	addParameter("simulationTime");
	addParameter("timestep");
	addParameter("min_domain");
	addParameter("min_coord");
	addParameter("max_domain");
	addParameter("max_coord");
	addParameter("ptcl_support");
	addParameter("ptcl_spacing");
	addParameter("dfsph_settings.densityError");
	addParameter("dfsph_settings.densitySolverIterations");
	addParameter("dfsph_settings.divergenceError");
	addParameter("dfsph_settings.divergenceSolverIterations");


	YAML::Emitter out;
	out << root;
	auto fYaml = fileName + ".yaml";
	//std::clog << "Writing config to" << fYaml << std::endl;
	//std::clog << out.c_str();
	std::ofstream config_file;
	config_file.open(fYaml);
	config_file << out.c_str();

	config_file.close();
	if (frame == 1) {
		YAML::Node root;
		root = ParameterManager::instance().buildTree();
		YAML::Emitter out;
		out << root;
		std::stringstream sstream;
		sstream << renderDir << "/base";
		auto fYaml = sstream.str() + ".yaml";
		//std::clog << "Writing config to" << fYaml << std::endl;
		//std::clog << out.c_str();
		std::ofstream config_file;
		config_file.open(fYaml);
		config_file << out.c_str();
		
		fs::copy_file(
			get<parameters::internal::config_file>(), 
			renderDir + "/" + "config.yaml",
			fs::copy_options::overwrite_existing);

	}

	//std::clog << "Done" << std::endl;


}
} // namespace IO
