//#define NO_OPERATORS
#define BOOST_USE_WINDOWS_H
#include <IO/vdb/vdb.h>
#include <iostream>
#include <utility/identifier.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <fstream>

#include <iostream>
#include <fstream>

#include <utility/generation.h>
#include <utility/sdf.h>
#include <utility/bullet/DynamicsWorld.h>

//#include <utility/generation/optimized_generation.h>

void IO::vdb::emitParticleVolumes() {
  //emit fluid volumes
	int fluidcnt = 0;
   for (auto fluidVolume : get<parameters::particle_volumes::volume>()) {
	float devfactor = 1.0f;
	//float devfactor = 3.17f;
	//if (fluidcnt == 1) devfactor = 2.3f;
	fluidcnt++;
    auto r = get<parameters::particle_settings::radius>() / devfactor;
    auto volume = PI4O3 * math::power<3>(r);
	auto scale = fluidVolume.scale;
	auto shift = fluidVolume.shift;
    auto out_points = generation::generateParticles(fluidVolume.fileName, r, genTechnique::hex_grid, false, scale);

    auto inserted_particles = (int32_t) out_points.size();

    int32_t old_ptcls = get<parameters::internal::num_ptcls>();
    if (old_ptcls + inserted_particles > get<parameters::simulation_settings::maxNumptcls>()) {
      std::cerr << "Not enough memory to insert particles " << inserted_particles << std::endl;
      continue;
    }

    get<parameters::internal::num_ptcls>() += inserted_particles;
    get<parameters::internal::num_ptcls_fluid>() += inserted_particles;
	
#ifdef UNIFIED_MEMORY
    for (int32_t i = old_ptcls; i < old_ptcls + inserted_particles; ++i) {
      openvdb::Vec4f ptcl_position = out_points[i - old_ptcls];
      get<arrays::position>()[i] =
          float4{ptcl_position.x(), ptcl_position.y(), ptcl_position.z(), ptcl_position.w()};
      get<arrays::velocity>()[i] = float4{0.f, 0.f, 0.f, 0.f};

      get<arrays::volume>()[i] = volume;
    }
#else
	std::vector<float4> positions;
	std::vector<float4> velocities;
	std::vector<float> volumes;
	std::vector<int> particle_type;
	std::vector<int> tobeoptimized;
	std::vector<int> uid;
	std::vector<int> particle_type_x;

  bool first = false;
	for (int32_t i = old_ptcls; i < old_ptcls + inserted_particles; ++i) {
		openvdb::Vec4f ptcl_position = out_points[i - old_ptcls];
		//std::cout << "smooth " << ptcl_position.w();
		positions.push_back(float4{ ptcl_position.x() + shift.x, ptcl_position.y() + shift.y, ptcl_position.z()+(first ? 20 : 0) + shift.z, ptcl_position.w() });
		velocities.push_back(float4{ fluidVolume.velocity.x,fluidVolume.velocity.y,fluidVolume.velocity.z, 0.f });
		volumes.push_back(volume);
		tobeoptimized.push_back(0);
		uid.push_back(i);
    particle_type.push_back(FLUID_PARTICLE);
    // particle_type_x.push_back(FLUID_PARTICLE);
    particle_type_x.push_back(first ? 1 : FLUID_PARTICLE);
    first = false;
	}
	cudaMemcpy(arrays::position::ptr + old_ptcls, positions.data(), inserted_particles * sizeof(float4), cudaMemcpyHostToDevice);
	cudaMemcpy(arrays::velocity::ptr + old_ptcls, velocities.data(), inserted_particles * sizeof(float4), cudaMemcpyHostToDevice);
	cudaMemcpy(arrays::volume::ptr + old_ptcls, volumes.data(), inserted_particles * sizeof(float), cudaMemcpyHostToDevice);
//#ifdef RIGID_PARTICLE_SUPPORT
	cudaMemcpy(arrays::particle_type::ptr + old_ptcls, particle_type.data(), inserted_particles * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(arrays::tobeoptimized::ptr + old_ptcls, tobeoptimized.data(), inserted_particles * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(arrays::uid::ptr + old_ptcls, uid.data(), inserted_particles * sizeof(int), cudaMemcpyHostToDevice);
//#endif
	//cudaMemcpy(arrays::particle_type_x::ptr + old_ptcls, particle_type_x.data(), inserted_particles * sizeof(float), cudaMemcpyHostToDevice);
#endif
  }
  
   
   std::vector<float> rigidbody_sdf;
   std::vector<float> densities;
   std::vector<float4> rigid_velocities;
   std::vector<float3> rigid_avelocities;
   std::vector<float> rigid_volumes;
   std::vector<float3> rigid_origins;
   std::vector<float4> rigid_quaternions;
   std::vector<std::string> rigid_files;
   std::vector<float> sdf_gridsize;
   std::vector<float4> sdf_minpoint1;
   std::cout << "rigids " << get<parameters::rigid_volumes::mesh_resolution>() << std::endl;
   std::cout << "rigids " << get<parameters::rigid_volumes::volume>().size() << std::endl;
   std::cout << "Raytracing " << get<parameters::modules::rayTracing>() << std::endl;
   get<parameters::particle_settings::sdf_resolution>() = 100;
   get<parameters::particle_settings::sdf_epsilon>() = 0.01;
   if (get<parameters::boundary_volumes::volumeBoundaryCounter>() == 0 && get<parameters::rigid_volumes::volume>().size() > 0) {
	   int rigid_nums = 1;
	   for (auto fluidVolume : get<parameters::rigid_volumes::volume>()) {
		   densities.push_back(fluidVolume.density);
		   rigid_velocities.push_back({ 0.f, 0.f, 0.f, 0.f });
		   rigid_avelocities.push_back({ 0.f, 0.f, 0.f });
		   rigid_quaternions.push_back({ 0.f, 0.f, 0.f, 1.f });
		   rigid_files.push_back(fluidVolume.fileName);

		   auto sht = fluidVolume.shift;
		   auto scale = fluidVolume.scale;
		   //auto scale1 = fluidVolume.s;
		   auto init_velocity = fluidVolume.init_velocity;
		   auto sampling = fluidVolume.sampling;
		   // std::cout << "saved origin: " << (maxmin["max"].x+maxmin["min"].x)/2 << " " << 
		   //   (maxmin["max"].y+maxmin["min"].y)/2 << " " << (maxmin["max"].z+maxmin["min"].z)/2 << std::endl;
		   //TODO: combine above and below functions and make a review on the body of function below

		   //auto r = get<parameters::particle_settings::radius>() / 3.17f;
		   auto r = get<parameters::particle_settings::radius>() / 1.0f;
		   auto volume = PI4O3 * math::power<3>(r);
		   std::cout << "VOL " << volume << std::endl;
		   //auto out_points = generation::generateParticlesRigid(fluidVolume.fileName, r, genTechnique::optimized_density, false, scale);
		   auto out_points = generation::generateParticlesRigid(fluidVolume.fileName, r, genTechnique::optimized_impulse, false, scale);
		   //out_points.erase(out_points.begin() + 100, out_points.end());
		   std::cout << "first point " << out_points[0].x() << " " << out_points[0].y() << " " << out_points[0].z() << " " << out_points[0].w() << "\n";
		   //out_points = generation::ImpulseOptimize(volume, out_points);
		   //out_points = generation::DensityOptimize(volume, out_points);
		   
		   auto path = resolveFile(fluidVolume.fileName.c_str(), {get<parameters::internal::config_folder>()});
		   auto [objVertices, objPlanes, objEdges, objMin, objMax] = generation::fileToObj(path);
		   std::cout << "afthermath" << "\n";

		   auto lnx = (objMax.x - objMin.x);
		   auto lny = (objMax.y - objMin.y);
		   auto lnz = (objMax.z - objMin.z);

		  

		   float mx = lnx;
		   if (mx < lny) mx = lny;
		   if (mx < lnz) mx = lnz;
		   std::cout << "mx " << mx << std::endl;
		   
		   float grdsize = mx / get<parameters::particle_settings::sdf_resolution>();
		   sdf_gridsize.push_back(grdsize);
		   auto sdf = sdf::meshToSDF(objVertices, objPlanes, grdsize, get<parameters::particle_settings::sdf_resolution>(), objMin);
		   std::cout << "grdsize " << objPlanes.size() << "\n";
		   for (int ii = 0; ii < (get<parameters::particle_settings::sdf_resolution>() + 20) * (get<parameters::particle_settings::sdf_resolution>() + 20) * (get<parameters::particle_settings::sdf_resolution>() + 20); ii++)
		   {
			   //std::cout << "ii " << ii << " sdf " << sdf[ii] << std::endl;
			   rigidbody_sdf.push_back(sdf[ii]);
		   }

		   float3 apoint = { 0, 0, 0 };
		   
		   float shiftsdf = grdsize * 10;
		   float3 minpoint{ objMin.x - shiftsdf, objMin.y - shiftsdf, objMin.z - shiftsdf};
		   //get<parameters::particle_settings::sdf_minpoint>() = {minpoint.x, minpoint.y, minpoint.z, 0.f};
		   float4 minpoint1 = {minpoint.x, minpoint.y, minpoint.z, 0.f};
		   sdf_minpoint1.push_back(minpoint1);
		   float* sdff = rigidbody_sdf.data();
		   std::cout << "sdftest " << sdff[(120*120*120-1) * rigid_nums] << "\n";
		   auto ppoint = sdf::lookupSDF1(sdff + 120*120*120 * (rigid_nums - 1), apoint, grdsize, get<parameters::particle_settings::sdf_resolution>(), minpoint);
		   
			std::cout << "dist " << ppoint << " parttype " << rigid_nums << std::endl;
			
			/*auto projpoint = sdf::projectOntoMesh1(rigidbody_sdf.data(), apoint, grdsize, get<parameters::particle_settings::sdf_resolution>(), 
			   get<parameters::particle_settings::sdf_epsilon>(), 10, minpoint);
       
			std::cout << "projpoint " << projpoint << std::endl;*/
		    /*for(auto v : objVertices){
				vdbVertices.push_back(vdb::Vec3s(v.x, v.y, v.z));
			}
			for(auto t : objPlanes){
				vdbIndices.push_back(vdb::Vec3I(t.i0, t.i1, t.i2));
			}*/

		   auto inserted_particles = (int32_t)out_points.size();

		    std::ofstream myfile;
			myfile.open ("points.txt");
			for (auto pt : out_points)
			{
				myfile << pt.x() << " " << pt.y() << " " << pt.z() << " " << pt.w() << "\n";
			}
			myfile.close();

		   std::cout << "rigid parts " << inserted_particles << std::endl;

		   int32_t old_ptcls = get<parameters::internal::num_ptcls>();
		   if (old_ptcls + inserted_particles > get<parameters::simulation_settings::maxNumptcls>()) {
			   std::cerr << "Not enough memory to insert particles." << std::endl;
			   continue;
		   }

		   get<parameters::internal::num_ptcls>() += inserted_particles;



#ifdef UNIFIED_MEMORY
		   for (int32_t i = old_ptcls; i < old_ptcls + inserted_particles; ++i) {
			   openvdb::Vec4f ptcl_position = out_points[i - old_ptcls];
			   get<arrays::position>()[i] =
				   float4{ ptcl_position.x(), ptcl_position.y(), ptcl_position.z(), ptcl_position.w() };
			   get<arrays::velocity>()[i] = float4{ 0.f, 0.f, 0.f, 0.f };

			   get<arrays::volume>()[i] = volume;
		   }
#else
		   std::vector<float4> positions;
		   std::vector<float4> velocities;
		   std::vector<float> volumes;
		   std::vector<int> particle_type;
		   std::vector<int> tobeoptimized;
		   std::vector<int> uid;
		   std::vector<int> particle_type_x;
		   // float4 center = {0, 0, 0, 0};
		   // for (int32_t i = 0; i < inserted_particles; ++i) {
		   // 	openvdb::Vec4f ptcl_position = out_points[i];
		   //   center += float4{ ptcl_position.x(), ptcl_position.y(), ptcl_position.z(), ptcl_position.w()};

		   // }

		   // center = center / inserted_particles;

		   auto shift = fluidVolume.shift;
		   for (int32_t i = old_ptcls; i < old_ptcls + inserted_particles; ++i) {
			   openvdb::Vec4f ptcl_position = out_points[i - old_ptcls];
			   positions.push_back(float4{ ptcl_position.x() + shift.x, ptcl_position.y() + shift.y, ptcl_position.z() + shift.z, ptcl_position.w() });
			   velocities.push_back(float4{ 0.f, 0.f, 0.f, 0.f });
			   particle_type.push_back(rigid_nums);
			   particle_type_x.push_back(1);
				tobeoptimized.push_back(0);
				uid.push_back(i);
			   volumes.push_back(volume);
			   /*if (i == 200)
			   {
				   volumes.back() = volume / 8;
				   positions.back().w = ptcl_position.w() / 2;
			   }*/
		   }
		   //for (auto& p_i : positions) {
			  // float v = 0.f;
			  // for (auto& p_j : positions) {
				 //  v += kernel(p_i, p_j);
			  // }
			  // volumes.push_back(get<parameters::rigid_volumes::gamma>() / v);
			  // // volumes.push_back(volume);

		   //}
		   //std::cout << "RIGIDF " << arrays::structureArrays::refinementList::ptr[0] << "\n";
		   //std::cout << "RIGIDF " << arrays::rigidbody_sdf::ptr[2] << "\n";
		   //std::cout << "rgidi sdf size " << rigidbody_sdf.size() << "\n";
		   //cudaMemcpy(arrays::rigidbody_sdf::ptr + old_ptcls, rigidbody_sdf.data(), inserted_particles * sizeof(float), cudaMemcpyHostToDevice);
		   //std::cout << "RIGIDF " << arrays::rigidbody_sdf::ptr[0] << "\n";
		   cudaMemcpy(arrays::position::ptr + old_ptcls, positions.data(), inserted_particles * sizeof(float4), cudaMemcpyHostToDevice);
		   std::cout << "RIGIDF " << arrays::position::ptr[0] << "\n";
		   cudaMemcpy(arrays::velocity::ptr + old_ptcls, velocities.data(), inserted_particles * sizeof(float4), cudaMemcpyHostToDevice);
		   cudaMemcpy(arrays::volume::ptr + old_ptcls, volumes.data(), inserted_particles * sizeof(float), cudaMemcpyHostToDevice);
		   cudaMemcpy(arrays::particle_type::ptr + old_ptcls, particle_type.data(), inserted_particles * sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(arrays::tobeoptimized::ptr + old_ptcls, tobeoptimized.data(), inserted_particles * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(arrays::uid::ptr + old_ptcls, uid.data(), inserted_particles * sizeof(int), cudaMemcpyHostToDevice);
		   //cudaMemcpy(arrays::particle_type_x::ptr + old_ptcls, particle_type_x.data(), inserted_particles * sizeof(float), cudaMemcpyHostToDevice);
#endif


  // DynamicsWorld::getInstance()->addBody(fluidVolume.fileName.value);
		   float total_vol = 0;
		   for (auto i = 0; i < inserted_particles; i++)
		   {
			   total_vol += volumes[i];
		   }

		   rigid_volumes.push_back(total_vol);
		   int32_t index = rigid_nums - 1;
		   rigid_nums++;
	   }
	   if (get<parameters::rigid_volumes::volume>().size()) {

		   int32_t rigidCount = get<parameters::rigid_volumes::volume>().size();
		   //DynamicsWorld::getInstance()->createBoundingBox();

		   arrays::rigidbody_sdf::allocate(sizeof(float) * rigidbody_sdf.size());
		   cudaMemcpy(arrays::rigidbody_sdf::ptr, rigidbody_sdf.data(), sizeof(float) * rigidbody_sdf.size(), cudaMemcpyHostToDevice);
		   std::cout << "RIGIDsdf size " << rigidbody_sdf.size() << std::endl;
		   //std::cout << "RIGID " << arrays::rigidbody_sdf::ptr[2] << "\n";
		   arrays::rigidOrigins::allocate(sizeof(float3) * rigidCount);
		   cudaMemcpy(arrays::rigidOrigins::ptr, rigid_origins.data(), sizeof(float3) * rigidCount, cudaMemcpyHostToDevice);


		   //arrays::rigidFiles::allocate(sizeof(std::string) * rigidCount);
		   //cudaMemcpy(arrays::rigidFiles::ptr, rigid_files.data(), sizeof(std::string) * rigidCount, cudaMemcpyHostToDevice);

		   //arrays::rigidQuaternions::allocate(sizeof(float4) * rigidCount);
		   //cudaMemcpy(arrays::rigidQuaternions::ptr, rigid_quaternions.data(), sizeof(float4) * rigidCount, cudaMemcpyHostToDevice);

		   /*arrays::rigidVolumes::allocate(sizeof(float) * rigidCount);
		   cudaMemcpy(arrays::rigidVolumes::ptr, rigid_volumes.data(), sizeof(float) * rigidCount, cudaMemcpyHostToDevice);
		   */
		   arrays::rigidDensities::allocate(sizeof(float) * rigidCount);
		   cudaMemcpy(arrays::rigidDensities::ptr, densities.data(), sizeof(float) * rigidCount, cudaMemcpyHostToDevice);
			
		   arrays::sdf_gridsize::allocate(sizeof(float) * rigidCount);
		   cudaMemcpy(arrays::sdf_gridsize::ptr, sdf_gridsize.data(), sizeof(float) * rigidCount, cudaMemcpyHostToDevice);
			
		   arrays::sdf_minpoint2::allocate(sizeof(float4) * rigidCount);
		   cudaMemcpy(arrays::sdf_minpoint2::ptr, sdf_minpoint1.data(), sizeof(float4) * rigidCount, cudaMemcpyHostToDevice);

		   arrays::rigidLinearVelocities::allocate(sizeof(float4) * rigidCount);
		   cudaMemcpy(arrays::rigidLinearVelocities::ptr, rigid_velocities.data(), sizeof(float4) * rigidCount, cudaMemcpyHostToDevice);

		   arrays::rigidAVelocities::allocate(sizeof(float3) * rigidCount);
		   cudaMemcpy(arrays::rigidAVelocities::ptr, rigid_avelocities.data(), sizeof(float3) * rigidCount, cudaMemcpyHostToDevice);

	   }
	   for (auto boundaryVolume : get<parameters::boundary_volumes::volume>()) {
		   //wrld->addBoundary(boundaryVolume.fileName);
	   }
   }
	   else {
		   rigid_origins.push_back({ 0.f, 0.f, 0.f });
		   arrays::rigidOrigins::allocate(sizeof(float3));
		   cudaMemcpy(arrays::rigidOrigins::ptr, rigid_origins.data(), sizeof(float3), cudaMemcpyHostToDevice);

		   rigid_files.push_back("");
		   //arrays::rigidFiles::allocate(sizeof(std::string));
		   //cudaMemcpy(arrays::rigidFiles::ptr, rigid_files.data(), sizeof(std::string), cudaMemcpyHostToDevice);

		   rigid_quaternions.push_back({ 0.f, 0.f, 0.f, 0.f });
		   arrays::rigidQuaternions::allocate(sizeof(float4));
		   cudaMemcpy(arrays::rigidQuaternions::ptr, rigid_quaternions.data(), sizeof(float4), cudaMemcpyHostToDevice);

		   rigid_volumes.push_back(0.f);
		   arrays::rigidVolumes::allocate(sizeof(float));
		   cudaMemcpy(arrays::rigidVolumes::ptr, rigid_volumes.data(), sizeof(float), cudaMemcpyHostToDevice);

		   densities.push_back(0.f);
		   arrays::rigidDensities::allocate(sizeof(float));
		   cudaMemcpy(arrays::rigidDensities::ptr, densities.data(), sizeof(float), cudaMemcpyHostToDevice);

		   rigid_velocities.push_back({ 0.0f, 0.0f, 0.0f, 0.0f });
		   arrays::rigidLinearVelocities::allocate(sizeof(float4));
		   cudaMemcpy(arrays::rigidLinearVelocities::ptr, rigid_velocities.data(), sizeof(float4), cudaMemcpyHostToDevice);

		   rigid_avelocities.push_back({ 0.0f, 0.0f, 0.0f });
		   arrays::rigidAVelocities::allocate(sizeof(float3));
		   cudaMemcpy(arrays::rigidAVelocities::ptr, rigid_avelocities.data(), sizeof(float3), cudaMemcpyHostToDevice);

	   }

}

void IO::vdb::recreateRigids() {
  int rigid_nums = 1;
  auto wrld = DynamicsWorld::getInstance();
  wrld->createWorld();
  for (auto fluidVolume : get<parameters::rigid_volumes::volume>()) {
    
    // DynamicsWorld::getInstance()->addBody(fluidVolume.fileName.value);
    int32_t index = rigid_nums - 1;
    auto dens = fluidVolume.density;
    auto vol = arrays::rigidVolumes::ptr[index];
    auto rfile = fluidVolume.fileName;
    wrld->addInfoBody(rfile, dens, {0, 0, 0});
    wrld->addBody(vol, index);
    rigid_nums++;
  }

  if (get<parameters::rigid_volumes::volume>().size()){
    
    DynamicsWorld::getInstance()->createBoundingBox();
  
  }
  else{}

  for (auto boundaryVolume : get<parameters::boundary_volumes::volume>()) {
    wrld->addBoundary(boundaryVolume.fileName);
  }
}