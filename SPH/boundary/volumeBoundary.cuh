#pragma once
#include <utility/identifier.h>
/*
Module used to implement vdb based solid objects. Maps vdb volumes (signed distance fields) to 3d cuda textures.
*/
namespace SPH{
	namespace volume{
		struct Memory{
			// basic information
			int32_t num_ptcls;
			uFloat<SI::s> timestep;
			uFloat<SI::m> radius;
			uFloat<SI::density> rest_density;
			int32_t maxNumptcls;
			uFloat<SI::m> ptcl_support;
			uFloat3<SI::m> min_domain;
			uFloat3<SI::m> max_domain;

			write_array_u<arrays::debugArray> debugArray;

			// parameters
			float boundaryDampening;
			int32_t volumeBoundaryCounter;

			// temporary resources (mapped as read/write)
			// input resources (mapped as read only)
			const_array_u<arrays::volumeBoundaryVolumes> volumeBoundaryVolumes;
			const_array_u<arrays::volumeBoundaryDimensions> volumeBoundaryDimensions;
			const_array_u<arrays::volumeBoundaryMin> volumeBoundaryMin;
			const_array_u<arrays::volumeBoundaryMax> volumeBoundaryMax;

			// output resources (mapped as read/write)
			// swap resources (mapped as read/write)
			swap_array_u<arrays::position> position;
			swap_array_u<arrays::velocity> velocity;

			// cell resources (mapped as read only)
			// neighborhood resources (mapped as read only)
			// virtual resources (mapped as read only)
			// volume boundary resources (mapped as read only)
			
			using swap_arrays = std::tuple<arrays::position, arrays::velocity>;
			using input_arrays = std::tuple<arrays::volumeBoundaryVolumes, arrays::volumeBoundaryDimensions, arrays::volumeBoundaryMin, arrays::volumeBoundaryMax>;
			using output_arrays = std::tuple<>;
			using temporary_arrays = std::tuple<>;
			using cell_info_params = std::tuple<>;
			using cell_info_arrays = std::tuple<>;
			using virtual_info_params = std::tuple<>;
			using virtual_info_arrays = std::tuple<>;
			using boundaryInfo_params = std::tuple<>;
			using boundaryInfo_arrays = std::tuple<>;
			using neighbor_info_params = std::tuple<>;
			using neighbor_info_arrays = std::tuple<>;
			constexpr static const bool resort = false;
constexpr static const bool inlet = false;
		};
#ifndef __CUDA_ARCH__
		//valid checking function
		inline bool valid(Memory){
			bool condition = false;
			condition = condition || get<parameters::modules::volumeBoundary>() == true;
			return condition;
		}
		inline void setupParameters(Memory& arrays){
			arrays.num_ptcls = uGet<parameters::internal::num_ptcls>();
			arrays.timestep = uGet<parameters::internal::timestep>();
			arrays.radius = uGet<parameters::particle_settings::radius>();
			arrays.rest_density = uGet<parameters::particle_settings::rest_density>();
			arrays.maxNumptcls = uGet<parameters::simulation_settings::maxNumptcls>();
			arrays.ptcl_support = uGet<parameters::internal::ptcl_support>();
			arrays.min_domain = uGet<parameters::internal::min_domain>();
			arrays.max_domain = uGet<parameters::internal::max_domain>();
			arrays.boundaryDampening = uGet<parameters::simulation_settings::boundaryDampening>();
			arrays.volumeBoundaryCounter = uGet<parameters::boundary_volumes::volumeBoundaryCounter>();

		}
#endif
		
		void init_volumes(Memory mem = Memory());
		void update(Memory mem = Memory());
	} // namspace volume
}// namespace SPH
