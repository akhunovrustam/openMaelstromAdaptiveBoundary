#pragma once
#include <utility/identifier.h>
/*
Module used to implement a simple resorting algorithm that uses a cell entry for every actual cell in the domain. Does not support infinite domains.
*/
namespace SPH{
	namespace Resort{
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
			int3 gridSize;
			uFloat3<SI::m> cell_size;

			// temporary resources (mapped as read/write)
			// input resources (mapped as read only)
			const_array_u<arrays::position> position;

			// output resources (mapped as read/write)
			write_array_u<arrays::cellBegin> cellBegin;
			write_array_u<arrays::cellEnd> cellEnd;
			write_array_u<arrays::resortIndex> resortIndex;
			write_array_u<arrays::particleparticleIndex> particleparticleIndex;

			// swap resources (mapped as read/write)
			// cell resources (mapped as read only)
			// neighborhood resources (mapped as read only)
			// virtual resources (mapped as read only)
			// volume boundary resources (mapped as read only)
			
			using swap_arrays = std::tuple<>;
			using input_arrays = std::tuple<arrays::position>;
			using output_arrays = std::tuple<arrays::cellBegin, arrays::cellEnd, arrays::resortIndex, arrays::particleparticleIndex>;
			using temporary_arrays = std::tuple<>;
			using cell_info_params = std::tuple<>;
			using cell_info_arrays = std::tuple<>;
			using virtual_info_params = std::tuple<>;
			using virtual_info_arrays = std::tuple<>;
			using boundaryInfo_params = std::tuple<>;
			using boundaryInfo_arrays = std::tuple<>;
			using neighbor_info_params = std::tuple<>;
			using neighbor_info_arrays = std::tuple<>;
			constexpr static const bool resort = true;
constexpr static const bool inlet = false;
		};
#ifndef __CUDA_ARCH__
		//valid checking function
		inline bool valid(Memory){
			bool condition = false;
			condition = condition || get<parameters::modules::resorting>() == "linear_cell";
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
			arrays.gridSize = uGet<parameters::internal::gridSize>();
			arrays.cell_size = uGet<parameters::internal::cell_size>();

		}
#endif
		
		void resortParticles(Memory mem = Memory());
	} // namspace Resort
}// namespace SPH
