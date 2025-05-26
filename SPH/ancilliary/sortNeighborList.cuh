#pragma once
#include <utility/identifier.h>
/*
Module used to create a Constrained NeighborList based on the original paper. Fairly slow for adaptive simulations.
*/
namespace SPH{
	namespace sortedNeighborList{
		struct Memory{
			// basic information
			int32_t num_ptcls;
			float timestep;
			float radius;
			float rest_density;
			int32_t maxNumptcls;
			float ptcl_support;
			float3 min_domain;
			float3 max_domain;

			write_array<arrays::debugArray> debugArray;

			// parameters
			// temporary resources (mapped as read/write)
			// input resources (mapped as read only)
			const_array<arrays::volume> volume;
			const_array<arrays::position> position;

			// output resources (mapped as read/write)
			write_array<arrays::neighborList> neighborList;
			write_array<arrays::neighborListLength> neighborListLength;

			// swap resources (mapped as read/write)
			// cell resources (mapped as read only)
			// neighborhood resources (mapped as read only)
			// virtual resources (mapped as read only)
			// volume boundary resources (mapped as read only)
			
			using swap_arrays = std::tuple<>;
			using input_arrays = std::tuple<arrays::volume, arrays::position>;
			using output_arrays = std::tuple<arrays::neighborList, arrays::neighborListLength>;
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
			condition = condition || get<parameters::modules::neighborSorting>() == true;
			return condition;
		}
		inline void setupParameters(Memory& arrays){
			arrays.num_ptcls = get<parameters::internal::num_ptcls>();
			arrays.timestep = get<parameters::internal::timestep>();
			arrays.radius = get<parameters::particle_settings::radius>();
			arrays.rest_density = get<parameters::particle_settings::rest_density>();
			arrays.maxNumptcls = get<parameters::simulation_settings::maxNumptcls>();
			arrays.ptcl_support = get<parameters::internal::ptcl_support>();
			arrays.min_domain = get<parameters::internal::min_domain>();
			arrays.max_domain = get<parameters::internal::max_domain>();

		}
#endif
		
		void sort(Memory mem = Memory());
	} // namspace sortedNeighborList
}// namespace SPH
