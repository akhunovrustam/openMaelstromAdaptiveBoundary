#pragma once
#include <utility/identifier.h>
/*
Module used to provide moving boundaries, e.g. wave walls, in simulations. Implicit plane based.
*/
namespace SPH{
	namespace moving_planes{
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
			int32_t boundaryCounter;
			float boundaryDampening;

			// temporary resources (mapped as read/write)
			// input resources (mapped as read only)
			// output resources (mapped as read/write)
			write_array_u<arrays::boundaryPlanes> boundaryPlanes;
			write_array_u<arrays::boundaryPlaneVelocity> boundaryPlaneVelocity;

			// swap resources (mapped as read/write)
			swap_array_u<arrays::position> position;
			swap_array_u<arrays::velocity> velocity;

			// cell resources (mapped as read only)
			// neighborhood resources (mapped as read only)
			// virtual resources (mapped as read only)
			// volume boundary resources (mapped as read only)
			
			using swap_arrays = std::tuple<arrays::position, arrays::velocity>;
			using input_arrays = std::tuple<>;
			using output_arrays = std::tuple<arrays::boundaryPlanes, arrays::boundaryPlaneVelocity>;
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
			condition = condition || get<parameters::modules::movingBoundaries>() == true;
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
			arrays.boundaryCounter = uGet<parameters::internal::boundaryCounter>();
			arrays.boundaryDampening = uGet<parameters::simulation_settings::boundaryDampening>();

		}
#endif
		
		void correct_position(Memory mem = Memory());
		void correct_velocity(Memory mem = Memory());
		void update_boundaries(Memory mem = Memory());
	} // namspace moving_planes
}// namespace SPH
