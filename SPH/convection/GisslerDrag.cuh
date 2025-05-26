#pragma once
#include <utility/identifier.h>
/*
Module used to provide air drag forces for particles to improve streams and overall behaviour of the simulaltion.
*/
namespace SPH{
	namespace GisslerDrag{
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
			uFloat4<SI::velocity> air_velocity;
			uFloat<SI::acceleration> tension_akinci;

			// temporary resources (mapped as read/write)
			// input resources (mapped as read only)
			const_array_u<arrays::velocity> velocity;
			const_array_u<arrays::position> position;
			const_array_u<arrays::density> density;
			const_array_u<arrays::volume> volume;

			// output resources (mapped as read/write)
			write_array_u<arrays::acceleration> acceleration;

			// swap resources (mapped as read/write)
			// cell resources (mapped as read only)
			int3 gridSize;
			uFloat3<SI::m> cell_size;
			uint32_t hash_entries;
			uFloat3<SI::m> min_coord;
			uint32_t mlm_schemes;

			const_array_u<arrays::cellBegin> cellBegin;
			const_array_u<arrays::cellEnd> cellEnd;
			const_array_u<arrays::cellSpan> cellSpan;
			const_array_u<arrays::hashMap> hashMap;
			const_array_u<arrays::compactHashMap> compactHashMap;
			const_array_u<arrays::compactCellSpan> compactCellSpan;
			const_array_u<arrays::MLMResolution> MLMResolution;

			// neighborhood resources (mapped as read only)
			const_array_u<arrays::neighborList> neighborList;
			const_array_u<arrays::neighborListLength> neighborListLength;
			const_array_u<arrays::spanNeighborList> spanNeighborList;
			const_array_u<arrays::compactCellScale> compactCellScale;
			const_array_u<arrays::compactCellList> compactCellList;
			const_array_u<arrays::neighborMask> neighborMask;

			// virtual resources (mapped as read only)
			// volume boundary resources (mapped as read only)
			
			using swap_arrays = std::tuple<>;
			using input_arrays = std::tuple<arrays::velocity, arrays::position, arrays::density, arrays::volume>;
			using output_arrays = std::tuple<arrays::acceleration>;
			using temporary_arrays = std::tuple<>;
			using cell_info_arrays = std::tuple<arrays::cellBegin, arrays::cellEnd, arrays::cellSpan, arrays::hashMap, arrays::compactHashMap, arrays::compactCellSpan, arrays::MLMResolution>;
			using virtual_info_params = std::tuple<>;
			using virtual_info_arrays = std::tuple<>;
			using boundaryInfo_params = std::tuple<>;
			using boundaryInfo_arrays = std::tuple<>;
			using neighbor_info_arrays = std::tuple<arrays::neighborList, arrays::neighborListLength, arrays::spanNeighborList, arrays::compactCellScale, arrays::compactCellList, arrays::neighborMask>;
			constexpr static const bool resort = false;
constexpr static const bool inlet = false;
		};
#ifndef __CUDA_ARCH__
		//valid checking function
		inline bool valid(Memory){
			bool condition = false;
			condition = condition || get<parameters::modules::drag>() == "Gissler17";
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
			arrays.hash_entries = uGet<parameters::simulation_settings::hash_entries>();
			arrays.min_coord = uGet<parameters::internal::min_coord>();
			arrays.mlm_schemes = uGet<parameters::simulation_settings::mlm_schemes>();
			arrays.air_velocity = uGet<parameters::particle_settings::air_velocity>();
			arrays.tension_akinci = uGet<parameters::particle_settings::tension_akinci>();

		}
#endif
		
		void drag(Memory mem = Memory());
	} // namspace GisslerDrag
}// namespace SPH
