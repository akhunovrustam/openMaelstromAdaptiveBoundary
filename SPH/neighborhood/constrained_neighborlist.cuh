#pragma once
#include <utility/identifier.h>
/*
Module used to create a Constrained NeighborList based on the original paper. Fairly slow for adaptive simulations.
*/
namespace SPH{
	namespace ConstrainedNeighborList{
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
			float omega;
			int32_t overhead_size;
			int32_t target_neighbors;
			int32_t support_leeway;
			int32_t error_factor;
			int resort_algorithm;
			int32_t neighborlimit;

			// temporary resources (mapped as read/write)
			write_array_u<arrays::closestNeighbor> closestNeighbor;
			write_array_u<arrays::supportEstimate> supportEstimate;
			write_array_u<arrays::support> support;
			write_array_u<arrays::supportMarker> supportMarker;
			write_array_u<arrays::supportMarkerCompacted> supportMarkerCompacted;
			write_array_u<arrays::neighborListSwap> neighborListSwap;
			write_array_u<arrays::neighborOverhead> neighborOverhead;
			write_array_u<arrays::neighborOverheadCount> neighborOverheadCount;

			// input resources (mapped as read only)
			const_array_u<arrays::volume> volume;

			// output resources (mapped as read/write)
			write_array_u<arrays::position> position;
			write_array_u<arrays::neighborList> neighborList;
			write_array_u<arrays::neighborListLength> neighborListLength;

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
			// virtual resources (mapped as read only)
			// volume boundary resources (mapped as read only)
			
			using swap_arrays = std::tuple<>;
			using input_arrays = std::tuple<arrays::volume>;
			using output_arrays = std::tuple<arrays::position, arrays::neighborList, arrays::neighborListLength>;
			using temporary_arrays = std::tuple<arrays::closestNeighbor, arrays::supportEstimate, arrays::support, arrays::supportMarker, arrays::supportMarkerCompacted, arrays::neighborListSwap, arrays::neighborOverhead, arrays::neighborOverheadCount>;
			using cell_info_arrays = std::tuple<arrays::cellBegin, arrays::cellEnd, arrays::cellSpan, arrays::hashMap, arrays::compactHashMap, arrays::compactCellSpan, arrays::MLMResolution>;
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
			condition = condition || get<parameters::modules::neighborhood>() == "constrained";
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
			arrays.omega = uGet<parameters::support::omega>();
			arrays.overhead_size = uGet<parameters::support::overhead_size>();
			arrays.target_neighbors = uGet<parameters::support::target_neighbors>();
			arrays.support_leeway = uGet<parameters::support::support_leeway>();
			arrays.error_factor = uGet<parameters::support::error_factor>();
			arrays.resort_algorithm = uGet<parameters::resort::resort_algorithm>();
			arrays.neighborlimit = uGet<parameters::simulation_settings::neighborlimit>();

		}
#endif
		
		void calculate_neighborlist(Memory mem = Memory());
	} // namspace ConstrainedNeighborList
}// namespace SPH
