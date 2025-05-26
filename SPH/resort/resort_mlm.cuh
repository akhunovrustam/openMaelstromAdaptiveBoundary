#pragma once
#include <utility/identifier.h>
/*
Module used to implement a compact hashing style resorting algorithm on GPUs with 64 bit indices, limiting the domain to 2,097,152^3 cells. Additionally implements a multi level scheme that creates multiplle hash tables for a much faster adaptive simulaiton.
*/
namespace SPH{
	namespace resort_mlm{
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
			int resort_algorithm;

			// temporary resources (mapped as read/write)
			write_array<arrays::resortIndex> resortIndex;
			write_array<arrays::particleparticleIndex> particleparticleIndex;
			write_array<arrays::ZOrder_64> ZOrder_64;
			write_array<arrays::ZOrder_32> ZOrder_32;
			write_array<arrays::cellSpanSwap> cellSpanSwap;
			write_array<arrays::cellparticleIndex> cellparticleIndex;
			write_array<arrays::compactparticleIndex> compactparticleIndex;
			write_array<arrays::resortArray> resortArray;
			write_array<arrays::resortArray4> resortArray4;

			// input resources (mapped as read only)
			const_array<arrays::position> position;

			// output resources (mapped as read/write)
			write_array<arrays::cellSpan> cellSpan;
			write_array<arrays::hashMap> hashMap;
			write_array<arrays::MLMResolution> MLMResolution;

			// swap resources (mapped as read/write)
			// cell resources (mapped as read only)
			int3 gridSize;
			float3 cell_size;
			uint32_t hash_entries;
			float3 min_coord;
			uint32_t mlm_schemes;

			const_array<arrays::cellBegin> cellBegin;
			const_array<arrays::cellEnd> cellEnd;
			const_array<arrays::compactHashMap> compactHashMap;
			const_array<arrays::compactCellSpan> compactCellSpan;

			// neighborhood resources (mapped as read only)
			// virtual resources (mapped as read only)
			// volume boundary resources (mapped as read only)
			
			using swap_arrays = std::tuple<>;
			using input_arrays = std::tuple<arrays::position>;
			using output_arrays = std::tuple<arrays::cellSpan, arrays::hashMap, arrays::MLMResolution>;
			using temporary_arrays = std::tuple<arrays::resortIndex, arrays::particleparticleIndex, arrays::ZOrder_64, arrays::ZOrder_32, arrays::cellSpanSwap, arrays::cellparticleIndex, arrays::compactparticleIndex, arrays::resortArray, arrays::resortArray4>;
			using cell_info_arrays = std::tuple<arrays::cellBegin, arrays::cellEnd, arrays::compactHashMap, arrays::compactCellSpan>;
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
			condition = condition || get<parameters::modules::resorting>() == "MLM";
			condition = condition || get<parameters::modules::resorting>() == "hashed_cell";
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
			arrays.gridSize = get<parameters::internal::gridSize>();
			arrays.cell_size = get<parameters::internal::cell_size>();
			arrays.hash_entries = get<parameters::simulation_settings::hash_entries>();
			arrays.min_coord = get<parameters::internal::min_coord>();
			arrays.mlm_schemes = get<parameters::simulation_settings::mlm_schemes>();
			arrays.resort_algorithm = get<parameters::resort::resort_algorithm>();

		}
#endif
		
		void resortParticles(Memory mem = Memory());
	} // namspace resort_mlm
}// namespace SPH
