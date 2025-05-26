#pragma once
#include <utility/identifier.h>
/*
Module used to implement a simple resorting algorithm that uses a cell entry for every actual cell in the domain. Does not support infinite domains.
*/
namespace SPH{
	namespace renderMLM{
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
			float auxScale;
			float internalLimit;
			int32_t vrtxNeighborLimit;
			int32_t pruneVoxel;

			// temporary resources (mapped as read/write)
			write_array<arrays::resortIndex> resortIndex;
			write_array<arrays::auxDistance> auxDistance;
			write_array<arrays::particleparticleIndex> particleparticleIndex;
			write_array<arrays::ZOrder_64> ZOrder_64;
			write_array<arrays::ZOrder_32> ZOrder_32;
			write_array<arrays::compactCellSpanSwap> compactCellSpanSwap;
			write_array<arrays::cellparticleIndex> cellparticleIndex;
			write_array<arrays::compactparticleIndex> compactparticleIndex;
			write_array<arrays::resortArray> resortArray;
			write_array<arrays::resortArray4> resortArray4;

			// input resources (mapped as read only)
			const_array<arrays::position> position;
			const_array<arrays::volume> volume;
			const_array<arrays::density> density;

			// output resources (mapped as read/write)
			write_array<arrays::auxLength> auxLength;
			write_array<arrays::anisotropicSupport> anisotropicSupport;
			write_array<arrays::centerPosition> centerPosition;
			write_array<arrays::compactCellSpan> compactCellSpan;
			write_array<arrays::compactHashMap> compactHashMap;
			write_array<arrays::auxHashMap> auxHashMap;
			write_array<arrays::auxCellSpan> auxCellSpan;
			write_array<arrays::auxIsoDensity> auxIsoDensity;
			write_array<arrays::auxTest> auxTest;

			// swap resources (mapped as read/write)
			// cell resources (mapped as read only)
			int3 gridSize;
			float3 cell_size;
			uint32_t hash_entries;
			float3 min_coord;
			uint32_t mlm_schemes;

			const_array<arrays::cellBegin> cellBegin;
			const_array<arrays::cellEnd> cellEnd;
			const_array<arrays::cellSpan> cellSpan;
			const_array<arrays::hashMap> hashMap;
			const_array<arrays::MLMResolution> MLMResolution;

			// neighborhood resources (mapped as read only)
			const_array<arrays::neighborList> neighborList;
			const_array<arrays::neighborListLength> neighborListLength;
			const_array<arrays::spanNeighborList> spanNeighborList;
			const_array<arrays::compactCellScale> compactCellScale;
			const_array<arrays::compactCellList> compactCellList;
			const_array<arrays::neighborMask> neighborMask;

			// virtual resources (mapped as read only)
			// volume boundary resources (mapped as read only)
			
			using swap_arrays = std::tuple<>;
			using input_arrays = std::tuple<arrays::position, arrays::volume, arrays::density>;
			using output_arrays = std::tuple<arrays::auxLength, arrays::anisotropicSupport, arrays::centerPosition, arrays::compactCellSpan, arrays::compactHashMap, arrays::auxHashMap, arrays::auxCellSpan, arrays::auxIsoDensity, arrays::auxTest>;
			using temporary_arrays = std::tuple<arrays::resortIndex, arrays::auxDistance, arrays::particleparticleIndex, arrays::ZOrder_64, arrays::ZOrder_32, arrays::compactCellSpanSwap, arrays::cellparticleIndex, arrays::compactparticleIndex, arrays::resortArray, arrays::resortArray4>;
			using cell_info_arrays = std::tuple<arrays::cellBegin, arrays::cellEnd, arrays::cellSpan, arrays::hashMap, arrays::MLMResolution>;
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
			bool condition = true;
			condition = condition && get<parameters::modules::rayTracing>() == true;
			condition = condition && get<parameters::modules::resorting>() == "compactMLM";
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
			arrays.auxScale = get<parameters::render_settings::auxScale>();
			arrays.internalLimit = get<parameters::render_settings::internalLimit>();
			arrays.vrtxNeighborLimit = get<parameters::render_settings::vrtxNeighborLimit>();
			arrays.pruneVoxel = get<parameters::color_map::pruneVoxel>();

		}
#endif
		
		void generateAuxilliaryGrid(Memory mem = Memory());
	} // namspace renderMLM
}// namespace SPH
