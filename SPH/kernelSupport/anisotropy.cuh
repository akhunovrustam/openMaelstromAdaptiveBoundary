#pragma once
#include <utility/identifier.h>
/*
Module used to implement a simple resorting algorithm that uses a cell entry for every actual cell in the domain. Does not support infinite domains.
*/
namespace SPH{
	namespace anisotropy{
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
			float anisotropicLambda;
			int32_t anisotropicNepsilon;
			float anisotropicKs;
			float anisotropicKr;
			float anisotropicKn;

			// temporary resources (mapped as read/write)
			// input resources (mapped as read only)
			const_array<arrays::position> position;
			const_array<arrays::volume> volume;
			const_array<arrays::density> density;

			// output resources (mapped as read/write)
			write_array<arrays::anisotropicSupport> anisotropicSupport;
			write_array<arrays::centerPosition> centerPosition;
			write_array<arrays::anisotropicMatrices> anisotropicMatrices;
			write_array<arrays::auxDistance> auxDistance;
			write_array<arrays::auxTest> auxTest;
			write_array<arrays::auxIsoDensity> auxIsoDensity;

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
			const_array<arrays::compactHashMap> compactHashMap;
			const_array<arrays::compactCellSpan> compactCellSpan;
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
			using output_arrays = std::tuple<arrays::anisotropicSupport, arrays::centerPosition, arrays::anisotropicMatrices, arrays::auxDistance, arrays::auxTest, arrays::auxIsoDensity>;
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
			bool condition = true;
			condition = condition && get<parameters::modules::rayTracing>() == true;
			condition = condition && get<parameters::modules::anisotropicSurface>() == true;
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
			arrays.anisotropicLambda = get<parameters::render_settings::anisotropicLambda>();
			arrays.anisotropicNepsilon = get<parameters::render_settings::anisotropicNepsilon>();
			arrays.anisotropicKs = get<parameters::render_settings::anisotropicKs>();
			arrays.anisotropicKr = get<parameters::render_settings::anisotropicKr>();
			arrays.anisotropicKn = get<parameters::render_settings::anisotropicKn>();

		}
#endif
		
		void generateAnisotropicMatrices(Memory mem = Memory());
	} // namspace anisotropy
}// namespace SPH
