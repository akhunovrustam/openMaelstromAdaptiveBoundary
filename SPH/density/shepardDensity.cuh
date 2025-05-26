#pragma once
#include <utility/identifier.h>
/*
Module used to implement a compact hashing style resorting algorithm on GPUs with 64 bit indices, limiting the domain to 2,097,152^3 cells. Additionally implements a multi level scheme that creates multiplle hash tables for a much faster adaptive simulaiton.
*/
namespace SPH{
	namespace shepardDensity{
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

			// temporary resources (mapped as read/write)
			// input resources (mapped as read only)
			const_array_u<arrays::position> position;
			const_array_u<arrays::volume> volume;
			const_array_u<arrays::boundaryPlanes> boundaryPlanes;
			const_array_u<arrays::velocity> velocity;

			// output resources (mapped as read/write)
			// swap resources (mapped as read/write)
			swap_array_u<arrays::density> density;

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
			int32_t boundaryCounter;
			uFloat<SI::m> ptcl_spacing;
			int32_t boundaryLUTSize;
			float LUTOffset;

			const_array_u<arrays::boundaryPlaneVelocity> boundaryPlaneVelocity;
			const_array_u<arrays::offsetLUT> offsetLUT;
			const_array_u<arrays::splineLUT> splineLUT;
			const_array_u<arrays::spline2LUT> spline2LUT;
			const_array_u<arrays::splineGradientLUT> splineGradientLUT;
			const_array_u<arrays::spikyLUT> spikyLUT;
			const_array_u<arrays::spikyGradientLUT> spikyGradientLUT;
			const_array_u<arrays::cohesionLUT> cohesionLUT;
			const_array_u<arrays::volumeLUT> volumeLUT;
			const_array_u<arrays::adhesionLUT> adhesionLUT;

			// volume boundary resources (mapped as read only)
			int32_t volumeBoundaryCounter;

			const_array_u<arrays::volumeBoundaryVolumes> volumeBoundaryVolumes;
			const_array_u<arrays::volumeBoundaryDimensions> volumeBoundaryDimensions;
			const_array_u<arrays::volumeBoundaryMin> volumeBoundaryMin;
			const_array_u<arrays::volumeBoundaryMax> volumeBoundaryMax;
			const_array_u<arrays::volumeBoundaryDensity> volumeBoundaryDensity;
			const_array_u<arrays::volumeBoundaryVolume> volumeBoundaryVolume;
			const_array_u<arrays::volumeBoundaryVelocity> volumeBoundaryVelocity;
			const_array_u<arrays::volumeBoundaryAngularVelocity> volumeBoundaryAngularVelocity;
			const_array_u<arrays::volumeBoundaryPosition> volumeBoundaryPosition;
			const_array_u<arrays::volumeBoundaryQuaternion> volumeBoundaryQuaternion;
			const_array_u<arrays::volumeBoundaryTransformMatrix> volumeBoundaryTransformMatrix;
			const_array_u<arrays::volumeBoundaryTransformMatrixInverse> volumeBoundaryTransformMatrixInverse;
			const_array_u<arrays::volumeBoundaryKind> volumeBoundaryKind;
			const_array_u<arrays::volumeBoundaryInertiaMatrix> volumeBoundaryInertiaMatrix;
			const_array_u<arrays::volumeBoundaryInertiaMatrixInverse> volumeBoundaryInertiaMatrixInverse;
			const_array_u<arrays::volumeBoundaryGVDBVolumes> volumeBoundaryGVDBVolumes;
			const_array_u<arrays::gvdbOffsets> gvdbOffsets;
			const_array_u<arrays::gvdbVoxelSizes> gvdbVoxelSizes;

			
			using swap_arrays = std::tuple<arrays::density>;
			using input_arrays = std::tuple<arrays::position, arrays::volume, arrays::boundaryPlanes, arrays::velocity>;
			using output_arrays = std::tuple<>;
			using temporary_arrays = std::tuple<>;
			using cell_info_arrays = std::tuple<arrays::cellBegin, arrays::cellEnd, arrays::cellSpan, arrays::hashMap, arrays::compactHashMap, arrays::compactCellSpan, arrays::MLMResolution>;
			using virtual_info_arrays = std::tuple<arrays::boundaryPlaneVelocity, arrays::offsetLUT, arrays::splineLUT, arrays::spline2LUT, arrays::splineGradientLUT, arrays::spikyLUT, arrays::spikyGradientLUT, arrays::cohesionLUT, arrays::volumeLUT, arrays::adhesionLUT>;
			using boundaryInfo_arrays = std::tuple<arrays::volumeBoundaryVolumes, arrays::volumeBoundaryDimensions, arrays::volumeBoundaryMin, arrays::volumeBoundaryMax, arrays::volumeBoundaryDensity, arrays::volumeBoundaryVolume, arrays::volumeBoundaryVelocity, arrays::volumeBoundaryAngularVelocity, arrays::volumeBoundaryPosition, arrays::volumeBoundaryQuaternion, arrays::volumeBoundaryTransformMatrix, arrays::volumeBoundaryTransformMatrixInverse, arrays::volumeBoundaryKind, arrays::volumeBoundaryInertiaMatrix, arrays::volumeBoundaryInertiaMatrixInverse, arrays::volumeBoundaryGVDBVolumes, arrays::gvdbOffsets, arrays::gvdbVoxelSizes>;
			using neighbor_info_arrays = std::tuple<arrays::neighborList, arrays::neighborListLength, arrays::spanNeighborList, arrays::compactCellScale, arrays::compactCellList, arrays::neighborMask>;
			constexpr static const bool resort = false;
constexpr static const bool inlet = false;
		};
#ifndef __CUDA_ARCH__
		//valid checking function
		inline bool valid(Memory){
			bool condition = false;
			condition = condition || get<parameters::modules::density>() == "shepard";
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
			arrays.boundaryCounter = uGet<parameters::internal::boundaryCounter>();
			arrays.ptcl_spacing = uGet<parameters::internal::ptcl_spacing>();
			arrays.boundaryLUTSize = uGet<parameters::internal::boundaryLUTSize>();
			arrays.LUTOffset = uGet<parameters::simulation_settings::LUTOffset>();
			arrays.volumeBoundaryCounter = uGet<parameters::boundary_volumes::volumeBoundaryCounter>();
			arrays.boundaryDampening = uGet<parameters::simulation_settings::boundaryDampening>();

		}
#endif
		
		void estimate_density(Memory mem = Memory());
		void update_density(Memory mem = Memory());
	} // namspace shepardDensity
}// namespace SPH
