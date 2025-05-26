#pragma once
#include <utility/identifier.h>
/*
Module used to iteratively created a stable surface distance for every particle in the simulation. Very slow.
*/
namespace SPH{
	namespace distance{
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
			float xsph_viscosity;
			uFloat<SI::m> surface_levelLimit;
			int32_t surface_neighborLimit;
			uFloat<SI::m> surface_phiMin;
			float surface_phiChange;
			uFloat3<SI::m> surface_distanceFieldDistances;

			// temporary resources (mapped as read/write)
			write_array_u<arrays::decisionBuffer> decisionBuffer;
			write_array_u<arrays::markerBuffer> markerBuffer;
			write_array_u<arrays::changeBuffer> changeBuffer;

			// input resources (mapped as read only)
			const_array_u<arrays::position> position;
			const_array_u<arrays::density> density;
			const_array_u<arrays::volume> volume;
			const_array_u<arrays::velocity> velocity;
			const_array_u<arrays::particle_type> particle_type;

			// output resources (mapped as read/write)
			// swap resources (mapped as read/write)
			swap_array_u<arrays::distanceBuffer> distanceBuffer;
			swap_array_u<arrays::surface_idxBuffer> surface_idxBuffer;

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

			const_array_u<arrays::boundaryPlanes> boundaryPlanes;
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

			
			using swap_arrays = std::tuple<arrays::distanceBuffer, arrays::surface_idxBuffer>;
			using input_arrays = std::tuple<arrays::position, arrays::density, arrays::volume, arrays::velocity, arrays::particle_type>;
			using output_arrays = std::tuple<>;
			using temporary_arrays = std::tuple<arrays::decisionBuffer, arrays::markerBuffer, arrays::changeBuffer>;
			using cell_info_arrays = std::tuple<arrays::cellBegin, arrays::cellEnd, arrays::cellSpan, arrays::hashMap, arrays::compactHashMap, arrays::compactCellSpan, arrays::MLMResolution>;
			using virtual_info_arrays = std::tuple<arrays::boundaryPlanes, arrays::boundaryPlaneVelocity, arrays::offsetLUT, arrays::splineLUT, arrays::spline2LUT, arrays::splineGradientLUT, arrays::spikyLUT, arrays::spikyGradientLUT, arrays::cohesionLUT, arrays::volumeLUT, arrays::adhesionLUT>;
			using boundaryInfo_arrays = std::tuple<arrays::volumeBoundaryVolumes, arrays::volumeBoundaryDimensions, arrays::volumeBoundaryMin, arrays::volumeBoundaryMax, arrays::volumeBoundaryDensity, arrays::volumeBoundaryVolume, arrays::volumeBoundaryVelocity, arrays::volumeBoundaryAngularVelocity, arrays::volumeBoundaryPosition, arrays::volumeBoundaryQuaternion, arrays::volumeBoundaryTransformMatrix, arrays::volumeBoundaryTransformMatrixInverse, arrays::volumeBoundaryKind, arrays::volumeBoundaryInertiaMatrix, arrays::volumeBoundaryInertiaMatrixInverse, arrays::volumeBoundaryGVDBVolumes, arrays::gvdbOffsets, arrays::gvdbVoxelSizes>;
			using neighbor_info_arrays = std::tuple<arrays::neighborList, arrays::neighborListLength, arrays::spanNeighborList, arrays::compactCellScale, arrays::compactCellList, arrays::neighborMask>;
			constexpr static const bool resort = false;
constexpr static const bool inlet = false;
		};
#ifndef __CUDA_ARCH__
		//valid checking function
		inline bool valid(Memory){
			bool condition = false;
			condition = condition || get<parameters::modules::surfaceDistance>() == true;
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
			arrays.xsph_viscosity = uGet<parameters::particle_settings::xsph_viscosity>();
			arrays.surface_levelLimit = uGet<parameters::surfaceDistance::surface_levelLimit>();
			arrays.surface_neighborLimit = uGet<parameters::surfaceDistance::surface_neighborLimit>();
			arrays.surface_phiMin = uGet<parameters::surfaceDistance::surface_phiMin>();
			arrays.surface_phiChange = uGet<parameters::surfaceDistance::surface_phiChange>();
			arrays.surface_distanceFieldDistances = uGet<parameters::surfaceDistance::surface_distanceFieldDistances>();

		}
#endif
		
		void distance(Memory mem = Memory());
	} // namspace distance
}// namespace SPH
