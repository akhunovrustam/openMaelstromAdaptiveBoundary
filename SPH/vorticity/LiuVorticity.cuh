#pragma once
#include <utility/identifier.h>
/*
Module used to provide a micropolar model for SPH to improve the vorticity of the simulation.
*/
namespace SPH{
	namespace LiuVorticity{
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
			float vorticityCoeff;

			// temporary resources (mapped as read/write)
			// input resources (mapped as read only)
			const_array<arrays::velocity> velocity;
			const_array<arrays::position> position;
			const_array<arrays::density> density;
			const_array<arrays::volume> volume;

			// output resources (mapped as read/write)
			write_array<arrays::acceleration> acceleration;

			// swap resources (mapped as read/write)
			swap_array<arrays::vorticity> vorticity;

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
			int32_t boundaryCounter;
			float ptcl_spacing;
			int32_t boundaryLUTSize;
			float LUTOffset;

			const_array<arrays::boundaryPlanes> boundaryPlanes;
			const_array<arrays::boundaryPlaneVelocity> boundaryPlaneVelocity;
			const_array<arrays::offsetLUT> offsetLUT;
			const_array<arrays::splineLUT> splineLUT;
			const_array<arrays::spline2LUT> spline2LUT;
			const_array<arrays::splineGradientLUT> splineGradientLUT;
			const_array<arrays::spikyLUT> spikyLUT;
			const_array<arrays::spikyGradientLUT> spikyGradientLUT;
			const_array<arrays::cohesionLUT> cohesionLUT;
			const_array<arrays::volumeLUT> volumeLUT;
			const_array<arrays::adhesionLUT> adhesionLUT;

			// volume boundary resources (mapped as read only)
			int32_t volumeBoundaryCounter;

			const_array<arrays::volumeBoundaryVolumes> volumeBoundaryVolumes;
			const_array<arrays::volumeBoundaryDimensions> volumeBoundaryDimensions;
			const_array<arrays::volumeBoundaryMin> volumeBoundaryMin;
			const_array<arrays::volumeBoundaryMax> volumeBoundaryMax;
			const_array<arrays::volumeBoundaryDensity> volumeBoundaryDensity;
			const_array<arrays::volumeBoundaryVolume> volumeBoundaryVolume;
			const_array<arrays::volumeBoundaryVelocity> volumeBoundaryVelocity;
			const_array<arrays::volumeBoundaryAngularVelocity> volumeBoundaryAngularVelocity;
			const_array<arrays::volumeBoundaryPosition> volumeBoundaryPosition;
			const_array<arrays::volumeBoundaryQuaternion> volumeBoundaryQuaternion;
			const_array<arrays::volumeBoundaryTransformMatrix> volumeBoundaryTransformMatrix;
			const_array<arrays::volumeBoundaryTransformMatrixInverse> volumeBoundaryTransformMatrixInverse;
			const_array<arrays::volumeBoundaryKind> volumeBoundaryKind;
			const_array<arrays::volumeBoundaryInertiaMatrix> volumeBoundaryInertiaMatrix;
			const_array<arrays::volumeBoundaryInertiaMatrixInverse> volumeBoundaryInertiaMatrixInverse;
			const_array<arrays::volumeBoundaryGVDBVolumes> volumeBoundaryGVDBVolumes;
			const_array<arrays::gvdbOffsets> gvdbOffsets;
			const_array<arrays::gvdbVoxelSizes> gvdbVoxelSizes;

			
			using swap_arrays = std::tuple<arrays::vorticity>;
			using input_arrays = std::tuple<arrays::velocity, arrays::position, arrays::density, arrays::volume>;
			using output_arrays = std::tuple<arrays::acceleration>;
			using temporary_arrays = std::tuple<>;
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
			condition = condition || get<parameters::modules::vorticity>() == "Liu";
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
			arrays.boundaryCounter = get<parameters::internal::boundaryCounter>();
			arrays.ptcl_spacing = get<parameters::internal::ptcl_spacing>();
			arrays.boundaryLUTSize = get<parameters::internal::boundaryLUTSize>();
			arrays.LUTOffset = get<parameters::simulation_settings::LUTOffset>();
			arrays.volumeBoundaryCounter = get<parameters::boundary_volumes::volumeBoundaryCounter>();
			arrays.vorticityCoeff = get<parameters::vorticitySettings::vorticityCoeff>();

		}
#endif
		
		void vorticity(Memory mem = Memory());
	} // namspace LiuVorticity
}// namespace SPH
