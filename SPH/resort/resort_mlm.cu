#include <SPH/resort/resort_mlm.cuh>
#include <utility/include_all.h>
#include <dummy.h>

namespace SPH {
	namespace resort_mlm {
		basicFunctionType cudaHashTablea(SPH::resort_mlm::Memory arrays, int32_t threads) {
			checkedThreadIdx(i);
			auto h = arrays.resortIndex[i];
			if (i == 0 || h != arrays.resortIndex[i - 1])
				arrays.hashMap[h].beginning = i;
		}

		basicFunctionType cudaHashTableb(SPH::resort_mlm::Memory arrays, int32_t threads) {
			checkedThreadIdx(i);
			auto h = arrays.resortIndex[i];
			if (i == threads - 1 || arrays.resortIndex[i + 1] != arrays.resortIndex[i])
				arrays.hashMap[h].length = i - arrays.hashMap[h].beginning + 1;
		}

		basicFunctionType cudaCellTablea(SPH::resort_mlm::Memory arrays, int32_t threads, int32_t *compact, float) {
			checkedThreadIdx(i);
			//auto x_i = arrays.position[compact[i]];
                        arrays.cellSpan[i] = cell_span{(uint32_t)compact[i], (uint32_t)(compact[i + 1] - compact[i])};
		}
		basicFunctionType cudaCellTableb(SPH::resort_mlm::Memory arrays, int32_t threads, int32_t *compact, float ratio) {
			checkedThreadIdx(i);
			auto x_i = arrays.position[compact[i]];
			//if (compact[i + 1] == arrays.num_ptcls)
			//	arrays.cellSpan[i].length++;
			arrays.resortIndex[i] = position_to_hash(x_i, arrays.min_coord, (arrays.cell_size.x) * ratio, arrays.hash_entries);
			arrays.particleparticleIndex[i] = i;
		}

		basicFunctionType cudaMLMResolution(SPH::resort_mlm::Memory arrays, float* volume) {
			checkedParticleIdx(idx);
			//float4 p_i = arrays.position[idx];

			float target_neighbors = Kernel<kernel_kind::spline4>::neighbor_number * 0.95f;
			float kernel_epsilon =
				(1.f) * powf((target_neighbors) / ((4.f / 3.f) * CUDART_PI_F), 1.f / 3.f) / Kernel<kernel_kind::spline4>::kernel_size();
			auto particle_volume = volume[idx];
			auto actual_support = kernel_epsilon * powf(particle_volume, 1.f / 3.f);

			float h_i = actual_support * Kernel<kernel_kind::spline4>::kernel_size();
			int32_t r_i = (int32_t) math::clamp(math::floorf(math::abs(log2f(arrays.cell_size.x / h_i))) - 0, 0, arrays.mlm_schemes - 1);
			float f_i = powf(0.5f, ((float)r_i));
			arrays.MLMResolution[idx] = r_i;
		}

		cellFunctionType cudaHashParticles(SPH::resort_mlm::Memory arrays, float ratio, float ratio2) {
			checkedParticleIdx(i);
			auto x_i = arrays.position[i];
			auto h_i = (x_i.w);
			if (h_i != FLT_MAX) {
				//if (hash_width == hash_length::bit_64)
					arrays.ZOrder_64[i] = position_to_morton(x_i, arrays, ratio);
				//else
					arrays.ZOrder_32[i] = static_cast<int32_t>(position_to_morton_32(x_i, arrays, ratio));
				arrays.resortIndex[i] = position_to_hash(x_i, arrays, ratio2);
				arrays.particleparticleIndex[i] = i;
			}
			else {
				//if (hash_width == hash_length::bit_64)
				
					arrays.ZOrder_64[i] = INT64_MAX;//idx3D_to_morton(uint3{(uint)arrays.gridSize.x + 1, (uint)arrays.gridSize.y, (uint)arrays.gridSize.z});
				//else
					arrays.ZOrder_32[i] = INT_MAX;
				arrays.resortIndex[i] = INT_MAX;
				arrays.particleparticleIndex[i] = i;
			}
		}

		cellFunctionType cudaIndexCells(SPH::resort_mlm::Memory arrays, int32_t threads, int32_t *cell_indices) {
			checkedThreadIdx(i);
			if (i == 0)
				cell_indices[0] = 0;
			i++;
			if (hash_width == hash_length::bit_64)
				cell_indices[i] = i == arrays.num_ptcls || arrays.ZOrder_64[i - 1] != arrays.ZOrder_64[i] ? i : -1;
			else
				cell_indices[i] = i == arrays.num_ptcls || arrays.ZOrder_32[i - 1] != arrays.ZOrder_32[i] ? i : -1;
		}

		template <typename T> hostDeviceInline void cudaSortCompactmlm(SPH::resort_mlm::Memory arrays, int32_t threads,
			T* input, T* output) {
			checkedThreadIdx(i);
			output[i] = input[arrays.particleparticleIndex[i]];
		}



		cellFunction(hashParticles, cudaHashParticles, "hashing particles", caches<float4>{});
		basicFunction(calculateResolution, cudaMLMResolution, "generating cell table");
		basicFunction(buildCellTable1, cudaCellTablea, "creating cell table I");
		basicFunction(buildCellTable2, cudaCellTableb, "creating cell table II");
		basicFunction(buildHashTable1, cudaHashTablea, "hashing cell table I");
		basicFunction(buildHashTable2, cudaHashTableb, "hashing cell table II");
		cellFunction(indexCells, cudaIndexCells, "indexing cells");
		basicFunction(sort, cudaSortCompactmlm, "compact resorting cells");


			template <typename... Ts> auto callSort2(Memory mem, int32_t threads, Ts... tup) {
			launch<sort, Ts...>(threads, mem, threads, tup...);
		}
		template <typename... Ts> auto callSort3(Memory mem, std::tuple<Ts...> tup) {
			callSort2(mem, mem.num_ptcls, std::make_pair(Ts::ptr, Ts::rear_ptr)...);
		}

		struct is_valid {
			hostDeviceInline bool operator()(const int x) { return x != -1; }
		};
		struct count_if {
                  hostDeviceInline bool operator()(const hash_span x) {
                    return x.beginning != INVALID_BEG && x.length > 1;
                  }
		};
		struct invalid_position {
			hostDeviceInline bool operator()(float4 x) { return x.w == FLT_MAX; }
		};
		struct hash_spans {
			hostDeviceInline hash_span operator()() { return hash_span{ INVALID_BEG,0 }; }
		};

	} // namespace resort_mlm

} // namespace SPH


void SPH::resort_mlm::resortParticles(Memory mem) {
	//auto thrust_ptr = [](auto ptr) { return thrust::device_pointer_cast(ptr); };
	
	if (mem.num_ptcls > 0) {
		 
		auto min_coord = math::to<float3>(algorithm::reduce_min(arrays::position::ptr, mem.num_ptcls));
		min_coord -= 2.f * mem.cell_size;
		//get<parameters::internal::min_coord>() = math::min(min_coord, get<parameters::internal::min_domain>());
		get<parameters::internal::min_coord>() = min_coord;
		
		//auto max_coord = math::max(math::to<float3>(algorithm::reduce_max(arrays::position::ptr, mem.num_ptcls)), get<parameters::internal::max_domain>());
		auto max_coord = math::to<float3>(algorithm::reduce_max(arrays::position::ptr, mem.num_ptcls));
		max_coord += 2.f * mem.cell_size;
		get<parameters::internal::max_coord>() = max_coord;
		

		float max_length = math::max_elem(max_coord - get<parameters::internal::min_coord>());
		get<parameters::internal::gridSize>() = math::to<int3>((max_coord - min_coord) / get<parameters::internal::cell_size>().x);
		int32_t cells = static_cast<int32_t>(max_length / get<parameters::internal::cell_size>().x);
		int32_t v = cells;
		v--;
		v |= v >> 1;
		v |= v >> 2;
		v |= v >> 4;
		v |= v >> 8;
		v |= v >> 16;
		v++; 

		mem.min_coord = get<parameters::internal::min_coord>();
		//mem.max_coord = get<parameters::internal::max_coord>();
		mem.gridSize = get<parameters::internal::gridSize>();


		if (get<parameters::internal::hash_size>() == hash_length::bit_32) {
			float factor_morton = 1.f / ((float)(1024 / v));
			get<parameters::resort::zOrderScale>() = factor_morton;
			launch<hashParticles>(mem.num_ptcls, mem, factor_morton, 1.f);
			algorithm::sort_by_key(mem.num_ptcls, mem.ZOrder_32, mem.particleparticleIndex);
		}
		else {
			float factor_morton = 1.f / ((float)(1048576 / v));
			get<parameters::resort::zOrderScale>() = factor_morton;
			launch<hashParticles>(mem.num_ptcls, mem, factor_morton, 1.f);			
			algorithm::sort_by_key(mem.num_ptcls, mem.ZOrder_64, mem.particleparticleIndex);
		}


		void* original = arrays::resortArray::ptr;
		void* original4 = arrays::resortArray4::ptr;
		for_each(sorting_list, [&mem](auto x) {
			using P = std::decay_t<decltype(x)>;
			if (!P::valid()) return;
			using T = typename P::type;
			if (sizeof(T) == sizeof(float)) {
				launch<sort>(mem.num_ptcls, mem, mem.num_ptcls, (float*)P::ptr, arrays::resortArray::ptr);
				void* tmp = P::ptr;
				P::ptr = (T*)arrays::resortArray::ptr;
				arrays::resortArray::ptr = (float*)tmp;
			}
			else if (sizeof(T) == sizeof(float4)) {
				launch<sort>(mem.num_ptcls, mem, mem.num_ptcls, (float4*)P::ptr, arrays::resortArray4::ptr);
				void* tmp = P::ptr;
				P::ptr = (T*)arrays::resortArray4::ptr;
				arrays::resortArray4::ptr = (float4*)tmp;
			}
			else
				LOG_ERROR << "Cannot sort array of data size " << sizeof(T) << std::endl;
		});
		for_each(sorting_list, [&mem, original, original4](auto x) {
			using P = std::decay_t<decltype(x)>;
			if (!P::valid()) return;
			using T = typename P::type;
			if (sizeof(T) == sizeof(float)) {
				if (P::ptr == original) {
					cuda::memcpy(arrays::resortArray::ptr, P::ptr, sizeof(T) * mem.num_ptcls, cudaMemcpyDeviceToDevice);
					P::ptr = (T*)arrays::resortArray::ptr;
					arrays::resortArray::ptr = (float*)original;
				}
			}
			else if (sizeof(T) == sizeof(float4)) {
				if (P::ptr == original4) {
					cuda::memcpy(arrays::resortArray4::ptr, P::ptr, sizeof(T) * mem.num_ptcls, cudaMemcpyDeviceToDevice);
					P::ptr = (T*)arrays::resortArray4::ptr;
					arrays::resortArray4::ptr = (float4*)original4;
				}
			}
		});

		mem.position =  arrays::position::ptr;

		cuda::sync();
		auto iter = algorithm::count_if(arrays::position::ptr, mem.num_ptcls, invalid_position());
		cuda::sync();
		cuda::sync();
		if (iter != 0) {
			auto diff = get<parameters::internal::num_ptcls>() - iter;
			if (!logger::silent) {
				std::cout << "Removing " << get<parameters::internal::num_ptcls>() - diff << "particles" << std::endl;
				std::cout << "Old particle count: " << get<parameters::internal::num_ptcls>() << std::endl;
				std::cout << "New particle count: " << diff << std::endl;
			}
			get<parameters::internal::num_ptcls>() = static_cast<int32_t>(diff);
			mem.num_ptcls = static_cast<int32_t>(diff);
		}
		cuda::sync();

		launch<calculateResolution>(mem.num_ptcls, mem, arrays::volume::ptr);
		cuda::sync();
		// if(iter != 0){
		// 	auto diff = get<parameters::internal::num_ptcls>() - iter;

		// for(int32_t i = 0; i < mem.num_ptcls; ++i)
		// 	std::cout << ( i < diff ? " o - ": " x - ") << mem.position[i].x << ", "<< mem.position[i].y << ", "<< mem.position[i].z << ", "<< mem.position[i].w << std::endl;
		// }
		cuda::sync();

		get<parameters::resort::valid_cells>() = 0;
		get<parameters::resort::collision_cells>() = 0;
		get<parameters::resort::occupiedCells>().clear();
		for (int32_t i = 0; i < (int32_t)get<parameters::simulation_settings::mlm_schemes>(); ++i) {
			float factor = 1.f;
			for (int ii = 0; ii < i; ++ii)
				factor *= 0.5f;

			hash_span *hashMap = arrays::hashMap::ptr + get<parameters::simulation_settings::hash_entries>() *i;
			cell_span *cellSpan = arrays::cellSpan::ptr + get<parameters::simulation_settings::maxNumptcls>() *i;
			mem.cellSpan = cellSpan;
			mem.hashMap = hashMap;
			cuda::arrayMemset<arrays::cellparticleIndex>(0xFFFFFFFF);
			cuda::arrayMemset<arrays::ZOrder_64>(0xFFFFFFFF);

			launch<hashParticles>(mem.num_ptcls, mem, factor, factor);
			algorithm::generate(hashMap, mem.hash_entries, hash_spans());
			launch<indexCells>(mem.num_ptcls, mem, mem.num_ptcls, arrays::cellparticleIndex::ptr);
			cuda::sync();
			int32_t diff = static_cast<int32_t>(algorithm::copy_if(arrays::cellparticleIndex::ptr, arrays::compactparticleIndex::ptr, mem.num_ptcls + 1, is_valid()));
			cuda::sync();
			//std::cout << diff << std::endl;
			launch<buildCellTable1>(diff, mem, diff, arrays::compactparticleIndex::ptr, factor);
			cuda::sync();
			launch<buildCellTable2>(diff, mem, diff, arrays::compactparticleIndex::ptr, factor);
			cuda::sync();
			diff--;
			get<parameters::resort::occupiedCells>().push_back(diff);
			algorithm::sort_by_key(diff, mem.resortIndex, mem.particleparticleIndex);
			cuda::sync();
			launch<sort>(diff, mem, diff, cellSpan, mem.cellSpanSwap);
			cuda::sync();
			cuda::memcpy(cellSpan, mem.cellSpanSwap, sizeof(cell_span) * diff, cudaMemcpyDeviceToDevice);
			cuda::sync();
			launch<buildHashTable1>(diff, mem, diff);
			cuda::sync();
			launch<buildHashTable2>(diff, mem, diff);
			cuda::sync();
			//int32_t collisionsg = thrust::count_if(thrust::device, thrust_ptr(mem.hashMap), thrust_ptr(mem.hashMap) + get<parameters::simulation_settings::hash_entries>(), count_if());

			get<parameters::resort::valid_cells>() += diff;
		}
		//cudaDeviceSynchronize();
		//for (int32_t i = 0; i < get<parameters::resort::occupiedCells>()[0]; ++i) {
		//	std::cout << arrays::cellSpan::ptr[i].beginning << std::endl;
		//}
		//cudaDeviceSynchronize();
	}
}
