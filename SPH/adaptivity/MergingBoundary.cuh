#pragma once
#include <SPH/adaptivity/ContinuousAdaptivity.cuh>
#include <utility/include_all.h>

// Function that markes all particles based on their classification as a particle that should be
// merged (-1) or any other particle (-3)
neighFunctionType mergeDetectBoundary(SPH::adaptive::Memory arrays) {
	checkedParticleIdx(i);
	if (arrays.particle_type[i] == 0) return;
	if (arrays.optimization_lifetime_merge[i] < 0) 
		arrays.optimization_lifetime_merge[i] = 0;
	arrays.mergeCounter[i] = 0;
	arrays.mergeable[i] = -4;
	
	if (arrays.adaptiveClassification[i] <= -1.5f && arrays.lifetime[i] >= 0.f)
	{
		//printf("uid %d mergeIndic %d class %f ltime %f\n", arrays.uid[i], arrays.mergeIndicator[i], arrays.adaptiveClassification[i], arrays.lifetime[i].val);
		arrays.mergeIndicator[i]++;
		if (arrays.mergeIndicator[i] < 100)
		{
			arrays.mergeable[i] = -4;
			return;
		}
		bool eligible = true;
		/*iterateNeighbors(j) {
			if (arrays.particle_type[j] == 0)
				eligible = false;
			for (int32_t jj1 = 0, j1 = arrays.neighborList[j]; jj1 < arrays.neighborListLength[j]; (++jj1), (j1 = arrays.neighborList[j + jj1 * arrays.maxNumptcls])) {
				if (arrays.particle_type[j1] == 0)
					eligible = false;
			}
		}*/
		if (eligible) {
			arrays.mergeable[i] = -1;
			//arrays.optimization_lifetime_merge[i] = 10;
			//printf("!!IS mergable uid %d\n", arrays.uid[i]);
		}
		return;
	}
	if (arrays.lifetime[i].val < 15.f * arrays.timestep.val)
		arrays.mergeable[i] = -4;
	arrays.mergeIndicator[i] = 0;
	
}

// Finds merging partners with even indices for particles with odd indices. Found particles will be
// marked with the index of the particle they will be merged with and merging candidates that have
// found a partner will be marked -2. This function will only find a partner if the partner is of
// the correct classification and the merge result would not violate the systemic limit to particle
// sizes.
neighFunctionType mergeGrabEvenBoundary(SPH::adaptive::Memory arrays, int32_t* idxArray) {
	checkedParticleIdx(i);
	i = idxArray[i];
	if (arrays.particle_type[i] == 0) return;
	if (i % 2 == 0 || arrays.mergeable[i] != -1 || arrays.adaptiveClassification[i] > -1.5f)
		return;
				
	float maxvol = PI4O3 * (arrays.radius * arrays.radius * arrays.radius).val;

	/*if (arrays.volume.first[i].val / maxvol <= 0.5) {
		printf("111111111111111111////////////////maxvol %f vol %f rat %f uid %d\n", maxvol, arrays.volume.first[i].val, arrays.volume.first[i].val / maxvol, arrays.uid[i]);
		iterateNeighbors(j) {
			if (i != j && arrays.particle_type[j] != 0 && arrays.volume.first[j].val / maxvol <= 0.5)
				printf("////////////////maxvol %f neighvol %f rat %f uid %d muid %d merge %d mergindic %d\n", 
					maxvol, arrays.volume.first[j].val, arrays.volume.first[j].val / maxvol, arrays.uid[j], arrays.uid[i], arrays.mergeable[j], arrays.mergeIndicator[j]);
		}
		printf("222222222////////////////maxvol %f vol %f rat %f uid %d\n", maxvol, arrays.volume.first[i].val, arrays.volume.first[i].val / maxvol, arrays.uid[i]);
	}*/
	/*if (arrays.particle_type[i] != 0) 
		printf(" !!!!!!!MERGE cadidate uid %d type %d\n", arrays.uid[i], arrays.particle_type[i]);
	*/
	float counter = arrays.mergeCounter[i] + 1.f;
	
	//printf("---- counter %f uid %d ind %d\n", counter, arrays.uid[i], i);
	iterateNeighbors(j) {
		if (arrays.particle_type[j] == 0 || i == j) continue;
	
		/*printf("mergeable %d uid %d ii %d jj %d\n", arrays.mergeable[j], arrays.uid[j], i, j);
		*/
		
		//printf("////////////////maxvol1111 %f neighvol %f rat %f uid %d muid %d\n", maxvol, arrays.volume.first[j].val, arrays.volume.first[j].val/maxvol, arrays.uid[j], arrays.uid[i]);
		
		//printf("////////////////maxvol22222 %f neighvol %f rat %f uid %d muid %d\n", maxvol, arrays.volume.first[j].val, arrays.volume.first[j].val/maxvol, arrays.uid[j], arrays.uid[i]);
		auto sum_volume = arrays.volume.first[j] + arrays.volume.first[i] / counter;
		auto sum_radius = math::power<ratio<1, 3>>(sum_volume * PI4O3_1);
		auto targetVolume = level_estimate(arrays, -arrays.distanceBuffer.first[j]);
				
		if (arrays.distanceBuffer.first[j].val >= 0.0) {
			int flid = (int)arrays.distanceBuffer.first[j].val;
			targetVolume.val = arrays.volume.first[flid].val * 4;
		}
		else targetVolume.val = PI4O3 * (arrays.radius * arrays.radius * arrays.radius).val;
		
		//printf("++++++++++++++sumvoleven %f volj %f %d uid %d %f voli %f %d uid %d %f cnt %f targetvol %f\n", 
			//sum_volume.val, arrays.volume.first[j].val, j, arrays.uid[j], arrays.position.first[j].val.w, arrays.volume.first[i].val, i, arrays.uid[i], arrays.position.first[i].val.w, counter, targetVolume.val);
	
		if (sum_volume > targetVolume *1.4f || sum_radius > arrays.radius) continue;

		if (sum_radius <= arrays.radius &&
			arrays.lifetime[j].val > 10.f * arrays.timestep.val) {
				
				
			//printf("////////////////maxvol3333 %f neighvol %f rat %f uid %d muid %d class %f\n", 
				//maxvol, arrays.volume.first[j].val, arrays.volume.first[j].val/maxvol, arrays.uid[j], arrays.uid[i], classification[j]);
			if (!(arrays.adaptiveClassification[j] < 1.5f))
				continue;
			//printf("////////////////maxvol4444 %f neighvol %f rat %f uid %d muid %d\n", maxvol, arrays.volume.first[j].val, arrays.volume.first[j].val/maxvol, arrays.uid[j], arrays.uid[i]);

			cuda_atomic<int32_t> neighbor_mergeable(arrays.mergeable + j);
			int32_t cas_val = neighbor_mergeable.CAS(-1, i);
			if (cas_val != -1)
				continue;
			//printf("////////////////maxvol5555 %f neighvol %f rat %f uid %d muid %d\n", maxvol, arrays.volume.first[j].val, arrays.volume.first[j].val/maxvol, arrays.uid[j], arrays.uid[i]);
					
			auto sum_volume = arrays.volume.first[j] + arrays.volume.first[i] / counter;;
			auto sum_radius = math::power<ratio<1, 3>>(sum_volume * PI4O3_1);
			auto targetVolume = level_estimate(arrays, -arrays.distanceBuffer.first[j]);
			
			cuda_atomic<int32_t> neighbor_mergeable1(arrays.mergeable + i);
			int32_t cas_val1 = neighbor_mergeable1.CAS(-1, -2);
			if (cas_val1 != -1 && cas_val1 != -2)
			{
				neighbor_mergeable.CAS(i, -1);
				return;
			}
			arrays.mergeCounter[i]++;
			//printf("---- counter last %d uid %d ind %d juid %d jind %d\n", arrays.mergeCounter[i], arrays.uid[i], i, arrays.uid[j], j);

			counter++;
				
		}
		
	}
	//arrays.debugArray[i] = dbgVal;
}

// Finds merging partners with odd indices for particles with even indices. Found particles will be
// marked with the index of the particle they will be merged with and merging candidates that have
// found a partner will be marked -2. This function will only find a partner if the partner is of
// the correct classification and the merge result would not violate the systemic limit to particle
// sizes. This function is an almost duplicate of mergeGrabEven.
neighFunctionType mergeGrabOddBoundary(SPH::adaptive::Memory arrays, int32_t* idxArray) {
	checkedParticleIdx(i);
	i = idxArray[i];
	if (arrays.particle_type[i] == 0) return;
	if (i % 2 != 0 || arrays.mergeable[i] != -1 || arrays.adaptiveClassification[i] > -1.5f)
		return;
				
	float maxvol = PI4O3 * (arrays.radius * arrays.radius * arrays.radius).val;

	float counter = arrays.mergeCounter[i] + 1.f;
	
	//printf("---- counter %f uid %d ind %d\n", counter, arrays.uid[i], i);
	iterateNeighbors(j) {
		if (arrays.particle_type[j] == 0 || i == j) continue;
		auto sum_volume = arrays.volume.first[j] + arrays.volume.first[i] / counter;
		auto sum_radius = math::power<ratio<1, 3>>(sum_volume * PI4O3_1);
		auto targetVolume = level_estimate(arrays, -arrays.distanceBuffer.first[j]);
				
		if (arrays.distanceBuffer.first[j].val >= 0.0) {
			int flid = (int)arrays.distanceBuffer.first[j].val;
			targetVolume.val = arrays.volume.first[flid].val * 4;
		}
		else targetVolume.val = PI4O3 * (arrays.radius * arrays.radius * arrays.radius).val;
		
		//printf("++++++++++++++sumvolodd %f volj %f %d uid %d %f voli %f %d uid %d %f cnt %f targetvol %f\n", 
			//sum_volume.val, arrays.volume.first[j].val, j, arrays.uid[j], arrays.position.first[j].val.w, arrays.volume.first[i].val, i, arrays.uid[i], arrays.position.first[i].val.w, counter, targetVolume.val);
	
		if (sum_volume > targetVolume *1.4f || sum_radius > arrays.radius) continue;

		if (sum_radius <= arrays.radius &&
			arrays.lifetime[j].val > 10.f * arrays.timestep.val) {
				
				
			//printf("////////////////maxvol3333 %f neighvol %f rat %f uid %d muid %d class %f\n", 
				//maxvol, arrays.volume.first[j].val, arrays.volume.first[j].val/maxvol, arrays.uid[j], arrays.uid[i], classification[j]);
			if (!(arrays.adaptiveClassification[j] < 1.5f))
				continue;
			//printf("////////////////maxvol4444 %f neighvol %f rat %f uid %d muid %d\n", maxvol, arrays.volume.first[j].val, arrays.volume.first[j].val/maxvol, arrays.uid[j], arrays.uid[i]);

			cuda_atomic<int32_t> neighbor_mergeable(arrays.mergeable + j);
			int32_t cas_val = neighbor_mergeable.CAS(-1, i);
			if (cas_val != -1)
				continue;
			//printf("////////////////maxvol5555 %f neighvol %f rat %f uid %d muid %d\n", maxvol, arrays.volume.first[j].val, arrays.volume.first[j].val/maxvol, arrays.uid[j], arrays.uid[i]);
					
			auto sum_volume = arrays.volume.first[j] + arrays.volume.first[i] / counter;;
			auto sum_radius = math::power<ratio<1, 3>>(sum_volume * PI4O3_1);
			auto targetVolume = level_estimate(arrays, -arrays.distanceBuffer.first[j]);
			
			cuda_atomic<int32_t> neighbor_mergeable1(arrays.mergeable + i);
			int32_t cas_val1 = neighbor_mergeable1.CAS(-1, -2);
			if (cas_val1 != -1 && cas_val1 != -2)
			{
				neighbor_mergeable.CAS(i, -1);
				return;
			}
			arrays.mergeCounter[i]++;
			//printf("---- counter last %d uid %d ind %d juid %d jind %d\n", arrays.mergeCounter[i], arrays.uid[i], i, arrays.uid[j], j);

			counter++;
				
		}
		
	}
}

// Variadic recursion base
hostDeviceInline void mergeValueBoundary(uint32_t, uint32_t, float_u<SI::volume>, float_u<SI::volume>) {}

// Caclulates the mass weighed average of a pointer pair stored in arg based on the particle masses.
// This function is recursive with respect to it's variadic argument list.
template <typename T, typename... Ts>
hostDeviceInline void mergeValueBoundary(uint32_t particle_idx, uint32_t neighbor_idx, float_u<SI::volume> particle_mass,
	float_u<SI::volume> partner_mass, T arg, Ts... ref) {
	if (arg != nullptr)
		arg[neighbor_idx] = static_cast<typename std::decay<decltype(*arg)>::type>(((arg[particle_idx] * particle_mass.val + arg[neighbor_idx] * partner_mass.val) / (particle_mass.val + partner_mass.val)));
	mergeValueBoundary(particle_idx, neighbor_idx, particle_mass, partner_mass, ref...);
}

// This function merges all particles that have found merging partners with their appropriate
// merging partners. The particle that is being merged is removed from the simulation by being
// marked invalid.
neighFunctionDeviceType mergeGrabbedBoundary(SPH::adaptive::Memory arrays, Ts... tup) {
	checkedParticleIdx(i);
	//if (arrays.particle_type[i] != 0) return;
	//return;
	if (arrays.particle_type[i] == 0) return;
	if (arrays.mergeable[i] != -2)
		return;
	if (arrays.adaptiveClassification[i] > -1.5f)
		return;
	cuda_atomic<int32_t> num_ptcls(arrays.ptclCounter);
	++num_ptcls;
	//printf("BIIIG type1 %d\n", arrays.particle_type[i]);

	float counter = static_cast<float>(arrays.mergeCounter[i]);
	float initS = M_PI * math::pow(3 * arrays.volume.first[i].val / 4 / M_PI, 2.0 / 3.0);
	auto divS = initS / counter;
	float V_i = math::pow(divS / M_PI, 3.0 / 2.0) * 4 * M_PI / 3;

	atomicAdd(arrays.adaptivityCounter + (math::clamp(arrays.mergeCounter[i], 1, 16) - 1), 1);
	iterateNeighbors(j) {
		if (arrays.particle_type[j] == 0) continue;
		if (arrays.mergeable[j] != i)
			continue;
		//printf("type1 %d type2 %d\n", arrays.particle_type[i], arrays.particle_type[j]);
		auto V_j = arrays.volume.first[j].val;
		auto V_m = V_i + V_j;
		//V_m.val = math::pow(V_m.val, 3.0/2.0);
		auto s_new = math::pow(3.0 / 4.0, 2.0 / 3.0) * math::pow(V_i, 2.0 / 3.0);
		auto s_old = math::pow(3.0 / 4.0, 2.0 / 3.0) * math::pow(V_j, 2.0 / 3.0);
		//V_m.val = math::pow((s_old + s_new)/s_old, 3.0/2.0) * math::pow(V_j.val, 2.0/3.0);
		
		V_m = math::pow(math::pow(V_i, 2.0 / 3.0) + math::pow(V_j, 2.0 / 3.0), 3.0/2.0);

		float firstS = M_PI * math::pow(3 * V_i / 4 / M_PI, 2.0 / 3.0);
		float lastS = M_PI * math::pow(3 * V_j / 4 / M_PI, 2.0 / 3.0);
		float curS = M_PI * math::pow(3 * V_m / 4 / M_PI, 2.0 / 3.0);

		int olduid = arrays.uid[j];
		//printf("++++++++++++++++++++++++OLD1 %f uid %d OLD2 %f uid %d new %f newV %f\n", firstS, arrays.uid[i], lastS, arrays.uid[j], curS, V_m);
		mergeValueBoundary(i, j, V_i, V_j, tup...);

		auto tmpvol = arrays.volume.first[j];
		tmpvol.val = V_m;

		arrays.splitIndicator[j] = 2;
		arrays.lifetime[j] = -0.f * arrays.blendSteps * arrays.timestep *0.25f;
		arrays.volume.first[j].val = V_m;
		arrays.uid[j] = olduid;
		if (arrays.particle_type[i] != 0) arrays.particle_type[j] = arrays.particle_type[i];
		arrays.optimization_lifetime_merge[j] = 10;
		arrays.optimization_group[j].val = i;
		auto h = support_from_volume(tmpvol).val;
		float threshold = 0.24509788f * h * kernelSize() * 1.f;
		auto pDist = planeBoundary::distance(arrays.position.first[j], V_m, arrays);
		//if (pDist.val.w < threshold)
		//	arrays.position.first[j] -= (pDist.val) * (pDist.val.w - threshold);
		math::unit_assign<4>(arrays.position.first[j], support_from_volume(V_m));
		for (int32_t jj1 = 0, j1 = arrays.neighborList[j]; jj1 < arrays.neighborListLength[j]; (++jj1), (j1 = arrays.neighborList[j + jj1 * arrays.maxNumptcls])) {
			if (arrays.particle_type[j1] == 0) continue;
			arrays.optimization_group[j1] = i;
			arrays.optimization_lifetime_merge[j1] = 10;
		}
		//printf("new type1 %d type2 %d\n", arrays.particle_type[i], arrays.particle_type[j]);
	}
	math::unit_assign<4>(arrays.position.first[i], float_u<SI::m>(FLT_MAX));
}

neighFunction(detectMergingParticlesBoundary, mergeDetectBoundary, "Adaptive: init (merge)");
neighFunction(grabEvenMergingParticlesBoundary, mergeGrabEvenBoundary, "Adaptive: find even partners (merge)", caches<float4, float, float>{});
neighFunction(grabOddMergingParticlesBoundary, mergeGrabOddBoundary, "Adaptive: find odd partners (merge)", caches<float4, float, float>{});
neighFunctionDevice(mergeParticlesBoundary, mergeGrabbedBoundary, "Adaptive: merging particles");

// Helper function to call the merging function properly by transforming the arguments from a tuple to a variadic list
template <typename... Ts> auto MergeGrabbedBoundary(std::tuple<Ts...>, SPH::adaptive::Memory arrays) {
	launchDevice<mergeParticlesBoundary>(
		arrays.num_ptcls, arrays, (typename Ts::unit_type *)Ts::ptr...);
}
