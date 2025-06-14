#include <SPH/surface/surfaceDetection.cuh>
#include <utility/include_all.h>

neighFunctionType estimateNormal(SPH::detection::Memory arrays) {
	checkedParticleIdx(i);
	if (arrays.particle_type[i] != 0) return;

	cache_arrays((pos, position), (vol, volume));
	float4 normal{ 0.f,0.f,0.f,0.f };
	iterateNeighbors(j) {
		if (i == j || arrays.particle_type[j] != 0) continue;
		auto distance = pos[i] - pos[j];
		//normal += arrays.volume[j] * math::normalize3(distance);
		normal += -arrays.volume[j] / arrays.density[j] * GW_ij;
	}
	arrays.particleNormal[i] = math::normalize3(math::castTo<float4>(normal));
}

neighFunctionType initFluidNeighbor(SPH::detection::Memory arrays) {
	checkedParticleIdx(i);
	cache_arrays((pos, position), (vol, volume));
	float4 normal{ 0.f,0.f,0.f,0.f };
	if (arrays.particle_type[i] == 0 || arrays.distanceBuffer.first[i] == -3) return;
	arrays.distanceBuffer.first[i] = -1; 
}

neighFunctionType estimateFluidNeighbor(SPH::detection::Memory arrays) {
	checkedParticleIdx(i);
	cache_arrays((pos, position), (vol, volume));
	float4 normal{ 0.f,0.f,0.f,0.f };
	if (arrays.particle_type[i] == 0 || arrays.distanceBuffer.first[i] == -3) return;
	iterateNeighbors(j) {
		if (arrays.particle_type[j] == 0)
			if (arrays.volume[(int)arrays.distanceBuffer.first[i].val].val < arrays.volume[j].val)
				arrays.distanceBuffer.first[i].val = j;
	}
}

neighFunctionType detectSurface(SPH::detection::Memory arrays) {
	checkedParticleIdx(i);
	//return;
	if (arrays.particle_type[i] != 0) return;
	float4 normal = arrays.particleNormal[i];
	bool state = false;
	bool bpnear = false;
	float dist_to_bparticle = 2 * arrays.radius.val;
	iterateNeighbors(j) {
		if (i == j || arrays.particle_type[j] != 0)
		{
			if (arrays.particle_type[j] != 0)
			{
				//state = true;
				auto distance = arrays.position[j] - arrays.position[i];
				float tmpdist = math::length3(distance).val;
				if (tmpdist < dist_to_bparticle) dist_to_bparticle = tmpdist;
			}
			continue;
		}
		//if (arrays.particle_type[j] != 0) bpnear = true;
		auto distance = arrays.position[j] - arrays.position[i];
		auto angle = acosf(math::dot3(normal, math::normalize3(distance)).val);
		state = state || angle <= CUDART_PI_F / 4.f;
	}

	//if (dist_to_bparticle < arrays.radius) state = false;

	

	auto V0 = 4.f / 3.f * CUDART_PI_F * math::cubic(arrays.radius);
	auto h0 = support_from_volume(V0) * kernelSize() * 2.f;
	float4 dd = {100.f, 0.f, 0.f, -1.f};

	iterateNeighbors(j) {
		if (arrays.particle_type[j] == 0) continue;
		auto tmpdd = arrays.position[j] - arrays.position[i];
		if (dd.w == -1 || math::length3(tmpdd.val) < math::length3(dd))
			dd = tmpdd.val;
	}
	auto angle1 = acosf(math::dot3(normal, math::normalize3(dd)));
	//arrays.debugArray[i] = float4{ 0.f, 0.f, 0.f, h0.val / 2.f * 1.1f };
	if (math::length3(dd) < h0) {
		//printf("((((BOUDNARy dist %f\n", math::length3(dd));

		state = state || angle1 <= CUDART_PI_F / 2.f;
	}
	//if (arrays.uid[i] == 1)
	//	printf("((((BOUDNARy dist %f\n", math::length3(dd));

	//printf("++++1 state %d\n", state);
	auto d = planeBoundary::distance(arrays.position[i], arrays.volume[i], arrays);
	auto db = d;
	//auto dist = math::planeDistance(E, arrays.position[i]);
	auto angle = acosf(math::dot3(normal, -math::normalize3(d)).val);
	//arrays.debugArray[i] = float4{ 0.f, 0.f, 0.f, h0.val / 2.f * 1.1f };
	bool volumeState = false;
	if (d.val.w < h0) {
		state = state || angle <= CUDART_PI_F /2.f;
		//arrays.debugArray[i] = float4{ d.val.x, d.val.y, d.val.z, d.val.w };

		auto x = d.val.w;
		auto h = support_from_volume(arrays.volume[i]);
		auto H = h.val * kernelSize();
		auto xRel = math::clamp((x + H) / (2.f * H), 0.f, 1.f) * ((float)arrays.boundaryLUTSize - 1.f);
		auto xL = math::floorf(xRel);
		auto xH = math::ceilf(xRel);
		auto xD = xRel - xL;
		int32_t xLi = math::clamp(static_cast<int32_t>(xL), 0, (int32_t)arrays.boundaryLUTSize - 1);
		int32_t xHi = math::clamp(static_cast<int32_t>(xH), 0, (int32_t)arrays.boundaryLUTSize - 1);
		auto lL = arrays.splineLUT[xLi];
		auto lH = arrays.splineLUT[xHi];
		auto val = lL * xD + (1.f - xD) * lH;
		//arrays.debugArray[i] = float4{ val.val, xRel, boundary::g(d, h), x };
		//arrays.debugArray[i] = db.val;
		//arrays.debugArray[i] = boundary::splineGradient(arrays.position[i], arrays.volume[i], uFloat<>{0.5f}, arrays, boundary::kind::plane, -1).val;
	}
	//printf("++++2 state %d\n", state);

	for (int32_t v = 0; v < arrays.volumeBoundaryCounter; ++v) {
		auto d = volumeBoundary::distance_fn(arrays.position[i], arrays.volume[i], arrays, v);

		//auto dist = math::planeDistance(E, arrays.position[i]);
		auto angle = acosf(math::dot3(normal, -math::normalize3(d)).val);
		auto h0 = support_from_volume(V0) * kernelSize() * 2.f;
		//arrays.debugArray[i] = float4{ 0.f, 0.f, 0.f, h0.val / 2.f * 1.05f };
		if (d.val.w < h0 && d.val.w < db.val.w) {
			//state = state || angle <= CUDART_PI_F / 2.f;

			auto x = d.val.w;
			auto h = support_from_volume(arrays.volume[i]);
			auto H = h.val * kernelSize();
			auto xRel = math::clamp((x + H) / (2.f * H), 0.f, 1.f) * ((float)arrays.boundaryLUTSize - 1.f);
			auto xL = math::floorf(xRel);
			auto xH = math::ceilf(xRel);
			auto xD = xRel - xL;
			int32_t xLi = math::clamp(static_cast<int32_t>(xL), 0, (int32_t)arrays.boundaryLUTSize - 1);
			int32_t xHi = math::clamp(static_cast<int32_t>(xH), 0, (int32_t)arrays.boundaryLUTSize - 1);
			auto lL = arrays.splineLUT[xLi];
			auto lH = arrays.splineLUT[xHi];
			auto val = lL * xD + (1.f - xD) * lH;
			//arrays.debugArray[i] = float4{ val.val, xRel, boundary::g(d, h), x };
			//arrays.debugArray[i] = d.val;
			volumeState = true;

			//arrays.debugArray[i] = float4{
			//	boundary::g(d,h),
			//	boundary::lookupGradient(arrays, arrays.splineGradientLUT, arrays.boundaryLUTSize, d, arrays.volume[i], uFloat<>{0.5f}, support_from_volume(arrays.volume[i]),0.f).val,
			//	boundary::dg(d,h),
			//	boundary::lookupValue(arrays, arrays.splineLUT, arrays.boundaryLUTSize, d, arrays.volume[i], uFloat<>{0.5f}, support_from_volume(arrays.volume[i]),0.f).val
			//};
			//arrays.debugArray[i] = boundary::internal::lookupGradient(arrays.splineLUT, arrays.splineGradientLUT, arrays.position[i], arrays.volume[i], uFloat<>{0.5f}, arrays, boundary::kind::volume, v).val;

				//arrays.debugArray[i] = boundary::splineGradient(arrays.position[i], arrays.volume[i], uFloat<>{0.5f}, arrays, boundary::kind::volume, v).val;
		}
	}

	//printf("++++3 state %d\n", state);
	if (arrays.neighborListLength[i] < 5) state = false;
	iterateBoundaryPlanes(E) {
		auto dist = math::planeDistance(E, arrays.position[i]);
		if (
			(dist.val < math::unit_get<1>(arrays.surface_distanceFieldDistances).val && fabsf(math::dot3(E, float4_u<>{1.f, 0.f, 0.f, 0.f}).val) > 0.5f) ||
			(dist.val < math::unit_get<2>(arrays.surface_distanceFieldDistances).val&& fabsf(math::dot3(E, float4_u<>{0.f, 1.f, 0.f, 0.f}).val) > 0.5f) ||
			(dist.val < math::unit_get<3>(arrays.surface_distanceFieldDistances).val&& fabsf(math::dot3(E, float4_u<>{0.f, 0.f, 1.f, 0.f}).val) > 0.5f && E.val.z > 0.f)) {
			//printf("b");
			state = true;
		}
	}
	//printf("++++4 state %d\n", state);

	if (bpnear) state = true;
	if (volumeState)
		state = true;
	//printf("++++5 state %d\n", state);
	//for (int32_t b = 0; b < arrays.volumeBoundaryCounter; ++b) {
	//	auto vos = volumeBoundary::distance_fn(arrays.position[i], V0, b);
	//	if (vos.val.w < HforV1 && math::dot3(vos, normal) < 0.f)
	//		state = true;
	//}
	//state = true;
	auto r = math::power<ratio<1, 3>>(arrays.volume[i] * PI4O3_1);
	auto phi = state ? arrays.surface_levelLimit : 0.f;
	auto phiOld = arrays.distanceBuffer.first[i];
	//printf("++++1 phi %f phiold %f\n", phi.val, phiOld.val);
	arrays.particleNormal[i].w = phiOld.val;
	phi = state ? arrays.surface_levelLimit : phi;
	//printf("++++2 phi %f phiold %f\n", phi.val, phiOld.val);
	phi = math::clamp(phi, arrays.surface_levelLimit, 0.f);
	//printf("++++3 phi %f phiold %f\n", phi.val, phiOld.val);
	phi = math::clamp(phi, phiOld - 1.0f * r, phiOld + 1.0f * r);
	//printf("state %d phi %f phiOld %f\n", state, phi.val, phiOld.val);
	arrays.distanceBuffer.second[i] = phi;
	arrays.distanceBuffer.first[i] = phiOld;
	if (arrays.uid[i] == 0 || arrays.uid[i] == 1000)
	//if (arrays.distanceBuffer.second[i].val == arrays.distanceBuffer.second[0].val)
		printf("distBuf1 new %f old %f id %d limit %f state %d\n", arrays.distanceBuffer.second[i].val, arrays.distanceBuffer.first[i].val, i, arrays.surface_levelLimit.val, state);
}


basicFunctionType correctEstimate(SPH::detection::Memory arrays) {
	checkedParticleIdx(i);
	if (arrays.particle_type[i] != 0) return;

	auto r = math::power<ratio<1, 3>>(arrays.volume[i] * PI4O3_1);

	auto phi = arrays.distanceBuffer.second[i];
	arrays.distanceBuffer.second[i] = phi;
	arrays.decisionBuffer[i] = phi >= -0.85f * r ? 1.f : 0.f;
	arrays.markerBuffer[i] = phi >= -0.85f * r ? 1.f : 0.f;

	if (arrays.markerBuffer[i] < 0.4f) {
		arrays.surface_idxBuffer.second[i] = i;
	}
	//if (arrays.uid[i] == 0 || arrays.uid[i] == 1000)
	//if (arrays.distanceBuffer.second[i].val == arrays.distanceBuffer.second[0].val)
	//	printf("distBuf2 new %f old %f id %d limit %f state %d\n", arrays.distanceBuffer.second[i].val, arrays.distanceBuffer.first[i].val, i, arrays.surface_levelLimit.val);

}
neighFunctionType propagateSurface(SPH::detection::Memory arrays, int32_t threads) {
	checkedThreadIdx(t);
	alias_arrays((pos, position));
	int32_t i = arrays.surface_idxBuffer.first[t];
	if (arrays.particle_type[i] != 0) return;
	//if (arrays.particle_type[t] != 0) return;

	if (i == INT_MIN)
		return;

	int32_t partnerIdx = INT_MAX;
	float_u<SI::m> partnerDistance{ FLT_MAX };
	auto partnerPhi = 0.0_m;
	//auto r = math::power<ratio<1, 3>>(arrays.volume[i] * PI4O3_1);
	arrays.markerBuffer[i] = arrays.decisionBuffer[i];
	float marker = arrays.markerBuffer[i];

	iterateNeighbors(j) {
		if (W_ij > 0.f) {
			if (j == i || arrays.particle_type[j] != 0)
				continue;
			float neighbor_decision = arrays.decisionBuffer[j];
			if (neighbor_decision > 0.2f && marker < 0.05f) {
				auto dist = math::abs(math::distance3(pos[i], pos[j]));
				if (dist < partnerDistance) {
					partnerIdx = j;
					partnerDistance = dist;
					partnerPhi = arrays.distanceBuffer.second[j];
				}
			}
		}
		if (partnerIdx != INT_MAX) {
			if (arrays.decisionBuffer[i] < 0.4f) {
				auto phi = partnerPhi - partnerDistance;
				bool changed = phi > arrays.surface_levelLimit.val * 2.f * arrays.radius;
				if (arrays.distanceBuffer.second[i] != phi && changed) {
					cuda_atomic<float> change(arrays.changeBuffer);
					change.add(1.f);
					arrays.distanceBuffer.second[i] = phi;
				}
				arrays.markerBuffer[i] = changed ? 0.5f : 0.1f;
			}
		}
	}
	if (arrays.markerBuffer[i] < 0.4f) {
		arrays.surface_idxBuffer.second[t] = i;
	}
	else {
		arrays.surface_idxBuffer.second[t] = 0xFFFFFFFF;
	}
	//if (arrays.uid[i] == 0 || arrays.uid[i] == 1000)
	/*if (arrays.distanceBuffer.second[i].val == arrays.distanceBuffer.second[0].val)
		printf("distBuf3 new %f old %f id %d limit %f state %d\n", arrays.distanceBuffer.second[i].val, arrays.distanceBuffer.first[i].val, i, arrays.surface_levelLimit.val);*/
}
neighFunctionType phiSmooth(SPH::detection::Memory arrays) {
	checkedParticleIdx(i);
	if (arrays.particle_type[i] != 0) return;

	cache_arrays((pos, position), (vol, volume));

	arrays.markerBuffer[i] = arrays.decisionBuffer[i];

	auto phiSum = 0.0_m;
	auto counter = 0.f;
	iterateNeighbors(j) {
		if (arrays.particle_type[j] != 0) continue;
		counter++;
		phiSum += arrays.distanceBuffer.second[j] * W_ij * vol[j]; // / arrays.density[neigh];
	}

	//SWH2<SPH::detection::Memory> swh(arrays, pos[i], vol[i]);
	auto POS = planeBoundary::distance(pos[i], vol[i], arrays);
	auto r = math::power<ratio<1, 3>>(arrays.volume[i] * PI4O3_1);
	//arrays.debugArray[i].w = phiOld;
	auto phiOld = arrays.particleNormal[i].w;

	//if (POS.val.w < 1e20f || counter < 5)
	phiSum = arrays.distanceBuffer.second[i];

	phiSum = math::clamp(phiSum, phiOld - 2.0f * r, phiOld + 2.0f * r);
	phiSum = math::clamp(phiSum, arrays.surface_levelLimit, 0.f);

	arrays.distanceBuffer.first[i] = math::max(phiSum, arrays.surface_levelLimit);
	/*if (arrays.distanceBuffer.first[i].val == arrays.distanceBuffer.first[0].val)
		printf("distBuf3 new %f old %f id %d limit %f state %d\n", arrays.distanceBuffer.second[i].val, arrays.distanceBuffer.first[i].val, i, arrays.surface_levelLimit.val);*/

	//printf("distanceBuffer %f\n", arrays.distanceBuffer.first[i].val);
}
basicFunction(correct, correctEstimate, "Surface: correct Distance");
neighFunction(propagate, propagateSurface, "Surface: Distance iteration");
neighFunction(smooth, phiSmooth, "Surface: smooth Distance", caches<float4, float>{});

struct is_set {
	hostDeviceInline bool operator()(const int x) { return x != -1; }
};

neighFunction(estimate, estimateNormal, "Surface: estimate Normal", caches<float4, float>{});
neighFunction(initFluid, initFluidNeighbor, "Surface: estimate Normal", caches<float4, float>{});
neighFunction(estimateFluid, estimateFluidNeighbor, "Surface: estimate Normal", caches<float4, float>{});
neighFunction(detect, detectSurface, "Surface: detect surface");

void SPH::detection::distance(Memory mem) {
	if (mem.num_ptcls == 0) return;
	int32_t diff = 0;
	auto compact_idx = [&]() {
		diff = (int32_t)algorithm::copy_if(arrays::surface_idxBuffer::rear_ptr, arrays::surface_idxBuffer::ptr, mem.num_ptcls, is_set());
		cuda::Memset(mem.surface_idxBuffer.second, 0xFF, sizeof(int32_t) * mem.num_ptcls);
	};
	cuda::Memset(mem.surface_idxBuffer.second, 0xFF, sizeof(int32_t) * mem.num_ptcls);
	launch<estimate>(mem.num_ptcls, mem);
	launch<detect>(mem.num_ptcls, mem);
	launch<correct>(mem.num_ptcls, mem);
	compact_idx();
	int32_t it = 0;
	do {
		cuda::Memset(mem.changeBuffer, 0x00, sizeof(float));
		launch<propagate>(diff, mem, diff);
		cuda::memcpy(&mem.surface_phiChange, mem.changeBuffer, sizeof(float), cudaMemcpyDeviceToHost);
		cuda::memcpy(mem.decisionBuffer, mem.markerBuffer, sizeof(float) * mem.num_ptcls);
		it++;
		if (it % 4 == 0)
			compact_idx();
	} while (mem.surface_phiChange >= 0.5f);
	get<parameters::surfaceDistance::surface_iterations>() = it;
	launch<smooth>(mem.num_ptcls, mem);
	uGet<parameters::surfaceDistance::surface_phiMin>() = algorithm::reduce_max(mem.distanceBuffer.first, mem.num_ptcls);
	float phimin = 0.123f;
	for (int i = 0; i < mem.num_ptcls; i++)
	{
		if ((phimin == 0.123f || phimin < mem.distanceBuffer.first[i].val) && mem.particle_type[i] == 0)
			phimin = mem.distanceBuffer.first[i].val;
	}
	std::cout << "8888888888888888!! phiorig " << uGet<parameters::surfaceDistance::surface_phiMin>().val << " phimine " << phimin << "\n";
	uGet<parameters::surfaceDistance::surface_phiMin>().val = phimin;
}

void SPH::detection::showBuffer(Memory mem) {
	for (int i = 0; i < mem.num_ptcls; i++)
	{
		//if (mem.particle_type[i] == 0) std::cout << "BUUUUUFFFF " << mem.distanceBuffer.first[0].val << " " << mem.distanceBuffer.second[0].val << std::endl;
		//if (mem.uid[i] == 0) mem.first_fluid = i;
		//if (mem.particle_type[i] != 0) continue;
		//if (mem.distanceBuffer.first[i].val == mem.distanceBuffer.first[0].val)
		//	std::cout << "*********!!!!!-----BUF " << mem.distanceBuffer.first[i].val << " " << mem.distanceBuffer.first[0].val << std::endl;
	}
	std::cout << "*********AAAAAAAAAAAAAAAAAA shobuf end" << std::endl;
}

void SPH::detection::fluidneighbor(Memory mem) {
	launch<initFluid>(mem.num_ptcls, mem);
	launch<estimateFluid>(mem.num_ptcls, mem);

}