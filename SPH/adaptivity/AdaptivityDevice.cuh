#pragma once
#include <SPH/adaptivity/ContinuousAdaptivity.cuh>
#include <utility/include_all.h>
#include <curand.h>
#include <curand_kernel.h>
#include <random>
#include <cstdlib>
#include <ctime>

/**Function used to estimate the target resolution (as volume) for a particle based on the distance
 * to the surface (as absolute distance not signed)**/
hostDeviceInline auto level_estimate(SPH::adaptive::Memory &arrays, value_unit<float, SI::m> val, int uid = -1) {
 auto clampedDistance = math::clamp(val - 2.f * math::abs(arrays.surface_phiMin), 0.0_m, math::abs(arrays.surface_levelLimit));
  auto scaledDistance  = clampedDistance / math::abs(arrays.surface_levelLimit);
  if (uid == 0)
	  printf("clampDist %f scaleDist %f val %f phimin %f\n", clampedDistance.val, scaledDistance.val, val.val, arrays.surface_phiMin.val);
  auto resolution      = (float) arrays.resolution;
  if (arrays.useVolume == 1) {
	   auto scalingFactor   = 1.f / resolution + scaledDistance * (1.f - 1.f / resolution);
	   return PI4O3 * math::power<3>(arrays.radius) * scalingFactor;
  } else {
    //auto scalingFactor = 1.f / (resolution - scaledDistance * (resolution - 1.f));
	// should be 1 @ 1
	// should be r @ 0
	// should be 0.5r @ 0.5
	// r - (r - 1) * s

    //return PI4O3 * math::power<3>(arrays.radius) * scalingFactor;
	  auto scalingFactor = 1.f / powf(resolution, 1.f / 3.f) + scaledDistance * (1.f - 1.f / powf(resolution, 1.f / 3.f));
	  auto volume = PI4O3 * math::power<3>(arrays.radius * scalingFactor);
	 return math::clamp(volume, PI4O3 * math::power<3>(arrays.radius) / ((float)resolution), PI4O3 * math::power<3>(arrays.radius));
  }
}

hostDeviceInline auto level_estimateVelocity(SPH::adaptive::Memory &arrays, value_unit<float4, SI::velocity> val, int id) {
  //printf("scaleDist %f val %f\n", scaledDistance.val, val.val);
	float threshold = 1.1f;
	auto resolution      = (float) arrays.resolution;
  float dif = arrays.max_vel - arrays.min_vel;
  //printf("dif %f max %f min %f\n", dif, arrays.max_vel, arrays.min_vel);
  if (dif < threshold) return arrays.volume.first[id];

  float scaledDistance = 1 - (math::length3(val).val - arrays.min_vel) / dif;
  //float scaledDistance = 1 ;

  auto scalingFactor   = 1.f / resolution + scaledDistance * (1.f - 1.f / resolution);
  return PI4O3 * math::power<3>(arrays.radius) * scalingFactor;
  
}

/** Function used to classify particles into sSlLo categories based on the difference to the ideal
  * particle size and their lifetime. **/
basicFunctionType decisionFunction(SPH::adaptive::Memory arrays) {
  checkedParticleIdx(i);
  //return;
  if (arrays.particle_type[i] != 0) return;
  if ((arrays.distanceBuffer.first[i] == -2 && arrays.particle_type[i] != 0)) return;
	/*if (arrays.distanceBuffer.first[i] == -3 && arrays.particle_type[i] != 0) {
		arrays.adaptiveClassification[i] = 0;
		return;
	}*/
  auto targetVolume  = level_estimate(arrays, -arrays.distanceBuffer.first[i], arrays.uid[i]);
  //auto targetVolume  = level_estimateVelocity(arrays, arrays.velocity[i], i);
  auto currentVolume = arrays.volume.first[i];

  if (arrays.uid[i] == 0 || arrays.uid[i] == 1000)
  //if (arrays.distanceBuffer.first[i].val != arrays.distanceBuffer.first[0].val)
	printf("-----++++distBuf %f uid %d type %d vol %f curvol %f\n", 
		arrays.distanceBuffer.first[i].val, arrays.uid[i], arrays.particle_type[i], targetVolume.val, currentVolume.val);
  

  auto ratio = ceilf(currentVolume.val / targetVolume.val);
  arrays.refinementRatio[i] = currentVolume.val / targetVolume.val;
  
  //printf("ratio %f targetvol %f curvol %f delay %f\n", ratio, targetVolume.val, currentVolume.val, arrays.delay.val);

	if (arrays.simulationTime < arrays.delay || arrays.lifetime[i] < 0.f /*|| (arrays.particle_type[i] != 0 && arrays.distanceBuffer.first[i].val == -1)*/)
		arrays.adaptiveClassification[i] = 0;
	else if (currentVolume < 0.5f * targetVolume) // S particle
		arrays.adaptiveClassification[i] = -2;
	else if (currentVolume < 0.9f * targetVolume) // s particle
		arrays.adaptiveClassification[i] = -1;
	else if (currentVolume > 1.99f * targetVolume)  // L particle
		arrays.adaptiveClassification[i] = ratio != ratio ? 0 : ratio;
	else if (currentVolume > 1.1f * targetVolume) // l particle
		arrays.adaptiveClassification[i] = 1;
	else                                      // 0 particle
		arrays.adaptiveClassification[i] = 0;

	  
}

basicFunctionType initArrayForDecision(SPH::adaptive::Memory arrays) {
	checkedParticleIdx(i);
	if (arrays.particle_type[i] == 0) return;
	arrays.mergeable[i] = -4;
	//printf("mergeable %d\n", arrays.mergeable[i]);
}

/** Function used to classify particles into sSlLo categories based on the difference to the ideal
  * particle size and their lifetime. **/
neighFunctionType decisionRefinementFunction(SPH::adaptive::Memory arrays) {
  checkedParticleIdx(i);
  //return;
  //if (arrays.particle_type[i] != 0) return;
  if (arrays.particle_type[i] == 0)
  {
	  int neighlist[1000] = { 0 };
	  int neighlistIncluded[1000] = { 0 };
	  int neighlen = 0;

	  float mindist = 100.f;
	  int minindex = 0;
	  
	  //iterateNeighbors(j) {
		 // for (int32_t jj1 = 0, j1 = arrays.neighborList[j]; jj1 < arrays.neighborListLength[j]; (++jj1), (j1 = arrays.neighborList[j + jj1 * arrays.maxNumptcls])) {
			//  if (arrays.particle_type[j1] == 0 || arrays.distanceBuffer.first[j1] == -3) continue;
			//  auto df = arrays.position.first[i] - arrays.position.first[j1];
			//  auto ln = math::length3(df);
			//  //printf("LLLLLEN %f\n", ln.val);
			//  if (mindist > ln.val) {
			//	  mindist = ln.val;
			//	  minindex = j1;
			//  }
		 // }
	  //}
	  
	  
	  iterateNeighbors(j) {
		  for (int32_t jj1 = 0, j1 = arrays.neighborList[j]; jj1 < arrays.neighborListLength[j]; (++jj1), (j1 = arrays.neighborList[j + jj1 * arrays.maxNumptcls])) {
			  if (arrays.particle_type[j1] == 0 /*|| arrays.distanceBuffer.first[j1] == -3*/) continue;
			  auto df = arrays.position.first[i] - arrays.position.first[j1];
			  auto ln = math::length3(df);
			  //printf("LLLLLEN %f\n", ln.val);
			  if (mindist > ln.val) {
				  mindist = ln.val;
				  minindex = j1;
			  }
		  }
	  }
	  //printf("//////////////////////////////////////mindist %f minid %d parent %d fluidparent %d\n", mindist, minindex, arrays.uid[minindex], arrays.uid[i]);
		  
	if (/*arrays.adaptiveClassification[j] > 1.f*/ mindist < arrays.radius.val*5)
	{
		for (int32_t jj1 = 0, j1 = arrays.neighborList[minindex]; jj1 < arrays.neighborListLength[minindex]; (++jj1), (j1 = arrays.neighborList[minindex + jj1 * arrays.maxNumptcls])) {
			//arrays.adaptiveClassification[j1] = 16;
		//printf("neighid %d\n", j1);
	//printf("neigh len %d %d\n", j1, arrays.neighborListLength[j1]);
			if (arrays.refinementListLength[j1] == 0) arrays.refinementListLength[j1] = arrays.neighborListLength[j1];
			if (arrays.particle_type[j1] != 0 /*&& arrays.refinementListLength[j1] >= 32*/)
			{

				bool isin = false;
				/*for (int nn = 0; nn < neighlen; nn++)
				{
					if (neighlist[nn] == j1) isin = true;
				}*/
				auto targetVolume  = level_estimate(arrays, -arrays.distanceBuffer.first[i]);
				targetVolume.val *= 4.0;
				//float rat = arrays.volume.first[j1].val / arrays.volume.first[i].val;
				float rat = arrays.volume.first[j1].val / targetVolume.val;
				/*if (rat > 5.0)
					printf("||||||||||RATIO %f TARGET %f\n", rat, targetVolume.val);*/
				if (rat < 0.5) {
					arrays.adaptiveClassification[j1] = -2;
					continue;
				}
				else if (rat >= 0.5 && rat < 0.9) {
					arrays.adaptiveClassification[j1] = -1;
					continue;
				}
				else if (!isin /*&& arrays.distanceBuffer.first[j1] != -3*/ && (rat > 2.6/* || rat < 0.5*/)) {
					cuda_atomic<int32_t> neighbor_mergeable(arrays.mergeable + j1);
					int32_t cas_val = neighbor_mergeable.CAS(-4, i);
					//printf("fuid %d buid %d cas_val %d merge %d i %d\n", arrays.uid[i], arrays.uid[j1], cas_val, arrays.mergeable[j1], i);
					if (cas_val != -4) continue;
					//printf("RRRAt %f \n", rat);
					if (arrays.splitIndicator[j1] == 2) {

						neighlist[neighlen++] = j1;
					}
					//else continue;
					//printf("-------------|||||| vol %f fluid %d\n", targetVolume.val, i);
					//auto tmp = arrays.volume.first[j1].val / targetVolume.val / 1;
					//auto tmp = arrays.volume.first[j1].val / arrays.volume.first[i].val / 4;
					//printf("NEIGH %d\n", j1);
					arrays.refinementRatio[j1] = rat;
					arrays.adaptiveClassification[j1] = 16;
					arrays.optimization_group[j1] = arrays.uid[i] + 5;
					//printf("RATIO %f\n", rat);
					/*if (minindex == j1)
						arrays.optimization_group[j1] += INT_MIN;
					*/		  
					//printf("post GROUOP %d minind %d j1 %d\n", arrays.optimization_group[j1].val, minindex, j1);

				}
				else arrays.adaptiveClassification[j1] = 0;
				
			}
		}
	}
		  
	  
	  //printf("NEIGHLEN %d\n", neighlen);
	float ThreeDTwoDRatio = 4.0;
	if (neighlen > 0)
	{
		//printf("---------------------begin++++++++++++\n");

		int grid[2000] = { 0 };
		float refinementRatio = ceil(math::pow(arrays.refinementRatio[neighlist[0]], 2.0/3.0));
		for (int ne = 0; ne < neighlen; ne++)
			if (arrays.refinementRatio[neighlist[ne]] > 1.0f)
			{
				refinementRatio = ceil(math::pow(arrays.refinementRatio[neighlist[ne]], 2.0/3.0));
				break;
			}
		
		for (int ne = 0; ne < neighlen; ne++)
			arrays.refinementRatio[neighlist[ne]] = refinementRatio;

		//float probability = ceil((refinementRatio - 1) * 100);

		int newAmount = neighlen * refinementRatio;
		newAmount *= 1.0;
		int difAmount = newAmount - neighlen;
		if (difAmount == 0) difAmount = ceil(neighlen / 2.0);
		/*if (difAmount != 0) difAmount += 2;
		if (neighlen > 0)
			difAmount = neighlen + 10;
		*/int cnt_grid = ceil(sqrt((float)difAmount));
		//int cnt_grid = ceil(sqrt(arrays.refinementListLength[i] / 2));
		//printf("break 1 id %d diff %d neigh %d new %d cntgrid %d refrat %f \n", i, difAmount, neighlen, newAmount, cnt_grid, refinementRatio);
		float maxy = std::numeric_limits<float>::lowest();
		float miny = std::numeric_limits<float>::max();
		float maxx = std::numeric_limits<float>::lowest();
		float minx = std::numeric_limits<float>::max();

		//printf("break 2 cntgrid %d\n", cnt_grid);
		for (int ii = 0; ii < neighlen; ii++)
		{
			auto ps = arrays.position.first[neighlist[ii]].val;
			//printf("neigh pos %f %f %f\n", ps.x, ps.y, ps.z);
			if (maxy < arrays.position.first[neighlist[ii]].val.y) maxy = arrays.position.first[neighlist[ii]].val.y;
			if (maxx < arrays.position.first[neighlist[ii]].val.x) maxx = arrays.position.first[neighlist[ii]].val.x;
			if (miny > arrays.position.first[neighlist[ii]].val.y) miny = arrays.position.first[neighlist[ii]].val.y;
			if (minx > arrays.position.first[neighlist[ii]].val.x) minx = arrays.position.first[neighlist[ii]].val.x;
		}
		//printf("break 3 maxy %f miny %f maxx %f minx %f cntgrid %d\n", maxy, miny, maxx, minx, cnt_grid);

		float stepx = (maxx - minx) / cnt_grid;
		float stepy = (maxy - miny) / cnt_grid;

		int allocatedAmount = 0;
		int fitInCell = 2;
		int splitFactor = 2;
		//printf("break 4\n");

		for (int kk = 0; kk < neighlen; kk++)
		{
			int indexy = floor((arrays.position.first[neighlist[kk]].val.y - miny) / stepy);
			int indexx = floor((arrays.position.first[neighlist[kk]].val.x - minx) / stepx);
			grid[indexy * cnt_grid + indexx]++;
		}

		int itr = 0;
		int itr_limit = 200;
		while (allocatedAmount < difAmount && itr < itr_limit)
		{
			itr++;
			//printf("selecting aloc %d dif %d\n", allocatedAmount, difAmount);
			int* tmpbuf = new int[neighlen]();

			for (int kk = 0; kk < neighlen; kk++)
			{
				int curindex = 0;

#if defined(__CUDA_ARCH__)
				curandState state;
				curand_init(clock64(), 0, 0, &state);
				curindex = floor(curand_uniform(&state) * neighlen);
				//curindex = curindex % neighlen;
				//whether_to_delete = false;

#else
				curindex = (std::rand() % (neighlen));
#endif

				int ii = curindex;
				if (arrays.adaptiveClassification[neighlist[ii]] >= 8 && arrays.adaptiveClassification[neighlist[ii]] != 16) {
					/*printf("i %d ii %d id %d adapclass %f neighlen %d refrat %f alloc %d diff %d\n", i, ii, neighlist[ii], arrays.adaptiveClassification[neighlist[ii]], 
						neighlen, refinementRatio, allocatedAmount, difAmount);
					*/
					if (neighlistIncluded[ii] < arrays.adaptiveClassification[neighlist[ii]])
					{
						//printf("AAAAAAAAAAAAAAAAAAAAAAAAAADATA\n");
						int addExtra = arrays.adaptiveClassification[neighlist[ii]] - neighlistIncluded[ii];
						allocatedAmount += addExtra;
						if (neighlistIncluded[ii] == 0) allocatedAmount -= 1;
						neighlistIncluded[ii] = arrays.adaptiveClassification[neighlist[ii]];
					}
					continue;
				}

				int isin = 0;
				for (int tt = 0; tt < kk; tt++)
					if (tmpbuf[tt] == ii)
					{
						isin = 1;
						kk--;
						break;
					}
				if (isin == 1) continue;

				tmpbuf[kk] = ii;
				//printf("break 6 neigh %d\n", neighlist[ii]);
					//arrays.adaptiveClassification[neighlist[ii]] = 2.0;
				/*int indexy = floor((arrays.position.first[neighlist[ii]].val.y - miny) / stepy);
				int indexx = floor((arrays.position.first[neighlist[ii]].val.x - minx) / stepx);
				*///printf("indexy %d indexx %d\n", indexy, indexx);
				//printf("break 7 neigh %d %d %d %d\n", indexy * cnt_grid + indexx, indexy, cnt_grid, indexx);
				if (/*grid[indexy * cnt_grid + indexx] < fitInCell &&*/ (arrays.adaptiveClassification[neighlist[ii]] == 16 || arrays.adaptiveClassification[neighlist[ii]] < splitFactor))
				{
					arrays.adaptiveClassification[neighlist[ii]] = splitFactor;
					//grid[indexy * cnt_grid + indexx] += splitFactor - neighlistIncluded[ii];
					allocatedAmount += splitFactor - neighlistIncluded[ii];
					if (neighlistIncluded[ii] == 0) allocatedAmount -= 1;
					neighlistIncluded[ii] = splitFactor;
				}
				else if (arrays.adaptiveClassification[neighlist[ii]] != 16 && neighlistIncluded[ii] < arrays.adaptiveClassification[neighlist[ii]])
				{
					//printf("AAAAAAAAAAAAAAAAAAAAAAAAAADATA\n");
					int addExtra = arrays.adaptiveClassification[neighlist[ii]] - neighlistIncluded[ii];
					allocatedAmount += addExtra;
					if (neighlistIncluded[ii] == 0) allocatedAmount -= 1;
					//grid[indexy * cnt_grid + indexx] += addExtra;
					neighlistIncluded[ii] = arrays.adaptiveClassification[neighlist[ii]];
				}
				//printf("break 8 neigh %d\n", neighlist[ii]);
				//else printf("indy %d indx %d incel %d\n", indexy, indexx, grid[indexy * cnt_grid + indexx]);
				if (allocatedAmount >= difAmount) break;
			}
			fitInCell++;
			splitFactor++;
			//printf("difamoung %d %d\n", difAmount, allocatedAmount );
			delete tmpbuf;
		}


		float refRat = 0.f;
		if (allocatedAmount < difAmount)
		{
			printf("**************************************\n");
			float newref = (allocatedAmount + neighlen)/neighlen;
			for (int kk = 0; kk < neighlen; kk++)
			{
				arrays.refinementRatio[neighlist[kk]] = newref;
			}
		}
		else if (allocatedAmount > difAmount)
		{
			refRat = float(allocatedAmount + neighlen) / neighlen;
			for (int ne = 0; ne < neighlen; ne++)
				arrays.refinementRatio[neighlist[ne]] = refRat;

		}
		/*for (int ne = 0; ne < neighlen; ne++)
			printf("====fuid %d adaptClass %f splitind %d partIndex %d\n", 
				arrays.uid[neighlist[ne]], arrays.adaptiveClassification[neighlist[ne]], arrays.splitIndicator[neighlist[ne]], arrays.particleIndex[neighlist[ne]]);
		*/
		//printf("||||||||||selecting for uid %d aloc %d dif %d oldref %f newref %f refrat %f neighlen %d\n", 
			//arrays.uid[i], allocatedAmount, difAmount, refinementRatio, arrays.refinementRatio[neighlist[0]], refRat, neighlen);
		//printf("end selecting111\n");


		/*for (int ii = 0; ii < neighlen; ii++)
		{
			printf("ii %d id %d refine %f\n", ii, neighlist[ii], arrays.adaptiveClassification[neighlist[ii]]);
		}*/
	}
  }
  else {
	  float mindist = 100.f;
	  int minindex = 0;
	  float maxvol = 0.f;
	  int maxvolindex = 0;

	  iterateNeighbors(j) {
		if (arrays.particle_type[j] != 0) continue;
		for (int32_t jj1 = 0, j1 = arrays.neighborList[minindex]; jj1 < arrays.neighborListLength[minindex]; (++jj1), (j1 = arrays.neighborList[minindex + jj1 * arrays.maxNumptcls])) {
			auto df = arrays.position.first[i] - arrays.position.first[j];
			auto ln = math::length3(df);
			if (mindist > ln.val) {
				mindist = ln.val;
				minindex = j;
			}
			if (maxvol < arrays.volume.first[j].val)
			{
				maxvol = arrays.volume.first[j].val;
				maxvolindex = j;
			}
		}
	  }

	  
	//  if (arrays.frame > 10 && arrays.frame < 13)
	//  {
	//	  //printf("FRAME NUM %d\n", arrays.frame);
	//		arrays.refinementRatio[i] = 22.0;
	//		arrays.adaptiveClassification[i] = ceil(math::pow(arrays.refinementRatio[i], 2.0/3.0));
	//		arrays.refinementRatio[i] = arrays.adaptiveClassification[i];
	//		arrays.optimization_group[i] = 0;
	//		return;
	//  }
	//else if (arrays.frame > 20 && arrays.frame < 233)
	//  {
	//	  //printf("MERGING FRAME NUM %d\n", arrays.frame);
	//		arrays.adaptiveClassification[i] = -2.0;
	//		
	//		return;
	//  }
			
	  if (maxvolindex != 0)
	  {
			//auto targetVolume  = level_estimate(arrays, -arrays.distanceBuffer.first[maxvolindex]);
			////float rat = arrays.volume.first[i].val / arrays.volume.first[minindex].val;
			//float rat = arrays.volume.first[i].val / targetVolume.val;
			//if (rat <= 0.5f) arrays.adaptiveClassification[i] = -2.0;
			//else if (rat > 0.5f && rat < 0.99f) arrays.adaptiveClassification[i] = -1.0;
			//else if (rat > 1.99f) arrays.adaptiveClassification[i] = rat != rat ? 0 : rat;
			//else if (rat > 1.1f) arrays.adaptiveClassification[i] = 1;
			//else arrays.adaptiveClassification[i] = 0;
	  }
	  else {
			auto vol = PI4O3 * (arrays.radius * arrays.radius * arrays.radius).val;
			float rat = arrays.volume.first[i].val / vol;
			if (rat <= 0.5) {
				arrays.adaptiveClassification[i] = -2.0;
				//printf("vol %f class %f\n", arrays.volume.first[i].val, arrays.adaptiveClassification[i]);
			}
			else {
				arrays.adaptiveClassification[i] = -1.0;

			}
		}
  }
}

basicFunction(decide, decisionFunction, "Adaptive: classify Particles");
neighFunction(decideRefinement, decisionRefinementFunction, "Adaptive: classify Particles");
basicFunction(initArrayForDec, initArrayForDecision, "Adaptive: classify Particles");
